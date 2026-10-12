"""Context manager for WRDS database connections.

    from wrds_context import WRDSConnection

    with WRDSConnection('cpehlivanian') as conn:
        df = conn.raw_sql("SELECT * FROM crsp.dsf_v2 LIMIT 10")

`__enter__` hands back the underlying ``wrds.Connection``, so everything inside
the block is ordinary WRDS API usage.  Only the username is passed; ``wrds``
reads the password from ~/.pgpass itself.  The connection is closed when the block
exits, including when it exits via an exception, ``return``, ``break``, or
Ctrl-C.

The object is also safe to use without a ``with`` block -- call ``connect()``
and ``close()`` yourself -- which is the convenient form in a notebook, where
you want one connection alive across many cells.  In that case a finalizer
still closes the socket if you forget: it fires when the object is garbage
collected or, failing that, at interpreter exit.  Nothing can close it if the
process dies without unwinding (SIGKILL, ``os._exit``, a Jupyter kernel
restart), so a long-lived kernel is still the place to be deliberate about it.
"""

from __future__ import annotations

import weakref
from typing import Optional

import wrds


def _quietly_close(conn) -> None:
    """Close a connection without ever raising.

    This runs from ``__exit__`` and from the finalizer.  A dead or already
    closed socket must not raise there: inside ``__exit__`` it would replace
    whatever exception was propagating out of the ``with`` block, and inside a
    finalizer during interpreter shutdown it would just print noise to stderr.
    """
    try:
        conn.close()
    except Exception:
        pass


class WRDSConnection:
    """A ``wrds.Connection`` with a lifetime you can reason about.

    Args:
        username: WRDS username.  The password is not passed here; ``wrds``
            picks it up from ~/.pgpass.
    """

    def __init__(self, username: str):
        self.username = username
        self.conn: Optional[wrds.Connection] = None
        self._finalizer: Optional[weakref.finalize] = None
        self._depth = 0

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> wrds.Connection:
        """Open the connection, or return the one already open."""
        if self.conn is None:
            conn = wrds.Connection(wrds_username=self.username)
            self.conn = conn
            # weakref.finalize, not atexit: it fires at garbage collection AND
            # at interpreter exit, and -- unlike atexit.register(self.close) --
            # it holds no reference to self, so it does not keep this object
            # (and its socket) alive for the life of the process.  Calling a
            # finalize object more than once is a no-op, which is what makes
            # close() idempotent.
            self._finalizer = weakref.finalize(self, _quietly_close, conn)
        return self.conn

    def close(self) -> None:
        """Close the connection.  Safe to call repeatedly, and never raises."""
        finalizer, self._finalizer = self._finalizer, None
        self.conn = None
        self._depth = 0
        if finalizer is not None:
            finalizer()          # dispatches to _quietly_close exactly once

    @property
    def is_open(self) -> bool:
        return self.conn is not None

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> wrds.Connection:
        # Count entries so a nested `with` on the same object does not close
        # the connection out from under the outer block.
        self._depth += 1
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._depth -= 1
        if self._depth <= 0:
            self.close()
        return False             # never swallow an exception

    def __repr__(self) -> str:
        state = "open" if self.is_open else "closed"
        return f"<WRDSConnection {self.username!r} {state}>"
