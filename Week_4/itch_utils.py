"""Inspection utilities for the NASDAQ ITCH 5.0 HDF5 store written by
create_ITCH_HDF5_store.py.

The store holds one table per ITCH message type, keyed by the single-letter
type code, with `stock_locate` indexed as a data column on every table.
Symbols themselves live only in the `R` (stock directory) table, so almost
everything here starts by mapping ticker <-> stock_locate.

Two conventions inherited from the parser, worth remembering:
  * prices are integers scaled by 10,000 (96800 -> $9.68)
  * timestamps are timedeltas since midnight ET (nanosecond resolution)

The store is assumed to be `itch.h5` in the current working directory, so run
from wherever it lives. Point elsewhere with `store=...` in library calls or
`--store` on the CLI.

Library use:
    import itch_utils as iu
    iu.store_info()
    iu.trade_counts(['AAPL', 'MSFT'])
    iu.trade_counts().head(25)          # all symbols, ranked
    tape = iu.trade_tape('AAPL')

CLI use:
    python itch_utils.py info
    python itch_utils.py symbols --pattern '^AA'
    python itch_utils.py counts --top 25
    python itch_utils.py counts AAPL MSFT TSLA
    python itch_utils.py tape AAPL --out aapl_trades.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Resolved against the current working directory, not the script's location,
# so it follows you between the notes directory and wherever the store lives.
# Override per call with store=..., or on the CLI with --store.
DEFAULT_STORE = Path('itch.h5')

PRICE_SCALE = 10_000

# regular trading hours, as offsets from midnight ET
MARKET_OPEN = pd.Timedelta('09:30:00')
MARKET_CLOSE = pd.Timedelta('16:00:00')

MESSAGE_LABELS = {
    'S': 'system event',
    'R': 'stock directory',
    'H': 'stock trading action',
    'Y': 'reg sho restriction',
    'L': 'market participant position',
    'V': 'mwcb decline level',
    'W': 'mwcb status',
    'K': 'ipo quoting period update',
    'J': 'luld auction collar',
    'h': 'operational halt',
    'A': 'add order (no attribution)',
    'F': 'add order (attributed)',
    'E': 'order executed',
    'C': 'order executed with price',
    'X': 'order cancel',
    'D': 'order delete',
    'U': 'order replace',
    'P': 'trade (non-cross)',
    'Q': 'cross trade',
    'B': 'broken trade',
    'I': 'noii / imbalance',
    'N': 'retail price improvement',
}

# Message types that represent an actual execution.
#   E - execution against a displayed order; price lives on the resting order
#   C - execution at a price other than the order's (carries execution_price)
#   P - non-displayable / hidden trade (carries price)
#   Q - cross trade: opening, closing, halt auctions (carries cross_price)
TRADE_TYPES = ('E', 'C', 'P', 'Q')

# per trade message type: (share column, price column or None)
_TRADE_COLS = {
    'E': ('executed_shares', None),
    'C': ('executed_shares', 'execution_price'),
    'P': ('shares', 'price'),
    'Q': ('shares', 'cross_price'),
}


# --------------------------------------------------------------------------
# store-level inspection
# --------------------------------------------------------------------------

def _open(store=DEFAULT_STORE):
    """Open the store read-only, with a useful message when the path is wrong."""
    path = Path(store)
    if not path.exists():
        raise FileNotFoundError(
            f'no HDF5 store at {path.resolve()}\n'
            f'(cwd is {Path.cwd()}; pass store=... or --store to point elsewhere)')
    return pd.HDFStore(path, mode='r')


def store_info(store=DEFAULT_STORE) -> pd.DataFrame:
    """One row per table: message type, human label, row count, columns."""
    rows = []
    with _open(store) as s:
        for key in s.keys():
            mtype = key.lstrip('/')
            storer = s.get_storer(key)
            rows.append({
                'type': mtype,
                'message': MESSAGE_LABELS.get(mtype, '?'),
                'rows': storer.nrows,
                'columns': ', '.join(s.select(key, start=0, stop=1).columns),
            })
    df = pd.DataFrame(rows).set_index('type').sort_values('rows', ascending=False)
    return df


def session_events(store=DEFAULT_STORE) -> pd.DataFrame:
    """The `S` table: market open/close boundaries for the session."""
    codes = {'O': 'start of messages', 'S': 'start of system hours',
             'Q': 'start of market hours', 'M': 'end of market hours',
             'E': 'end of system hours', 'C': 'end of messages'}
    with _open(store) as s:
        df = s.select('S')[['timestamp', 'event_code']]
    df['event'] = df.event_code.map(codes)
    return df.sort_values('timestamp').reset_index(drop=True)


# --------------------------------------------------------------------------
# symbol <-> stock_locate
# --------------------------------------------------------------------------

def stock_directory(store=DEFAULT_STORE) -> pd.DataFrame:
    """The `R` table, indexed by stock_locate.

    A symbol is occasionally re-broadcast during the session (HERO in the
    2019-10-30 file), so drop duplicate locates and keep the latest record.
    """
    with _open(store) as s:
        df = s.select('R')
    df = df.sort_values('timestamp').drop_duplicates('stock_locate', keep='last')
    return df.set_index('stock_locate').sort_index()


def symbol_map(store=DEFAULT_STORE) -> pd.Series:
    """stock_locate -> ticker."""
    return stock_directory(store).stock


def locates(symbols, store=DEFAULT_STORE) -> dict:
    """ticker -> stock_locate for the given symbols. Raises on unknown tickers."""
    if isinstance(symbols, str):
        symbols = [symbols]
    smap = symbol_map(store)
    inverse = pd.Series(smap.index.values, index=smap.values)
    wanted = [s.strip().upper() for s in symbols]
    missing = [s for s in wanted if s not in inverse.index]
    if missing:
        raise KeyError(f'not in the stock directory: {missing}')
    return {s: int(inverse[s]) for s in wanted}


def _where_clause(symbols, store):
    """Build a `where=` predicate on the indexed stock_locate column."""
    if symbols is None:
        return None, None
    loc = locates(symbols, store)
    return f'stock_locate in {list(loc.values())}', loc


# --------------------------------------------------------------------------
# trade counting
# --------------------------------------------------------------------------

def trade_counts(symbols=None, kinds=TRADE_TYPES, store=DEFAULT_STORE,
                 drop_broken=True, rth=False, chunksize=5_000_000) -> pd.DataFrame:
    """Number of executions and shares traded per symbol.

    symbols : list of tickers, or None for every symbol in the session.
    kinds   : which execution message types to count. The default counts all
              four. Note the course notebook's trade summary uses only
              ('P', 'Q'), which captures hidden and auction prints but misses
              the ~7.4M executions against displayed orders in E and C --
              i.e. the large majority of the tape.
    rth     : restrict to 09:30-16:00 rather than the full 04:00-20:00 feed.

    Returns a frame indexed by ticker with one count column per message type,
    plus `trades` (total prints) and `shares`.
    """
    where, _ = _where_clause(symbols, store)
    smap = symbol_map(store)

    counts, shares = {}, {}
    with _open(store) as s:
        available = {k.lstrip('/') for k in s.keys()}
        broken = set()
        if drop_broken and 'B' in available:
            broken = set(s.select('B', columns=['match_number']).match_number)

        for kind in kinds:
            if kind not in available:
                continue
            share_col, _ = _TRADE_COLS[kind]
            cols = ['stock_locate', share_col, 'match_number']
            if rth:
                cols.append('timestamp')
            c_acc = pd.Series(dtype='int64')
            s_acc = pd.Series(dtype='int64')
            for chunk in s.select(kind, columns=cols, where=where,
                                  chunksize=chunksize):
                if broken:
                    chunk = chunk[~chunk.match_number.isin(broken)]
                if rth:
                    chunk = chunk[(chunk.timestamp >= MARKET_OPEN)
                                  & (chunk.timestamp < MARKET_CLOSE)]
                g = chunk.groupby('stock_locate')[share_col]
                c_acc = c_acc.add(g.size(), fill_value=0)
                s_acc = s_acc.add(g.sum(), fill_value=0)
            counts[kind] = c_acc
            shares[kind] = s_acc

    out = pd.DataFrame(counts).fillna(0).astype('int64')
    out['trades'] = out.sum(axis=1)
    out['shares'] = pd.DataFrame(shares).fillna(0).sum(axis=1).astype('int64')
    out.index = out.index.map(smap)
    out.index.name = 'stock'
    return out.sort_values('trades', ascending=False)


def message_counts(symbols, store=DEFAULT_STORE) -> pd.DataFrame:
    """Full message-type breakdown per symbol (adds, cancels, deletes, ...).

    Requires a symbol subset -- an all-symbol version would scan every table,
    including the 127M-row `A` table.
    """
    where, loc = _where_clause(symbols, store)
    if where is None:
        raise ValueError('message_counts requires an explicit symbol list')
    smap = symbol_map(store)

    frames = {}
    with _open(store) as s:
        for key in s.keys():
            mtype = key.lstrip('/')
            if mtype in ('R', 'S', 'V', 'W'):        # not per-symbol order flow
                continue
            df = s.select(key, columns=['stock_locate'], where=where)
            if len(df):
                frames[mtype] = df.groupby('stock_locate').size()

    out = pd.DataFrame(frames).fillna(0).astype('int64')
    out.index = out.index.map(smap)
    out.index.name = 'stock'
    out = out.reindex(columns=sorted(out.columns))
    out['total'] = out.sum(axis=1)
    return out


# --------------------------------------------------------------------------
# trade tape
# --------------------------------------------------------------------------

def _order_prices(where, store):
    """order_reference_number -> price, for one symbol subset.

    E messages carry no price -- the execution happens at the resting order's
    price. Rebuild that lookup from the add-order messages (A, F) plus the
    replace messages (U), which re-price an order under a new reference number.

    `stock_locate` must stay in the selected columns: past a handful of
    symbols pandas stops pushing the `where` down into the PyTables index and
    post-filters the returned frame instead, which fails if the filtered
    field was projected away.
    """
    parts = []
    with _open(store) as s:
        for key, ref in (('A', 'order_reference_number'),
                         ('F', 'order_reference_number'),
                         ('U', 'new_order_reference_number')):
            if '/' + key not in s.keys():
                continue
            df = s.select(key, columns=['stock_locate', ref, 'price'],
                          where=where)
            parts.append(df[[ref, 'price']].rename(columns={ref: 'ref'}))
    if not parts:
        return pd.Series(dtype='float64')
    allrefs = pd.concat(parts, ignore_index=True)
    return allrefs.drop_duplicates('ref', keep='last').set_index('ref').price


def trade_tape(symbols, store=DEFAULT_STORE, kinds=TRADE_TYPES,
               drop_broken=True, rth=False, scale_price=True) -> pd.DataFrame:
    """Time-ordered tape of executions for the given symbols.

    Columns: stock, timestamp, kind, shares, price, value, match_number.
    E prints get their price by joining back to the resting order (A/F/U),
    so this is restricted to a symbol subset -- the join is cheap per symbol
    but would mean scanning ~250M rows for the whole session.
    """
    where, loc = _where_clause(symbols, store)
    if where is None:
        raise ValueError('trade_tape requires an explicit symbol list')
    smap = symbol_map(store)

    frames = []
    with _open(store) as s:
        available = {k.lstrip('/') for k in s.keys()}
        broken = set()
        if drop_broken and 'B' in available:
            broken = set(s.select('B', columns=['match_number']).match_number)

        need_ref = 'E' in kinds and 'E' in available
        prices = _order_prices(where, store) if need_ref else None

        for kind in kinds:
            if kind not in available:
                continue
            share_col, price_col = _TRADE_COLS[kind]
            cols = ['stock_locate', 'timestamp', share_col, 'match_number']
            if price_col:
                cols.append(price_col)
            if kind == 'E':
                cols.append('order_reference_number')
            df = s.select(kind, columns=cols, where=where)
            if not len(df):
                continue
            df = df.rename(columns={share_col: 'shares'})
            if price_col:
                df['price'] = df[price_col]
            else:
                df['price'] = df.order_reference_number.map(prices)
            df['kind'] = kind
            frames.append(df[['stock_locate', 'timestamp', 'kind',
                              'shares', 'price', 'match_number']])

    if not frames:
        return pd.DataFrame(columns=['stock', 'timestamp', 'kind', 'shares',
                                     'price', 'value', 'match_number'])

    tape = pd.concat(frames, ignore_index=True)
    if broken:
        tape = tape[~tape.match_number.isin(broken)]
    if rth:
        tape = tape[(tape.timestamp >= MARKET_OPEN)
                    & (tape.timestamp < MARKET_CLOSE)]
    tape['stock'] = tape.stock_locate.map(smap)
    if scale_price:
        tape['price'] = tape.price / PRICE_SCALE
    tape['value'] = tape.shares * tape.price
    tape = tape.sort_values(['stock', 'timestamp']).reset_index(drop=True)
    return tape[['stock', 'timestamp', 'kind', 'shares', 'price',
                 'value', 'match_number']]


def trade_summary(symbols, store=DEFAULT_STORE, **kw) -> pd.DataFrame:
    """Per-symbol trade count, shares, notional, VWAP, and price range."""
    tape = trade_tape(symbols, store=store, **kw)
    g = tape.groupby('stock')
    out = pd.DataFrame({
        'trades': g.size(),
        'shares': g.shares.sum(),
        'notional': g.value.sum(),
        'vwap': g.value.sum() / g.shares.sum(),
        'first': g.price.first(),
        'last': g.price.last(),
        'low': g.price.min(),
        'high': g.price.max(),
    })
    return out.sort_values('notional', ascending=False)


def resample_tape(tape, freq='5min') -> pd.DataFrame:
    """OHLCV bars from a tape produced by trade_tape().

    Bins are floored against midnight rather than against the session's first
    print, so edges land on clean 09:30 / 09:35 / ... boundaries.
    (`resample(origin=...)` is ignored for a timedelta index, hence the
    explicit floor.) Empty bins are omitted rather than forward-filled.
    """
    df = tape.copy()
    df['bar'] = df.timestamp.dt.floor(freq)
    bars = df.sort_values('timestamp').groupby(['stock', 'bar']).agg(
        open=('price', 'first'), high=('price', 'max'),
        low=('price', 'min'), close=('price', 'last'),
        volume=('shares', 'sum'), trades=('price', 'size'),
        notional=('value', 'sum'))
    bars['vwap'] = bars.notional / bars.volume
    return bars


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--store', default=str(DEFAULT_STORE), help='path to itch.h5')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('info', help='tables, row counts, columns')
    sub.add_parser('session', help='system event timeline')

    sp = sub.add_parser('symbols', help='list symbols in the stock directory')
    sp.add_argument('--pattern', help='regex filter on ticker')

    sc = sub.add_parser('counts', help='trades per symbol')
    sc.add_argument('symbols', nargs='*', help='tickers; omit for all symbols')
    sc.add_argument('--kinds', default=''.join(TRADE_TYPES),
                    help='execution message types to count (default ECPQ)')
    sc.add_argument('--top', type=int, default=50)
    sc.add_argument('--rth', action='store_true',
                    help='regular hours 09:30-16:00 only (default: full feed)')
    sc.add_argument('--out', help='write full result to CSV')

    sm = sub.add_parser('msgcounts', help='full message breakdown per symbol')
    sm.add_argument('symbols', nargs='+')

    st = sub.add_parser('tape', help='execution tape for symbols')
    st.add_argument('symbols', nargs='+')
    st.add_argument('--out', help='write to CSV')
    st.add_argument('--bars', help='resample to OHLCV at this freq, e.g. 5min')
    st.add_argument('--rth', action='store_true', help='regular hours only')

    ss = sub.add_parser('summary', help='trades/shares/notional/VWAP per symbol')
    ss.add_argument('symbols', nargs='+')
    ss.add_argument('--rth', action='store_true', help='regular hours only')

    a = p.parse_args()
    store = a.store
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 40)

    if a.cmd == 'info':
        print(store_info(store).to_string())
    elif a.cmd == 'session':
        print(session_events(store).to_string())
    elif a.cmd == 'symbols':
        d = stock_directory(store)[['stock', 'market_category', 'round_lot_size']]
        if a.pattern:
            d = d[d.stock.str.match(a.pattern)]
        print(d.to_string())
        print(f'\n{len(d)} symbols')
    elif a.cmd == 'counts':
        df = trade_counts(a.symbols or None, kinds=tuple(a.kinds), store=store,
                          rth=a.rth)
        print(df.head(a.top).to_string())
        print(f'\n{len(df)} symbols, {df.trades.sum():,} trades, '
              f'{df.shares.sum():,} shares')
        if a.out:
            df.to_csv(a.out)
            print(f'wrote {a.out}')
    elif a.cmd == 'msgcounts':
        print(message_counts(a.symbols, store).to_string())
    elif a.cmd == 'tape':
        t = trade_tape(a.symbols, store, rth=a.rth)
        if a.bars:
            t = resample_tape(t, a.bars)
        print(t.head(50).to_string())
        print(f'\n{len(t):,} rows')
        if a.out:
            t.to_csv(a.out, index=bool(a.bars))
            print(f'wrote {a.out}')
    elif a.cmd == 'summary':
        print(trade_summary(a.symbols, store, rth=a.rth).to_string())


if __name__ == '__main__':
    _main()
