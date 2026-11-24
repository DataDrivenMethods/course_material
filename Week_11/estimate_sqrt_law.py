#!/usr/bin/env python3
"""
estimate_sqrt_law.py
====================
Estimate the square-root-law coefficient c and the power delta for one stock
from a NASDAQ ITCH 5.0 Parquet store (written by create_ITCH_parquet_store.py).

Model, on intervals of length T:

    y = c * sgn(x) * |x|**delta + noise

    x = dQ / V       net signed shares in the interval / day's volume
    y = r / sigma    log mid return over the interval / day's mid volatility

Steps
  1. Rebuild the top of the order book from the A, F, U, E, C, X, D messages.
  2. Sign the trades (9:30-16:00 only, auction crosses Q excluded):
       E, C  executions against a resting order: sign = minus the resting side
             (a resting sell hit -> buyer-initiated -> +1)
       P     hidden executions: quote rule (above the mid +1, below -1,
             at the mid dropped)
  3. V     = total shares traded 9:30-16:00
     sigma = sqrt(sum of the 390 squared 1-minute log mid returns)
  4. For each T: dQ_j = sum of signed shares, r_j = log mid change -> x_j, y_j.
     Intervals start at 9:30; a remainder at the end of the day is dropped
     (T = 60 gives 6 intervals, 9:30-15:30).
  5. Fit
       (a) c with delta fixed at 1/2: OLS of y on sgn(x) sqrt|x|, no constant,
           HAC (Newey-West) standard error
       (b) c and delta jointly: nonlinear least squares

Usage
    python estimate_sqrt_law.py AAPL --store ../Week_4/itch_parquet
    python estimate_sqrt_law.py AAPL --store ../Week_4/itch_parquet --intervals 1 5 15 --plot
"""

import argparse
from heapq import heappush, heappop
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import statsmodels.api as sm
from scipy.optimize import curve_fit

NS_PER_MIN = 60 * 1_000_000_000
OPEN_NS = (9 * 60 + 30) * NS_PER_MIN       # 9:30, as ns since midnight
CLOSE_NS = 16 * 60 * NS_PER_MIN            # 16:00
N_MINUTES = 390

# event kinds, in the order they are applied when timestamps tie
ADD, REPLACE, EXEC, CANCEL, DELETE = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def stock_locate(store, ticker):
    r = ds.dataset(Path(store) / "msg_type=R", format="parquet")
    r = r.to_table(columns=["stock_locate", "stock"]).to_pandas()
    hit = r[r.stock.str.strip() == ticker]
    if hit.empty:
        raise SystemExit(f"{ticker} not found in the stock directory (R)")
    return int(hit.stock_locate.iloc[0])


def load(store, mtype, loc, columns):
    """All messages of one type for one stock; timestamp as int ns since midnight."""
    d = ds.dataset(Path(store) / f"msg_type={mtype}", format="parquet")
    df = d.to_table(columns=columns, filter=ds.field("stock_locate") == loc).to_pandas()
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def order_events(store, loc):
    """A, F, U, E, C, X, D messages merged into one time-ordered event table."""
    cols = ["timestamp", "kind", "ref", "new_ref", "side", "shares",
            "price", "exec_price", "printable"]
    frames = []

    for m in ("A", "F"):
        df = load(store, m, loc, ["timestamp", "order_reference_number",
                                  "buy_sell_indicator", "shares", "price"])
        frames.append(pd.DataFrame({
            "timestamp": df.timestamp, "kind": ADD,
            "ref": df.order_reference_number, "side": df.buy_sell_indicator,
            "shares": df.shares, "price": df.price}))

    df = load(store, "U", loc, ["timestamp", "original_order_reference_number",
                                "new_order_reference_number", "shares", "price"])
    frames.append(pd.DataFrame({
        "timestamp": df.timestamp, "kind": REPLACE,
        "ref": df.original_order_reference_number,
        "new_ref": df.new_order_reference_number,
        "shares": df.shares, "price": df.price}))

    df = load(store, "E", loc, ["timestamp", "order_reference_number", "executed_shares"])
    frames.append(pd.DataFrame({
        "timestamp": df.timestamp, "kind": EXEC,
        "ref": df.order_reference_number, "shares": df.executed_shares,
        "printable": 1}))

    df = load(store, "C", loc, ["timestamp", "order_reference_number", "executed_shares",
                                "execution_price", "printable"])
    frames.append(pd.DataFrame({
        "timestamp": df.timestamp, "kind": EXEC,
        "ref": df.order_reference_number, "shares": df.executed_shares,
        "exec_price": df.execution_price, "printable": df.printable}))

    df = load(store, "X", loc, ["timestamp", "order_reference_number", "cancelled_shares"])
    frames.append(pd.DataFrame({
        "timestamp": df.timestamp, "kind": CANCEL,
        "ref": df.order_reference_number, "shares": df.cancelled_shares}))

    df = load(store, "D", loc, ["timestamp", "order_reference_number"])
    frames.append(pd.DataFrame({
        "timestamp": df.timestamp, "kind": DELETE, "ref": df.order_reference_number}))

    ev = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    ev = ev.fillna(0).astype("int64")
    # stable sort: by time, ties broken by kind, then by original message order
    order = np.lexsort((ev.kind.values, ev.timestamp.values))
    return ev.iloc[order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

def replay_book(ev):
    """Apply the events in order.

    Returns
      ts, best_bid, best_ask : top of book after each event (prices x 10,000)
      lit                    : executions E, C as (timestamp, sign, shares, price)
      missing                : events whose order reference was not found
    """
    orders = {}                          # ref -> [side, price, shares]
    qty = {1: {}, -1: {}}                # side -> {price: shares}
    heaps = {1: [], -1: []}              # bids stored as -price, asks as price
    n = len(ev)
    best_bid = np.zeros(n, dtype=np.int64)
    best_ask = np.zeros(n, dtype=np.int64)
    lit = []
    missing = 0

    def add(ref, side, shares, price):
        orders[ref] = [side, price, shares]
        book = qty[side]
        q = book.get(price, 0)
        if q <= 0:
            heappush(heaps[side], -price if side == 1 else price)
        book[price] = q + shares

    def remove(ref, shares):
        """Take `shares` off order `ref` (all of it if shares is None)."""
        o = orders.get(ref)
        if o is None:
            return None
        side, price, left = o
        take = left if shares is None else min(shares, left)
        o[2] = left - take
        book = qty[side]
        book[price] = book.get(price, 0) - take
        if book[price] <= 0:
            del book[price]
        if o[2] <= 0:
            del orders[ref]
        return side, price

    def top(side):
        h, book = heaps[side], qty[side]
        while h and book.get(-h[0] if side == 1 else h[0], 0) <= 0:
            heappop(h)
        if not h:
            return 0
        return -h[0] if side == 1 else h[0]

    cols = [ev[c].tolist() for c in ("timestamp", "kind", "ref", "new_ref", "side",
                                     "shares", "price", "exec_price", "printable")]
    for i, (t, kind, ref, new_ref, side, shares, price, xprice, printable) in enumerate(zip(*cols)):
        if kind == ADD:
            add(ref, side, shares, price)
        elif kind == REPLACE:
            o = remove(ref, None)
            if o is None:
                missing += 1
            else:
                add(new_ref, o[0], shares, price)
        elif kind == EXEC:
            o = remove(ref, shares)
            if o is None:
                missing += 1
            elif printable:
                rest_side, rest_price = o
                lit.append((t, -rest_side, shares, xprice if xprice else rest_price))
        elif kind == CANCEL:
            if remove(ref, shares) is None:
                missing += 1
        else:  # DELETE
            if remove(ref, None) is None:
                missing += 1
        best_bid[i] = top(1)
        best_ask[i] = top(-1)

    lit = pd.DataFrame(lit, columns=["timestamp", "sign", "shares", "price"])
    return ev.timestamp.values, best_bid, best_ask, lit, missing


def mid_lookup(ts, best_bid, best_ask):
    """Function: times -> mid (x 10,000) from the last valid book state at or before each time."""
    valid = (best_bid > 0) & (best_ask > 0) & (best_bid < best_ask)
    mid = pd.Series(np.where(valid, (best_bid + best_ask) / 2.0, np.nan)).ffill().values

    def at(times):
        idx = np.searchsorted(ts, times, side="right") - 1
        out = np.full(len(times), np.nan)
        ok = idx >= 0
        out[ok] = mid[idx[ok]]
        return out
    return at


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def power_law(x, c, delta):
    return c * np.sign(x) * np.abs(x) ** delta


def fit(x, y):
    n = len(y)
    z = np.sign(x) * np.sqrt(np.abs(x))
    lags = max(1, int(4 * (n / 100) ** (2 / 9)))
    ols = sm.OLS(y, z).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    c_half, se_half = float(ols.params[0]), float(ols.bse[0])
    try:
        (c, d), cov = curve_fit(power_law, x, y, p0=[c_half, 0.5],
                                bounds=([-np.inf, 0.05], [np.inf, 2.0]))
        se_c, se_d = np.sqrt(np.diag(cov))
    except (RuntimeError, ValueError):
        c = d = se_c = se_d = np.nan
    return dict(n=n, c_half=c_half, se_half=se_half, c=c, se_c=se_c, delta=d, se_delta=se_d)


def interval_data(trades, mid_at, V, sigma, T):
    """x_j and y_j on intervals of T minutes starting at 9:30."""
    n = N_MINUTES // T
    edges = OPEN_NS + np.arange(n + 1) * T * NS_PER_MIN
    dQ, _ = np.histogram(trades.timestamp, bins=edges,
                         weights=trades.sign * trades.shares)
    r = np.diff(np.log(mid_at(edges)))
    return dQ / V, r / sigma


def binned(x, y, nbins=10):
    """Bin means of x and y over equal-count bins of x, and the standard error of mean y."""
    b = pd.qcut(x, min(nbins, len(x)), labels=False, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "b": b}).groupby("b")
    se = (g.y.std() / np.sqrt(g.y.count())).values
    return g.x.mean().values, g.y.mean().values, se


def plot_single(ticker, T, x, y, est, path):
    """One clean figure for a single interval length T."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=surface)
    ax.set_facecolor(surface)

    pct = 100.0                                   # plot x in % of daily volume
    bx, by, bse = binned(x, y)
    gx = np.linspace(x.min(), x.max(), 400)

    ax.scatter(pct * x, y, s=10, color=ink2, alpha=0.18, linewidths=0,
               label=f"{len(x)} intervals", zorder=1)
    ax.plot(pct * gx, power_law(gx, est["c_half"], 0.5), color=blue, lw=2, zorder=3,
            label=rf"$\delta = 1/2$:  $\hat c = {est['c_half']:.2f}$ ({est['se_half']:.2f})")
    if np.isfinite(est["delta"]):
        ax.plot(pct * gx, power_law(gx, est["c"], est["delta"]), color=orange, lw=2,
                ls=(0, (5, 3)), zorder=3,
                label=rf"joint:  $\hat c = {est['c']:.2f}$ ({est['se_c']:.2f})   "
                      rf"$\hat\delta = {est['delta']:.2f}$ ({est['se_delta']:.2f})")
    ax.errorbar(pct * bx, by, yerr=bse, fmt="o", ms=7, color=ink, mec=surface, mew=1.5,
                elinewidth=1.2, capsize=0, zorder=4, label="bin means $\\pm$ 1 s.e.")

    ax.axhline(0, color=ink2, lw=0.6, zorder=0)
    ax.axvline(0, color=ink2, lw=0.6, zorder=0)
    ax.grid(color=grid, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelcolor=ink2)

    ax.set_xlabel(r"Net order flow  $x = \Delta Q / V$  (% of daily volume)", color=ink)
    ax.set_ylabel(r"Mid return  $y = r / \sigma$", color=ink)
    fig.suptitle(f"{ticker}: square-root law, T = {T} min", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=ink)
    ax.set_title(r"$y = c \cdot \mathrm{sgn}(x) \cdot |x|^{\delta}$, standard errors in parentheses",
                 loc="left", fontsize=9.5, color=ink2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ink)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Estimate c and delta in y = c sgn(x)|x|^delta")
    p.add_argument("ticker")
    p.add_argument("--store", required=True, help="path to itch_parquet/")
    p.add_argument("--intervals", type=int, nargs="+", default=[1, 5, 15, 30, 60],
                   help="interval lengths T in minutes")
    p.add_argument("--plot", action="store_true",
                   help="save all T in one figure, plus a single figure for T = 1")
    a = p.parse_args()

    ticker = a.ticker.upper()
    loc = stock_locate(a.store, ticker)
    print(f"{ticker}: stock_locate {loc}")

    # 1. order book
    ev = order_events(a.store, loc)
    ts, bb, ba, lit, missing = replay_book(ev)
    mid_at = mid_lookup(ts, bb, ba)
    print(f"  order events {len(ev):,}   unmatched references {missing:,}")

    # 2. signed trades, 9:30-16:00
    hid = load(a.store, "P", loc, ["timestamp", "shares", "price"])
    hid_mid = mid_at(hid.timestamp.values)
    hid["sign"] = np.sign(hid.price.values - hid_mid).astype(float)
    hid.loc[np.isnan(hid_mid), "sign"] = 0.0
    trades = pd.concat([lit, hid[["timestamp", "sign", "shares", "price"]]], ignore_index=True)
    trades = trades[(trades.timestamp >= OPEN_NS) & (trades.timestamp < CLOSE_NS)]

    # 3. daily normalisers
    V = trades.shares.sum()                         # all trades, signed or not
    trades = trades[trades.sign != 0]
    grid = OPEN_NS + np.arange(N_MINUTES + 1) * NS_PER_MIN
    sigma = np.sqrt(np.nansum(np.diff(np.log(mid_at(grid))) ** 2))
    print(f"  trades {len(trades):,} (lit {len(lit):,}  hidden {len(hid):,})")
    print(f"  V = {V:,.0f} shares   sigma = {1e4 * sigma:.1f} bps")

    # 4-5. per interval length
    rows, data = [], {}
    for T in a.intervals:
        x, y = interval_data(trades, mid_at, V, sigma, T)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        data[T] = (x, y)
        rows.append(dict(T=T, **fit(x, y)))

    out = pd.DataFrame(rows).set_index("T")
    print("\n  delta = 1/2 fixed: c_half (se_half)    joint fit: c (se_c), delta (se_delta)\n")
    print(out.to_string(float_format=lambda v: f"{v:8.3f}"))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(a.intervals)
        fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.2), squeeze=False)
        for ax, T in zip(axes[0], a.intervals):
            x, y = data[T]
            bx, by, _ = binned(x, y)
            grid_x = np.linspace(x.min(), x.max(), 200)
            ax.scatter(x, y, s=4, alpha=0.25, color="gray")
            ax.plot(bx, by, "o", color="black", label="bin means")
            ax.plot(grid_x, power_law(grid_x, out.loc[T, "c_half"], 0.5), label=r"$\delta = 1/2$")
            if np.isfinite(out.loc[T, "delta"]):
                ax.plot(grid_x, power_law(grid_x, out.loc[T, "c"], out.loc[T, "delta"]),
                        "--", label=rf"$\delta = {out.loc[T, 'delta']:.2f}$")
            ax.axhline(0, lw=0.5, color="k")
            ax.axvline(0, lw=0.5, color="k")
            ax.set_title(f"{ticker}  T = {T} min")
            ax.set_xlabel(r"$x = \Delta Q / V$")
        axes[0][0].set_ylabel(r"$y = r / \sigma$")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        path = f"sqrt_law_{ticker}.png"
        fig.savefig(path, dpi=150)
        print(f"\n  plot saved to {path}")

        T0 = 1 if 1 in a.intervals else min(a.intervals)
        path = f"sqrt_law_{ticker}_T{T0}.png"
        plot_single(ticker, T0, *data[T0], out.loc[T0], path)
        print(f"  plot saved to {path}")


if __name__ == "__main__":
    main()
