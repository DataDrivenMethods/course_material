"""
vwap_ic.py
==========
Compute IC of the VWAP-deviation signal against forward returns, using
minute bars from minute_bars.h5, over a grid of
  signal lookback   N bars in the rolling VWAP        (--windows)
  forward horizon   h bars ahead for the ex-post return (--periods)
Both default to 1, 2, 5, 10, 15, 25, 50, 100 bars. Output: one
lookback × horizon IC grid per symbol, then the grid averaged across symbols.

Two signal modes:
  cumulative  – deviation from intraday cumulative VWAP (resets at open).
                Tends to drift with time-of-day; use for reference only.
  rolling     – deviation from rolling N-bar VWAP (default N=30).
                Stationary within the day; preferred for IC analysis.

mean_ic() is adapted from ../Week_6/IC_study.py (same signature/logic).
We cannot import that module directly because it runs WRDS connection code
at module level.

Usage
-----
    python vwap_ic.py
    python vwap_ic.py --windows 5 15 30 --periods 1 5 15 30
    python vwap_ic.py --signal cumulative          # no lookback: one row per grid
    python vwap_ic.py --symbols AAPL MSFT
    python vwap_ic.py --h5 data/minute_bars.h5 --method pearson
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_PATH = Path("data/minute_bars.h5")
PERIODS  = (1, 2, 5, 10, 15, 25, 50, 100)   # forward horizons h (bars)
WINDOWS  = (1, 2, 5, 10, 15, 25, 50, 100)   # signal lookbacks N (bars)
FREQ     = "1min"

# ---------------------------------------------------------------------------
# mean_ic — adapted verbatim from ../Week_6/IC_study.py
# ---------------------------------------------------------------------------

def mean_ic(symbol_data, signal_col, fwd_periods=PERIODS, freq=FREQ, method="spearman"):
    """Compute mean IC for a signal across all symbols.

    Parameters
    ----------
    symbol_data : dict  {sym: (xm, midprice_s)}
        xm          – DataFrame with signal column(s), timestamp index
        midprice_s  – Series of prices, timestamp index
    signal_col  : str   column name inside xm
    fwd_periods : tuple of ints  look-ahead horizons (in bars at *freq*)
    freq        : str   resample frequency (default '1min')
    method      : 'spearman' or 'pearson'

    Returns
    -------
    dict {period: mean_ic_float}
    """
    ics = {p: [] for p in fwd_periods}
    for sym, (xm, midprice_s) in symbol_data.items():
        sig = xm[signal_col].resample(freq).mean().dropna()
        mid = midprice_s.resample(freq).last().ffill()
        for p in fwd_periods:
            fwd = mid.pct_change(p).shift(-p)
            df  = pd.concat([sig, fwd], axis=1).dropna()
            if len(df) < 10:
                continue
            if method == "spearman":
                ic, _ = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
            else:
                ic = df.iloc[:, 0].corr(df.iloc[:, 1])
            ics[p].append(ic)
    return {p: (np.mean(v) if v else np.nan) for p, v in ics.items()}

# ---------------------------------------------------------------------------
# Signal: VWAP deviation (cumulative or rolling)
# ---------------------------------------------------------------------------

def to_minute_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Put bars on the full 09:30-15:59 clock-minute grid.

    Minutes with no trades get zero volume and carry the last close forward,
    so rolling windows and shifts count clock minutes, not bars.
    """
    day = bars.index[0].normalize()
    minutes = pd.date_range(day + pd.Timedelta("9h30min"), day + pd.Timedelta("15h59min"), freq="1min")
    grid = bars.reindex(minutes)
    grid[["volume", "dollar_volume"]] = grid[["volume", "dollar_volume"]].fillna(0)
    grid["close"] = grid["close"].ffill()
    return grid


def add_vwap_dev(bars: pd.DataFrame, mode: str = "rolling", window: int = 30) -> pd.DataFrame:
    """Add vwap_dev column to a single-symbol bar DataFrame.

    mode='cumulative'
        vwap_cum = Σ dollar_volume / Σ volume since open (resets each day).
        Drifts with time-of-day; spurious at long horizons on a single day.

    mode='rolling'
        vwap_roll = rolling N-bar dollar_volume / volume.
        Stationary within the day; preferred for IC analysis.
    """
    b = bars.copy()
    if mode == "cumulative":
        b["vwap_ref"] = b["dollar_volume"].cumsum() / b["volume"].cumsum()
    else:
        b["vwap_ref"] = (b["dollar_volume"].rolling(window).sum() /
                         b["volume"].rolling(window).sum())
    b["vwap_dev"] = (b["close"] - b["vwap_ref"]) / b["vwap_ref"]
    return b

# ---------------------------------------------------------------------------
# Load & build symbol_data
# ---------------------------------------------------------------------------

def load_symbol_data(h5path: Path, symbols: list[str],
                     mode: str = "rolling", window: int = 30,
                     verbose: bool = True) -> dict:
    """Return symbol_data dict compatible with mean_ic()."""
    symbol_data = {}
    with pd.HDFStore(str(h5path), mode="r") as store:
        available = [k.split("/")[-1] for k in store.keys()]
        for sym in symbols:
            if sym not in available:
                print(f"  ⚠  {sym} not in {h5path}  (available: {available})")
                continue
            raw  = store[f"minute_bars/{sym}"]
            bars = add_vwap_dev(to_minute_grid(raw), mode=mode, window=window)
            # signal only in minutes that traded; prices on the full minute grid
            xm         = bars.loc[raw.index, ["vwap_dev"]].dropna()
            midprice_s = bars["close"].rename(sym)
            symbol_data[sym] = (xm, midprice_s)
            if verbose:
                print(f"  Loaded {sym}: {len(bars)} bars  (signal mode={mode})")
    return symbol_data

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ic_grid(h5path: Path, symbols: list[str], mode: str, windows, periods,
            method: str) -> pd.DataFrame:
    """IC for every (symbol, lookback N, horizon h).

    Returns DataFrame indexed by (symbol, lookback) with one column per h.
    In cumulative mode there is no lookback; the row is labelled 'cum'.
    """
    rows = {}
    for n in (windows if mode == "rolling" else ["cum"]):
        symbol_data = load_symbol_data(h5path, symbols, mode=mode,
                                       window=n if mode == "rolling" else 0,
                                       verbose=False)
        for sym, data in symbol_data.items():
            rows[(sym, n)] = mean_ic({sym: data}, "vwap_dev",
                                     fwd_periods=periods, freq=FREQ, method=method)
    grid = pd.DataFrame(rows).T
    grid.index.names = ["symbol", "lookback"]
    return grid[list(periods)]

# ---------------------------------------------------------------------------
# Cross-sectional IC and ICIR
# ---------------------------------------------------------------------------

MIN_STOCKS = 20     # minutes with fewer stocks are skipped


def cross_sectional_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Spearman IC across stocks at each minute: one IC_t per row.

    signal, fwd_ret: index = minute, columns = symbol.
    """
    ok = signal.notna() & fwd_ret.notna()
    s = signal.where(ok).rank(axis=1)
    r = fwd_ret.where(ok).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    r = r.sub(r.mean(axis=1), axis=0)
    ic_t = (s * r).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (r ** 2).sum(axis=1))
    return ic_t[ok.sum(axis=1) >= MIN_STOCKS].dropna()


def icir_stats(ic_t: pd.Series, h: int) -> dict:
    """Mean IC_t, ICIR = mean / std over minutes, and an overlap-adjusted t-stat.

    Forward returns over h minutes overlap, so only about T/h of the T minutes
    are independent: t = ICIR * sqrt(T / h).
    """
    icir = ic_t.mean() / ic_t.std()
    return {"mean": ic_t.mean(), "icir": icir, "t": icir * np.sqrt(len(ic_t) / h)}


def icir_grid(h5path: Path, symbols: list[str], mode: str, windows, periods) -> dict:
    """Cross-sectional mean IC, ICIR and t for every (lookback N, horizon h).

    Returns {"mean": grid, "icir": grid, "t": grid}, each lookback × horizon.
    """
    stats = {}
    for n in (windows if mode == "rolling" else ["cum"]):
        symbol_data = load_symbol_data(h5path, symbols, mode=mode,
                                       window=n if mode == "rolling" else 0,
                                       verbose=False)
        signal = pd.DataFrame({sym: xm["vwap_dev"] for sym, (xm, _) in symbol_data.items()})
        close = pd.DataFrame({sym: px for sym, (_, px) in symbol_data.items()})
        for h in periods:
            fwd_ret = close.shift(-h) / close - 1
            stats[(n, h)] = icir_stats(cross_sectional_ic(signal, fwd_ret), h)
    df = pd.DataFrame(stats).T
    return {key: df[key].unstack()[list(periods)] for key in ("mean", "icir", "t")}


def print_grid(grid: pd.DataFrame, title: str, fmt: str = "+9.4f"):
    """Print a lookback × horizon grid with labelled axes."""
    print(f"\n{title}")
    print("  rows: signal lookback N (bars)   columns: forward horizon h (bars)")
    cols = "".join(f"{f'h={h}':>9s}" for h in grid.columns)
    print(f"{'N':>6s} {cols}")
    print("─" * (7 + 9 * len(grid.columns)))
    for n, row in grid.iterrows():
        label = f"N={n}" if n != "cum" else "cum"
        print(f"{label:>6s} " + "".join(format(v, fmt) for v in row))


def main():
    parser = argparse.ArgumentParser(description="VWAP-deviation IC study on minute bars.")
    parser.add_argument("--h5",      type=Path,  default=H5_PATH)
    parser.add_argument("--symbols", nargs="+",  default=None,
                        help="Symbols to include (default: all in HDF5)")
    parser.add_argument("--periods", nargs="+",  type=int, default=list(PERIODS),
                        help="Forward-return horizons h, in bars")
    parser.add_argument("--windows", nargs="+",  type=int, default=list(WINDOWS),
                        help="Signal lookbacks N: bars in the rolling VWAP (only with --signal rolling)")
    parser.add_argument("--method",  default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--signal",  default="rolling",  choices=["rolling", "cumulative"],
                        help="VWAP reference: rolling N-bar window or cumulative from open")
    args = parser.parse_args()

    # Discover available symbols if not specified
    if args.symbols is None:
        with pd.HDFStore(str(args.h5), mode="r") as store:
            args.symbols = [k.split("/")[-1] for k in store.keys()]

    periods = sorted(args.periods)
    windows = sorted(args.windows)
    print(f"\nLoading minute bars from {args.h5}")
    grid = ic_grid(args.h5, args.symbols, args.signal, windows, periods, args.method)
    if grid.empty:
        print("No symbols loaded. Exiting.")
        return
    symbols = list(dict.fromkeys(grid.index.get_level_values("symbol")))

    if args.signal == "rolling":
        vwap_def = "VWAP over the last N bars, t-N+1..t"
        lookback = ", ".join(str(n) for n in windows) + " bars"
    else:
        vwap_def = "VWAP over all bars since the open, 09:30..t"
        lookback = "none (cumulative since open)"
    corr = "Spearman rank" if args.method == "spearman" else "Pearson"

    print(f"""
Definitions  (1 bar = 1 minute)
───────────
  Signal (known at bar t)    vwap_dev_t = (close_t - VWAP_t) / VWAP_t,  {vwap_def}
  Signal lookback N          {lookback}
  Forward return (ex post)   fwd_ret_t(h) = close_(t+h) / close_t - 1
  Forward horizon h          {", ".join(str(h) for h in periods)} bars
  IC(symbol, N, h)           {corr} correlation of vwap_dev_t with fwd_ret_t(h) over all
                             minutes t of the day for that symbol (time series)
  Mean IC(N, h)              average of IC(symbol, N, h) across the {len(symbols)} symbols
  Cross-sectional IC_t       Spearman correlation ACROSS stocks at minute t of vwap_dev_t
                             with fwd_ret_t(h): one IC per minute (needs >= {MIN_STOCKS} stocks)
  ICIR(N, h)                 mean(IC_t) / std(IC_t) over the minutes of the day
  t(N, h)                    ICIR * sqrt(T / h): T minutes, of which ~T/h are independent
  Reading the sign           IC < 0: price above VWAP tends to fall back (mean reversion)
                             IC > 0: price above VWAP tends to keep rising (momentum)
  Caveat                     one day = 390 bars; large N and h leave few, heavily
                             overlapping observations, so those cells are noisy""")

    for sym in symbols:
        print_grid(grid.loc[sym], f"IC — {sym}")

    print_grid(grid.groupby(level="lookback", sort=False).mean(),
               f"Mean IC across {len(symbols)} symbols")

    xs = icir_grid(args.h5, symbols, args.signal, windows, periods)
    print_grid(xs["mean"], "Cross-sectional IC: mean of IC_t over minutes")
    print_grid(xs["icir"], "ICIR: mean(IC_t) / std(IC_t)")
    print_grid(xs["t"], "t-stat: ICIR * sqrt(T / h)", fmt="+9.2f")

if __name__ == "__main__":
    main()
