"""
flow_ic_mi.py
=============
IC and normalised mutual information of order-flow imbalance against forward returns,
using minute bars from data/minute_bars.h5.

Signals (lookback k bars), each scored against every stock's forward return:
    own_flow   flow imbalance of the stock itself
    spy_gap    r_SPY,k - r_i,k: SPY return minus the stock's return over the
               last k bars. Lead-lag: if SPY moved and the stock has not yet,
               the stock should catch up, so expect a positive IC.
    spy_resid  β_i·r_SPY,k - r_i,k: as spy_gap but beta-adjusted, so it measures
               the stock's move relative to what the market move implies.
               β_i is a rolling OLS beta of 1-min returns (past BETA_WINDOW bars).

    flow_imb_k = Σ(n_at_ask - n_at_bid) / Σ(n_at_ask + n_at_bid)   over last k bars

SPY-based signals are not scored against SPY itself.

Target (horizon h bars):
    fwd_ret_h  = close[t+h] / close[t] - 1

For each (k, h) pair the IC / NMI is computed per symbol (time series over the
day), then averaged across symbols. NMI is geometric-normalised mutual
information on equal-count bins (--nmi-bins, default 5); pure noise gives about
(bins-1)^2 / (2 n ln bins), not 0. In addition, a cross-sectional Spearman IC
is computed at each minute across symbols (IC_t), giving ICIR = mean / std of
IC_t over the day and t = ICIR * sqrt(T / h) (T minutes, ~T/h independent).

Usage
-----
    python flow_ic_mi.py
    python flow_ic_mi.py --lookbacks 1 5 15 --horizons 1 5 10
    python flow_ic_mi.py --symbols AAPL MSFT --method pearson
    python flow_ic_mi.py --signals spy_gap spy_resid
    python flow_ic_mi.py --nmi-bins 3

Output: for each signal, IC (mean and t-stat across symbols) and NMI (mean
across symbols, with its noise level) as lookback × horizon grids, then the
cross-sectional mean IC, ICIR and t.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from vwap_ic import cross_sectional_ic, icir_stats

H5_PATH   = Path("data/minute_bars.h5")
LOOKBACKS = (1, 5, 10, 15, 30)
HORIZONS  = (1, 5, 10, 15, 30)
MIN_OBS   = 30
MARKET    = "SPY"
BETA_WINDOW = 60
NMI_BINS  = 5

# ---------------------------------------------------------------------------
# Metrics — self-contained, take two aligned Series
# ---------------------------------------------------------------------------

def ic(signal: pd.Series, fwd_ret: pd.Series, method: str = "spearman") -> float:
    """Information coefficient: correlation of signal with forward return.

    NaNs are dropped pairwise. Returns NaN if fewer than MIN_OBS points remain.
    """
    df = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(df) < MIN_OBS:
        return np.nan
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    if method == "spearman":
        return spearmanr(x, y)[0]
    return x.corr(y)


def normalized_mutual_info(signal: pd.Series, fwd_ret: pd.Series, bins: int = NMI_BINS) -> float:
    """Normalised mutual information between signal and forward return, in [0, 1].

    Both series are cut into `bins` equal-count bins (ranks break ties), and
    NMI = I(X;Y) / sqrt(H(X) H(Y)) is computed on the bin labels. NaNs are
    dropped pairwise. Captures non-linear dependence that IC misses.
    """
    df = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(df) < MIN_OBS:
        return np.nan
    x, y = (pd.qcut(df.iloc[:, j].rank(method="first"), bins, labels=False) for j in (0, 1))
    return normalized_mutual_info_score(x, y, average_method="geometric")


def nmi_noise_level(n: int, bins: int = NMI_BINS) -> float:
    """Approximate NMI of two independent series of length n: (bins-1)^2 / (2 n ln bins)."""
    return (bins - 1) ** 2 / (2 * n * np.log(bins))

# ---------------------------------------------------------------------------
# Signal and target
# ---------------------------------------------------------------------------

def flow_imbalance(bars: pd.DataFrame, k: int) -> pd.Series:
    """Net signed volume / total volume over the last k bars, in [-1, 1]."""
    net   = (bars["n_at_ask"] - bars["n_at_bid"]).rolling(k).sum()
    total = (bars["n_at_ask"] + bars["n_at_bid"]).rolling(k).sum()
    return net / total


def forward_return(bars: pd.DataFrame, h: int) -> pd.Series:
    """Close-to-close return over the next h bars."""
    return bars["close"].pct_change(h).shift(-h)


def past_return(bars: pd.DataFrame, k: int) -> pd.Series:
    """Close-to-close return over the last k bars (known at t)."""
    return bars["close"].pct_change(k)


def rolling_beta(bars: pd.DataFrame, mkt: pd.DataFrame, window: int = BETA_WINDOW) -> pd.Series:
    """Rolling OLS beta of 1-min stock returns on 1-min market returns.

    Uses bars up to and including t only (no lookahead).
    """
    r = pd.concat([past_return(bars, 1), past_return(mkt, 1)], axis=1, keys=["i", "m"])
    roll = r.rolling(window, min_periods=window // 2)
    return roll.cov().xs("m", level=1)["i"] / roll.var()["m"]


def market_residual(bars: dict, sym: str, k: int) -> pd.Series:
    """β_i·r_SPY,k - r_i,k: market-implied move minus the stock's actual move."""
    b, m = bars[sym], bars[MARKET]
    return rolling_beta(b, m) * past_return(m, k) - past_return(b, k)


# Each signal: (all_bars, symbol, k) -> Series on that signal's own minute index.
# Alignment with the target happens by timestamp inside ic() / normalized_mutual_info().
SIGNALS = {
    "own_flow": lambda bars, sym, k: flow_imbalance(bars[sym], k),
    "spy_gap":   lambda bars, sym, k: past_return(bars[MARKET], k) - past_return(bars[sym], k),
    "spy_resid": market_residual,
}
MARKET_SIGNALS = {"spy_gap", "spy_resid"}

# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def load_bars(h5path: Path, symbols=None) -> dict:
    """Return {symbol: minute-bar DataFrame}."""
    with pd.HDFStore(str(h5path), mode="r") as store:
        keys = [k for k in store.keys() if k.startswith("/minute_bars/")]
        bars = {k.split("/")[-1]: store[k] for k in keys}
    if symbols:
        missing = set(symbols) - set(bars)
        if missing:
            print(f"  ⚠  not in {h5path}: {sorted(missing)}")
        bars = {s: b for s, b in bars.items() if s in symbols}
    return bars


def run_study(bars: dict, signal_fn, metric, skip=(),
              lookbacks=LOOKBACKS, horizons=HORIZONS) -> pd.DataFrame:
    """Apply metric(signal, fwd_ret) per symbol for every (lookback, horizon).

    signal_fn(bars, sym, k) builds the signal; symbols in *skip* are not scored.
    Returns long DataFrame: symbol, lookback, horizon, value.
    """
    rows = []
    for sym, b in bars.items():
        if sym in skip:
            continue
        fwd = {h: forward_return(b, h) for h in horizons}
        for k in lookbacks:
            sig = signal_fn(bars, sym, k)
            for h in horizons:
                rows.append((sym, k, h, metric(sig, fwd[h])))
    return pd.DataFrame(rows, columns=["symbol", "lookback", "horizon", "value"])


def cross_sectional_study(bars: dict, signal_fn, skip=(),
                          lookbacks=LOOKBACKS, horizons=HORIZONS) -> dict:
    """Cross-sectional IC across symbols at each minute, summarised per (k, h).

    Returns {"mean": grid, "icir": grid, "t": grid}, each lookback × horizon.
    """
    syms = [sym for sym in bars if sym not in skip]
    fwd = {h: pd.DataFrame({sym: forward_return(bars[sym], h) for sym in syms}) for h in horizons}
    stats = {}
    for k in lookbacks:
        sig = pd.DataFrame({sym: signal_fn(bars, sym, k) for sym in syms})
        for h in horizons:
            stats[(k, h)] = icir_stats(cross_sectional_ic(sig, fwd[h]), h)
    df = pd.DataFrame(stats).T
    df.index.names = ["lookback", "horizon"]
    return {key: df[key].unstack() for key in ("mean", "icir", "t")}


def summarise_cross_sectional(xs: dict, title: str):
    """Print cross-sectional mean IC, ICIR and t as lookback × horizon grids."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print("Mean IC_t over minutes  (rows: lookback k bars, cols: horizon h bars)")
    print(xs["mean"].map(lambda v: format(v, "+.4f")).to_string())
    print("\nICIR = mean(IC_t) / std(IC_t)")
    print(xs["icir"].map(lambda v: format(v, "+.4f")).to_string())
    print("\nt = ICIR * sqrt(T / h)")
    print(xs["t"].round(1).to_string())


def summarise(results: pd.DataFrame, title: str, fmt: str, show_t: bool = True):
    """Print the mean (and t-stat) across symbols as lookback × horizon grids."""
    g = results.groupby(["lookback", "horizon"])["value"]
    mean = g.mean().unstack()
    tstat = (g.mean() / (g.std(ddof=1) / np.sqrt(g.count()))).unstack()

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Mean across {results['symbol'].nunique()} symbols  "
          "(rows: lookback k bars, cols: horizon h bars)")
    print(mean.map(lambda v: format(v, fmt)).to_string())
    if show_t:
        print("\nt-stat across symbols")
        print(tstat.round(1).to_string())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flow-imbalance IC and NMI on minute bars.")
    parser.add_argument("--h5",        type=Path, default=H5_PATH)
    parser.add_argument("--symbols",   nargs="+", default=None,
                        help="Symbols to include (default: all in HDF5)")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=list(LOOKBACKS))
    parser.add_argument("--horizons",  nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--method",    default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--signals",   nargs="+", default=list(SIGNALS), choices=list(SIGNALS))
    parser.add_argument("--nmi-bins",  type=int, default=NMI_BINS,
                        help="Equal-count bins per variable for NMI (default 5)")
    args = parser.parse_args()

    bars = load_bars(args.h5)
    if MARKET not in bars and MARKET_SIGNALS & set(args.signals):
        print(f"{MARKET} not in {args.h5}; cannot build market signals. Exiting.")
        return
    targets = set(args.symbols or bars) & set(bars)
    if not targets:
        print("No symbols loaded. Exiting.")
        return
    print(f"Loaded {len(bars)} symbols from {args.h5}; scoring {', '.join(sorted(targets))}")
    skip = set(bars) - targets

    ic_fn = lambda s, r: ic(s, r, method=args.method)
    nmi_fn = lambda s, r: normalized_mutual_info(s, r, bins=args.nmi_bins)
    n_typical = int(np.median([len(b) for b in bars.values()]))
    noise = nmi_noise_level(n_typical, args.nmi_bins)
    for name in args.signals:
        sk = skip | ({MARKET} if name in MARKET_SIGNALS else set())
        ic_res = run_study(bars, SIGNALS[name], ic_fn, sk, args.lookbacks, args.horizons)
        summarise(ic_res, f"IC ({args.method}) — {name} vs forward return", "+.4f")
        nmi_res = run_study(bars, SIGNALS[name], nmi_fn, sk, args.lookbacks, args.horizons)
        summarise(nmi_res, f"NMI (geometric, {args.nmi_bins} equal-count bins) — {name} vs forward return\n"
                           f"pure noise gives about {noise:.4f} at n = {n_typical}", ".4f", show_t=False)
        xs = cross_sectional_study(bars, SIGNALS[name], sk, args.lookbacks, args.horizons)
        summarise_cross_sectional(xs, f"Cross-sectional IC (spearman) and ICIR — {name}")


if __name__ == "__main__":
    main()
