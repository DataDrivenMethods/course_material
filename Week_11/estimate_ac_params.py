#!/usr/bin/env python3
"""
estimate_ac_params.py
=====================
Estimate the Almgren-Chriss parameters sigma, epsilon, eta and theta for one
stock from a NASDAQ ITCH 5.0 Parquet store (written by create_ITCH_parquet_store.py).

Model (buy side, as in almgren_chriss_derivation.tex, plus a fixed cost epsilon).
The parent order is worked in slices of n = v tau shares every tau seconds:

    d S_bar = sigma S_0 dW
    S_t     = S_bar_t + epsilon sgn(v_t) + eta v_t + theta (X_0 - X_t)

Estimates
  1. sigma    sum of the 390 squared 1-minute log mid returns / 23,400 s
              (as in estimate_sqrt_law.py), so sigma S_0 is in $ per sqrt(s)
  2. epsilon, eta
              temporary impact = cost of consuming liquidity within one slice.
              Every `--snap` seconds the displayed book is snapshotted and the
              cost per share of a market order of n shares is computed by walking
              it:  c(n) = s (average fill price - mid).  Averaged over snapshots
              and both sides, then
                  c(n) = epsilon + (eta / tau) n            (OLS over an n grid)
              so eta = slope x tau.
  3. theta    permanent impact: on bins of `--bin` seconds
                  m1 - m0 = theta dQ + noise                (no constant, HAC s.e.)
              with dQ the net signed traded shares (signed as in estimate_sqrt_law.py).

Why the book and not realised trades for eta: regressing the cost of realised
aggressive orders on their size (or on the binned trading rate) is biased to
zero because traders size their orders to the displayed depth. The slope is
not identified that way (try it: the standard errors exceed the estimates).

Usage
    python estimate_ac_params.py AAPL --store ../Week_4/itch_parquet
    python estimate_ac_params.py AAPL --store ../Week_4/itch_parquet --tau 10 --plot
    python estimate_ac_params.py AAPL --store ../Week_4/itch_parquet --risk-aversion 1e-6 --T 60
"""

import argparse
from heapq import heappush, heappop

import numpy as np
import statsmodels.api as sm

import estimate_sqrt_law as esl
import estimate_ow_decay as eow

NS_PER_SEC = 1_000_000_000
PRICE_SCALE = 10_000.0                    # ITCH prices are dollars x 10,000
SESSION_SEC = (esl.CLOSE_NS - esl.OPEN_NS) / NS_PER_SEC
N_GRID = 25


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def book_snapshots(ev, times):
    """Replay the book (same logic as estimate_sqrt_law.replay_book) and return the
    full bid and ask ladders [(price, shares), ...], best first, at each time."""
    orders, qty = {}, {1: {}, -1: {}}
    snaps, k = [], 0

    def add(ref, side, shares, price):
        orders[ref] = [side, price, shares]
        qty[side][price] = qty[side].get(price, 0) + shares

    def remove(ref, shares):
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
        return side

    def ladder(side):
        lv = [(p, q) for p, q in qty[side].items() if q > 0]
        return sorted(lv, reverse=(side == 1))

    cols = [ev[c].tolist() for c in ("timestamp", "kind", "ref", "new_ref", "side", "shares", "price")]
    for t, kind, ref, new_ref, side, shares, price in zip(*cols):
        while k < len(times) and t > times[k]:
            snaps.append((ladder(1), ladder(-1)))
            k += 1
        if k == len(times):
            break
        if kind == esl.ADD:
            add(ref, side, shares, price)
        elif kind == esl.REPLACE:
            s = remove(ref, None)
            if s is not None:
                add(new_ref, s, shares, price)
        elif kind == esl.EXEC or kind == esl.CANCEL:
            remove(ref, shares)
        else:
            remove(ref, None)
    while k < len(times):
        snaps.append((ladder(1), ladder(-1)))
        k += 1
    return snaps


def walk_cost(snaps, grid):
    """Average over snapshots and sides of s (VWAP of an n-share market order - mid), $ per share."""
    costs = []
    for bids, asks in snaps:
        if not bids or not asks or bids[0][0] >= asks[0][0]:
            continue
        mid = (bids[0][0] + asks[0][0]) / 2.0
        for s, lv in ((1, asks), (-1, bids)):
            p = np.array([x[0] for x in lv], dtype=float)
            cq = np.cumsum([x[1] for x in lv])
            row = np.full(len(grid), np.nan)
            for j, n in enumerate(grid):
                i = np.searchsorted(cq, n)
                if i >= len(cq):
                    break
                prev = cq[i - 1] if i > 0 else 0
                ntl = (p[:i] * np.diff(np.concatenate(([0], cq[:i])))).sum() + p[i] * (n - prev)
                row[j] = s * (ntl / n - mid) / PRICE_SCALE
            costs.append(row)
    costs = np.array(costs)
    return np.nanmean(costs, axis=0), np.nanstd(costs, axis=0) / np.sqrt(np.sum(np.isfinite(costs), axis=0)), len(costs) // 2


def sigma_daily(mid_at):
    """Relative volatility from the 390 one-minute log mid returns (as in estimate_sqrt_law.py)."""
    edges = esl.OPEN_NS + np.arange(esl.N_MINUTES + 1) * esl.NS_PER_MIN
    r = np.diff(np.log(mid_at(edges)))
    return np.sqrt(np.sum(r[np.isfinite(r)] ** 2))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def fit_temporary(grid, cost, se, tau):
    ok = np.isfinite(cost)
    X = sm.add_constant(grid[ok].astype(float))
    r = sm.WLS(cost[ok], X, weights=1.0 / se[ok] ** 2).fit()
    return dict(eps=r.params[0], se_eps=r.bse[0], k=r.params[1], se_k=r.bse[1],
                eta=r.params[1] * tau, se_eta=r.bse[1] * tau, r2=r.rsquared)


def fit_permanent(trades, mid_at, dt, lags):
    n = (esl.CLOSE_NS - esl.OPEN_NS) // (dt * NS_PER_SEC)
    edges = esl.OPEN_NS + np.arange(n + 1) * dt * NS_PER_SEC
    dQ, _ = np.histogram(trades.timestamp, bins=edges, weights=trades.sign * trades.shares)
    dm = np.diff(mid_at(edges)) / PRICE_SCALE
    ok = np.isfinite(dm)
    r = sm.OLS(dm[ok], dQ[ok][:, None]).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(theta=r.params[0], se_theta=r.bse[0], r2=r.rsquared, n=int(ok.sum()))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_temp(ticker, tau, grid, cost, se, est, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue = "#2a78d6"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=surface)
    ax.set_facecolor(surface)

    ax.errorbar(grid, 100 * cost, yerr=100 * se, fmt="o", ms=4, color=ink, mec=surface,
                mew=0.6, elinewidth=0.8, zorder=3,
                label="average cost of walking the displayed book")
    ax.plot(grid, 100 * (est["eps"] + est["k"] * grid), color=blue, lw=2, zorder=4,
            label=rf"$\hat\epsilon + (\hat\eta / \tau) n$:  $\hat\epsilon = {100 * est['eps']:.3f}$ cents" "\n"
                  rf"$\hat\eta = {est['eta']:.2e}$ ({est['se_eta']:.1e}) \$ per share/s,  $\tau = {tau:g}$ s")
    ax.grid(color=grid_c, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelcolor=ink2)

    ax.set_xlabel(r"Slice size  $n = v \tau$  (shares)", color=ink)
    ax.set_ylabel("Cost per share vs mid  (cents)", color=ink)
    fig.suptitle(f"{ticker}: Almgren-Chriss temporary impact from the order book", x=0.06,
                 ha="left", fontsize=13, fontweight="bold", color=ink)
    ax.set_title(r"$S_t = \bar S_t + \epsilon \mathrm{sgn}(v_t) + \eta v_t + \theta (X_0 - X_t)$,"
                 " standard error in parentheses", loc="left", fontsize=9.5, color=ink2)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(ink)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Estimate Almgren-Chriss sigma, epsilon, eta and theta")
    p.add_argument("ticker")
    p.add_argument("--store", required=True, help="path to itch_parquet/")
    p.add_argument("--tau", type=float, default=10.0, help="slice interval in seconds (eta = slope x tau)")
    p.add_argument("--snap", type=int, default=10, help="book snapshot interval in seconds")
    p.add_argument("--max-frac", type=float, default=1e-3,
                   help="largest slice in the n grid, as a fraction of daily volume")
    p.add_argument("--bin", type=int, default=10, help="bin length in seconds for theta")
    p.add_argument("--risk-aversion", type=float, default=None,
                   help="lambda in 1/$: also print kappa and kappa*T")
    p.add_argument("--T", type=float, default=60.0, help="horizon in minutes for kappa*T")
    p.add_argument("--plot", action="store_true", help="save the book-walk cost against the fitted line")
    a = p.parse_args()

    ticker = a.ticker.upper()
    loc = esl.stock_locate(a.store, ticker)
    print(f"{ticker}: stock_locate {loc}")

    trades, mid_at, V, n_events, missing = eow.signed_trades(a.store, loc)
    print(f"  order events {n_events:,}   unmatched references {missing:,}")
    print(f"  trades {len(trades):,}   V = {V:,.0f} shares")

    # 1. volatility
    edges = esl.OPEN_NS + np.arange(esl.N_MINUTES + 1) * esl.NS_PER_MIN
    S0 = np.nanmedian(mid_at(edges)) / PRICE_SCALE
    sig_day = sigma_daily(mid_at)
    sig = sig_day / np.sqrt(SESSION_SEC)

    # 2. temporary impact from the displayed book
    times = np.arange(esl.OPEN_NS + a.snap * NS_PER_SEC, esl.CLOSE_NS, a.snap * NS_PER_SEC)
    snaps = book_snapshots(esl.order_events(a.store, loc), times)
    n_max = max(N_GRID, int(a.max_frac * V))
    grid = np.linspace(n_max / N_GRID, n_max, N_GRID)
    cost, se, n_snap = walk_cost(snaps, grid)
    temp = fit_temporary(grid, cost, se, a.tau)

    # 3. permanent impact
    perm = fit_permanent(trades, mid_at, a.bin, max(1, 300 // a.bin))

    print(f"\n  S0        = {S0:.2f} dollars (median mid)")
    print(f"  sigma     = {sig:.3e} per sqrt(second)   = {100 * sig_day:.2f}% per day")
    print(f"  sigma S0  = {sig * S0:.4f} dollars per sqrt(second)          -> JSON sigma_usd_per_sqrt_s")
    print(f"\n  book snapshots {n_snap:,} every {a.snap} s   slices n = {grid[0]:.0f} .. {grid[-1]:.0f} shares")
    print(f"  epsilon   = {temp['eps']:.5f} ({temp['se_eps']:.5f}) dollars per share             -> JSON ac.epsilon_usd")
    print(f"  slope     = {temp['k']:.3e} ({temp['se_k']:.1e}) dollars per share per share   R2 = {temp['r2']:.3f}"
          "   (not used: eta = slope x tau)")
    print(f"  eta       = {temp['eta']:.3e} ({temp['se_eta']:.1e}) dollars per (share/second)   -> JSON ac.eta_usd_s_per_sh2")
    print(f"  tau       = {a.tau:g} seconds                                    -> JSON ac.tau_sec")
    print(f"\n  {a.bin}-second bins: n = {perm['n']:,}")
    print(f"  theta     = {perm['theta']:.3e} ({perm['se_theta']:.1e}) dollars per share   R2 = {perm['r2']:.3f}"
          "   -> JSON ac.theta_usd_per_sh")

    if a.risk_aversion is not None and temp["eta"] > 0:
        kappa = S0 * sig * np.sqrt(a.risk_aversion / temp["eta"])
        print(f"\n  lambda    = {a.risk_aversion:.2e} per dollar                        -> JSON risk_aversion_per_usd")
        print(f"  kappa     = S0 sigma sqrt(lambda / eta) = {kappa:.3e} per second")
        print(f"  kappa T   = {kappa * a.T * 60:.3f}   (T = {a.T:g} minutes)          -> JSON ac.kappa_T (optional override)")

    # Paste-ready lines for the C++ exec apps (exchange_simulator config JSON),
    # in the same format as config/example_params.json.
    print(f'\n  JSON for the exec apps: paste into "symbols": {{ "{ticker}": {{ ... }} }}')
    print(f'    "sigma_usd_per_sqrt_s": {sig * S0:.4f},')
    print(f'    "ac": {{ "epsilon_usd": {temp["eps"]:.4f}, "eta_usd_s_per_sh2": {temp["eta"]:.3e}, '
          f'"tau_sec": {a.tau:g}, "theta_usd_per_sh": {perm["theta"]:.3e}, "kappa_T": null }},')
    print('  kappa_T = null: the app computes kappa = sigma_usd_per_sqrt_s sqrt(risk_aversion_per_usd / eta)')
    print('  with risk_aversion_per_usd (your choice, set in "defaults"). A number instead fixes kappa T.')
    print('  The "ow" entry comes from estimate_ow_decay.py.')

    if a.plot:
        path = f"ac_params_{ticker}.png"
        plot_temp(ticker, a.tau, grid, cost, se, temp, path)
        print(f"\n  plot saved to {path}")


if __name__ == "__main__":
    main()
