#!/usr/bin/env python3
"""
estimate_ow_decay.py
====================
Estimate the Obizhaeva-Wang impact parameters lambda and beta for one stock
from a NASDAQ ITCH 5.0 Parquet store (written by create_ITCH_parquet_store.py).

Model (impact I, cumulative signed volume Q):

    dI = -beta I dt + lambda dQ

On bins of length dt, with phi = exp(-beta dt):

    I_t  = phi I_{t-1} + lambda dQ_t
    dm_t = I_t - I_{t-1} + noise
         = lambda ( dQ_t - (1 - phi) J_{t-1} ) + noise
    J_t  = phi J_{t-1} + dQ_t

dm_t is the log mid change in bps and dQ_t the net signed volume in % of daily
volume, so lambda is in bps per 1% of daily volume and beta is per second.

Steps
  1-3. Order book, signed trades and V as in estimate_sqrt_law.py (imported).
  4.   Bins of dt seconds from 9:30 to 16:00: dQ_t and dm_t.
  5.   Nonlinear least squares over lambda and beta: a grid over beta for a
       starting point (lambda is linear given beta), then a joint fit.

Usage
    python estimate_ow_decay.py AAPL --store ../Week_4/itch_parquet
    python estimate_ow_decay.py AAPL --store ../Week_4/itch_parquet --bin 10 --plot
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

import estimate_sqrt_law as esl

NS_PER_SEC = 1_000_000_000
PRICE_SCALE = 10_000.0                    # ITCH prices are dollars x 10,000
MAX_LAG_SEC = 600                         # horizon of the empirical kernel in the plot


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def signed_trades(store, loc):
    """Order book, signed trades 9:30-16:00 and V (steps 1-3 of estimate_sqrt_law.py)."""
    ev = esl.order_events(store, loc)
    ts, bb, ba, lit, missing = esl.replay_book(ev)
    mid_at = esl.mid_lookup(ts, bb, ba)

    hid = esl.load(store, "P", loc, ["timestamp", "shares", "price"])
    hid_mid = mid_at(hid.timestamp.values)
    hid["sign"] = np.sign(hid.price.values - hid_mid).astype(float)
    hid.loc[np.isnan(hid_mid), "sign"] = 0.0

    trades = pd.concat([lit, hid[["timestamp", "sign", "shares", "price"]]], ignore_index=True)
    trades = trades[(trades.timestamp >= esl.OPEN_NS) & (trades.timestamp < esl.CLOSE_NS)]
    V = trades.shares.sum()
    trades = trades[trades.sign != 0]
    return trades, mid_at, V, len(ev), missing


def bin_data(trades, mid_at, V, dt):
    """dQ_t (% of daily volume) and dm_t (bps) on bins of dt seconds from 9:30 to 16:00."""
    n = (esl.CLOSE_NS - esl.OPEN_NS) // (dt * NS_PER_SEC)
    edges = esl.OPEN_NS + np.arange(n + 1) * dt * NS_PER_SEC
    dQ, _ = np.histogram(trades.timestamp, bins=edges,
                         weights=trades.sign * trades.shares)
    dm = 1e4 * np.diff(np.log(mid_at(edges)))
    ok = np.isfinite(dm)
    dm[~ok] = 0.0
    return 100.0 * dQ / V, dm


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def regressor(dQ, phi):
    """dQ_t - (1 - phi) J_{t-1}, with J_t = phi J_{t-1} + dQ_t."""
    J = lfilter([1.0], [1.0, -phi], dQ)
    J_prev = np.concatenate(([0.0], J[:-1]))
    return dQ - (1.0 - phi) * J_prev


def fit(dQ, dm, dt):
    # starting point: profile over beta (half-lives from dt to 1 hour)
    best = None
    for h in np.geomspace(dt, 3600.0, 60):
        beta = np.log(2.0) / h
        X = regressor(dQ, np.exp(-beta * dt))
        lam = X @ dm / (X @ X)
        ssr = np.sum((dm - lam * X) ** 2)
        if best is None or ssr < best[0]:
            best = (ssr, lam, beta)
    _, lam0, beta0 = best

    # joint nonlinear least squares in lambda and log beta
    def resid(p):
        return dm - p[0] * regressor(dQ, np.exp(-np.exp(p[1]) * dt))

    sol = least_squares(resid, x0=[lam0, np.log(beta0)])
    lam, beta = sol.x[0], np.exp(sol.x[1])
    n, k = len(dm), 2
    s2 = np.sum(sol.fun ** 2) / (n - k)
    cov = s2 * np.linalg.inv(sol.jac.T @ sol.jac)
    se_lam = np.sqrt(cov[0, 0])
    se_beta = beta * np.sqrt(cov[1, 1])          # delta method from log beta
    r2 = 1.0 - np.sum(sol.fun ** 2) / np.sum((dm - dm.mean()) ** 2)
    return dict(n=n, lam=lam, se_lam=se_lam, beta=beta, se_beta=se_beta,
                half_life=np.log(2.0) / beta, r2=r2)


def empirical_kernel(dQ, dm, dt):
    """Cumulative coefficients of dm_t on dQ_t ... dQ_{t-L}: the response G at lags 0..L."""
    L = max(1, MAX_LAG_SEC // dt)
    n = len(dQ)
    X = np.zeros((n - L, L + 1))
    for l in range(L + 1):
        X[:, l] = dQ[L - l:n - l]
    g, *_ = np.linalg.lstsq(X, dm[L:], rcond=None)
    return dt * np.arange(L + 1), np.cumsum(g)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_kernel(ticker, dt, tau, G, est, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue = "#2a78d6"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor=surface)
    ax.set_facecolor(surface)

    t = np.linspace(0, tau.max(), 400)
    ax.plot(tau, G, "o", ms=4, color=ink, mec=surface, mew=0.6, zorder=3,
            label="empirical response (cumulative lag coefficients)")
    ax.plot(t, est["lam"] * np.exp(-est["beta"] * t), color=blue, lw=2, zorder=4,
            label=rf"$\hat\lambda e^{{-\hat\beta \tau}}$:  $\hat\lambda = {est['lam']:.1f}$ "
                  rf"({est['se_lam']:.1f})" "\n"
                  rf"$\hat\beta = {est['beta']:.4f}$ ({est['se_beta']:.4f}) per s"
                  rf"   half-life $= {est['half_life']:.0f}$ s")
    ax.axhline(0, color=ink2, lw=0.6, zorder=0)
    ax.grid(color=grid, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelcolor=ink2)

    ax.set_xlabel(r"Time since trade  $\tau$  (seconds)", color=ink)
    ax.set_ylabel("Impact of 1% of daily volume  (bps)", color=ink)
    fig.suptitle(f"{ticker}: Obizhaeva-Wang impact decay, {dt}-second bins", x=0.06,
                 ha="left", fontsize=13, fontweight="bold", color=ink)
    ax.set_title(r"$dI = -\beta I dt + \lambda dQ$, standard error in parentheses",
                 loc="left", fontsize=9.5, color=ink2)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(ink)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Estimate Obizhaeva-Wang lambda and beta")
    p.add_argument("ticker")
    p.add_argument("--store", required=True, help="path to itch_parquet/")
    p.add_argument("--bin", type=int, default=1, help="bin length in seconds")
    p.add_argument("--plot", action="store_true",
                   help="save the empirical response against the fitted exponential")
    a = p.parse_args()

    ticker = a.ticker.upper()
    loc = esl.stock_locate(a.store, ticker)
    print(f"{ticker}: stock_locate {loc}")

    trades, mid_at, V, n_events, missing = signed_trades(a.store, loc)
    print(f"  order events {n_events:,}   unmatched references {missing:,}")
    print(f"  trades {len(trades):,}   V = {V:,.0f} shares")

    dQ, dm = bin_data(trades, mid_at, V, a.bin)
    est = fit(dQ, dm, a.bin)
    mid = np.nanmedian(mid_at(np.array([esl.OPEN_NS, esl.CLOSE_NS]))) / PRICE_SCALE
    lam_usd = est["lam"] * 1e-4 * mid / (0.01 * V) * 1000.0

    print(f"\n  {a.bin}-second bins: n = {est['n']:,}")
    print(f"  lambda    = {est['lam']:.3f} ({est['se_lam']:.3f}) bps per 1% of daily volume")
    print(f"            = {lam_usd:.4f} dollars per 1,000 shares")
    print(f"  beta      = {est['beta']:.5f} ({est['se_beta']:.5f}) per second")
    print(f"  half-life = {est['half_life']:.1f} seconds")
    print(f"  R2        = {est['r2']:.3f}")

    if a.plot:
        tau, G = empirical_kernel(dQ, dm, a.bin)
        path = f"ow_decay_{ticker}.png"
        plot_kernel(ticker, a.bin, tau, G, est, path)
        print(f"\n  plot saved to {path}")


if __name__ == "__main__":
    main()
