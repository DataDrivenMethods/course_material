"""PEAD with a hybrid (fundamental + statistical) covariance estimate and walk-forward MVO.

Same signal, universe, timing, portfolios and costs as pead_mvo.py; only the covariance
estimate changes. Each day t several estimates are built from the trailing COV_WINDOW days
of de-meaned returns R (n x T, data up to t), with sample covariance S = R R' / (T - 1), and
the MVO portfolio is formed with each:

  PCA k        two-stage PCA, Sigma = B diag(F) B' + D, for each k in PCA_FACTORS.

  Fundamental  Sigma = X Sigma_f X' + D. X (n x p) holds exposures known at t:
                 industries  one dummy per 2-digit SIC code (codes with fewer than
                             MIN_INDUSTRY names are pooled into one group); the dummies
                             sum to one for every stock, so together they also carry the
                             market
                 size        log market cap
                 momentum    sum of returns over t-251..t-21
                 volatility  realized volatility over t-62..t
               Styles are winsorized at +-3 and z-scored across the names each day.
               X is given, so only Sigma_f and D are estimated. With A a left inverse of X
               (A X = I),
                   Sigma_f = A S A'
                   D       = diag of (I - X A) S (I - X A)' (the variance outside span(X))
               A is the generalized least squares left inverse, in two steps as in
               two-stage PCA:
                   1. A_0 = (X'X)^-1 X', and D_0 the diagonal of (I - X A_0) S (I - X A_0)'
                   2. A   = (X' D_0^-1 X)^-1 X' D_0^-1

  Hybrid       The fundamental model, then two-stage PCA with k = N_STAT_FACTORS on the
               residuals U = (I - X A) R. With B the top k left singular vectors of U,
               L = [X, B] and C = [A; B'(I - X A)] (so that C L = I),
                   Sigma = L (C S C') L' + D
               with D the variances of the residuals after both stages.

  Hybrid, factor-neutral
               The hybrid estimate, with the MVO portfolio also constrained to zero
               exposure to every column of X: X' w = 0 (industries, size, momentum and
               volatility; since the industry dummies sum to one, this includes 1'w = 0).

All are applied by the Woodbury identity. The comparison is the bias ratio: realized
volatility / predicted volatility of the portfolios. A ratio of 1 means the estimate gets
the risk of the optimized portfolio right; above 1 it underpredicts.

Usage:
    python pead_hybrid_mvo.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import pead_mvo as pm

HERE = os.path.dirname(os.path.abspath(__file__))
SIC_CACHE = os.path.join(HERE, 'data', 'pead_sic.pkl')
PLOT = os.path.join(HERE, 'plot', 'pead_hybrid_mvo_equity.png')

PCA_FACTORS = [10, 25, 50]      # k for the PCA models
N_STAT_FACTORS = 5              # statistical factors on the fundamental residuals
MIN_INDUSTRY = 3                # smallest industry kept as its own factor
STYLES = ['size', 'momentum', 'volatility']
MODELS = [f'PCA k={k}' for k in PCA_FACTORS] + ['Fundamental', 'Hybrid', 'Hybrid, factor-neutral']
HALFLIFE = None                 # days; if set, exponentially weight the days in S


# ---------------------------------------------------------------------------
# Exposures
# ---------------------------------------------------------------------------

def sic_panel(dates, names):
    """dates x names 2-digit SIC code in effect on each date (point in time), -1 if unknown."""
    sic = pd.read_pickle(SIC_CACHE)
    out = np.full((len(dates), len(names)), -1)
    col = names.get_indexer(sic.permno.astype(int))
    for j, s, e, code in zip(col, sic.namedt, sic.nameendt, sic.siccd):
        if j >= 0 and pd.notna(code):
            out[(dates >= s) & (dates <= e), j] = int(code) // 100
    return out


def rolling_sum(X, w):
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(np.nan_to_num(X), 0)])
    out = np.full(X.shape, np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def style_panels(Rv, capv):
    """dates x names raw style values, each known at the close of the date."""
    mom = np.vstack([np.full((21, Rv.shape[1]), np.nan), rolling_sum(Rv, 231)[:-21]])
    m1, m2 = rolling_sum(Rv, 63) / 63, rolling_sum(Rv ** 2, 63) / 63
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'size': np.log(capv), 'momentum': mom, 'volatility': np.sqrt(np.maximum(m2 - m1 ** 2, 0))}


def exposures(t, idx, sic, styles):
    """n x p exposure matrix on day t for names idx: industry dummies, then z-scored styles."""
    codes = sic[t, idx].copy()
    vals, counts = np.unique(codes, return_counts=True)
    small = vals[(counts < MIN_INDUSTRY) | (vals < 0)]
    codes[np.isin(codes, small)] = -1                                  # pooled "other" group
    inds = np.unique(codes)
    X = [(codes == c).astype(float) for c in inds]
    for s in STYLES:
        x = styles[s][t, idx]
        ok = np.isfinite(x)
        z = np.zeros(len(idx))
        if ok.sum() > 1:
            z[ok] = (x[ok] - x[ok].mean()) / x[ok].std()
            z = np.clip(z, -3, 3)
            z[ok] = (z[ok] - z[ok].mean()) / z[ok].std()
        X.append(z)                                                    # missing -> 0, the mean
    return np.column_stack(X), len(inds)


# ---------------------------------------------------------------------------
# Covariance estimates, all of the form Sigma = L G L' + diag(D) with G a full k x k matrix
# ---------------------------------------------------------------------------

def pca_model(R, k):
    return pm.two_stage_cov(R, k)                                     # F is already the k x k factor covariance


def left_inverse(X, w):
    """A = (X' W X)^-1 X' W with W = diag(w), so that A X = I."""
    XW = X * w[:, None]
    return np.linalg.lstsq(XW.T @ X, XW.T, rcond=None)[0]


def fundamental_model(R, X, k_stat):
    """R n x T de-meaned returns, X n x p exposures. Returns L, G, D with Sigma = L G L' + diag(D)."""
    T = R.shape[1]
    A0 = left_inverse(X, np.ones(len(X)))                             # step 1: least squares
    D0 = (R - X @ (A0 @ R)).var(1, ddof=1)
    A = left_inverse(X, 1 / D0)                                       # step 2: GLS, weights 1 / D_0
    f = A @ R                                                         # p x T; Sigma_f = A S A' = cov(f)
    U = R - X @ f
    L, g = X, f
    if k_stat > 0:
        Uc = U - U.mean(1, keepdims=True)
        B = np.linalg.svd(Uc, full_matrices=False)[0][:, :k_stat]
        h = B.T @ U                                                   # k_stat x T
        U = U - B @ h
        L, g = np.hstack([X, B]), np.vstack([f, h])
    if HALFLIFE is None:
        G, D = np.cov(g), U.var(1, ddof=1)
    else:
        a = 0.5 ** (np.arange(T)[::-1] / HALFLIFE)                    # weight 1 on day t
        G = np.cov(g, aweights=a)
        D = ((U - (U * a).sum(1, keepdims=True) / a.sum()) ** 2 * a).sum(1) / a.sum()
    return L, G, D


def solve(L, G, D, V):
    """Sigma^-1 V for Sigma = L G L' + diag(D), by the Woodbury identity."""
    DV, DL = V / D[:, None], L / D[:, None]
    M = np.linalg.pinv(G) + L.T @ DL
    return DV - DL @ np.linalg.lstsq(M, L.T @ DV, rcond=None)[0]


def variance(L, G, D, w):
    a = L.T @ w
    return a @ G @ a + (w ** 2 * D).sum()


def mvo_weights(sue, L, G, D, C=None):
    """Closed-form MVO subject to C'w = 0, alpha mu_i = sigma_i z_i, scaled to TARGET_VOL.

    max w'mu - lambda w'Sigma w subject to C'w = 0 has the solution
        w proportional to Sigma^-1 (mu - C nu),  nu = (C' Sigma^-1 C)^-1 C' Sigma^-1 mu
    C = 1 (the default) is dollar neutral.
    """
    sigma = np.sqrt(np.einsum('ij,jk,ik->i', L, G, L) + D)
    z = norm.ppf((sue.rank().to_numpy() - 0.5) / len(sue))
    if C is None:
        C = np.ones((len(sue), 1))
    C = C[:, np.abs(C).sum(0) > 0]                                    # drop industries with no active names
    Y = solve(L, G, D, np.column_stack([sigma * z, C]))
    nu = np.linalg.lstsq(C.T @ Y[:, 1:], C.T @ Y[:, 0], rcond=None)[0]
    w = Y[:, 0] - Y[:, 1:] @ nu
    return w * pm.TARGET_VOL / np.sqrt(252 * variance(L, G, D, w))


# ---------------------------------------------------------------------------

def build_weights(events, rets):
    held = events.loc[events.day.between(pm.PF_FIRST - 1, pm.PF_LAST - 1) & events.sue3.notna()]
    held = held.sort_values('rdq1').drop_duplicates(subset=['date', 'permno'], keep='last')
    R = rets.pivot(index='date', columns='permno', values='ret').sort_index()
    S = held.pivot(index='date', columns='permno', values='sue3').reindex(index=R.index, columns=R.columns)
    Rv, Sv = R.to_numpy(dtype=float), S.to_numpy(dtype=float)
    dates, names = R.index, pd.Index(R.columns.astype(int))
    cap = pm.load_mcap(names).pivot(index='date', columns='permno', values='mcap')
    capv = cap.reindex(index=dates, columns=names).ffill().to_numpy(dtype=float)
    sic = sic_panel(dates, names)
    styles = style_panels(Rv, capv)

    n_dates, n_names = Rv.shape
    W = {m: np.zeros((n_dates, n_names)) for m in ['Stratified'] + MODELS}
    pred = {(p, m): [] for p in ['Stratified', 'MVO'] for m in MODELS}
    n_ind = []
    for t in range(pm.COV_WINDOW - 1, n_dates):
        active_today = np.isfinite(Sv[t])
        if active_today.sum() < 2:
            continue
        win = Rv[t - pm.COV_WINDOW + 1:t + 1].T
        ok = (np.isfinite(win).mean(1) >= pm.MIN_OBS) & np.isfinite(capv[t])
        idx = np.flatnonzero(ok)
        Rw = np.nan_to_num(win[ok] - np.nanmean(win[ok], axis=1, keepdims=True))
        X, k_ind = exposures(t, idx, sic, styles)
        n_ind.append(k_ind)
        models = {f'PCA k={k}': pca_model(Rw, k) for k in PCA_FACTORS}
        models['Fundamental'] = fundamental_model(Rw, X, 0)
        models['Hybrid'] = fundamental_model(Rw, X, N_STAT_FACTORS)
        models['Hybrid, factor-neutral'] = models['Hybrid']

        act = np.isfinite(Sv[t, idx])
        if act.sum() < 2:                                            # need a long and a short
            continue
        ia = idx[act]
        sue = pd.Series(Sv[t, ia])
        ws = pm.stratified_weights(sue)
        W['Stratified'][t, ia] = ws
        for m, (L, G, D) in models.items():
            L, D = L[act], D[act]
            C = X[act] if m.endswith('factor-neutral') else None
            w = mvo_weights(sue, L, G, D, C)
            W[m][t, ia] = w
            pred[('MVO', m)].append(np.sqrt(252 * variance(L, G, D, w)))
            pred[('Stratified', m)].append(np.sqrt(252 * variance(L, G, D, ws)))
    return W, R, {k: np.mean(v) for k, v in pred.items()}, np.mean(n_ind)


def main():
    events, rets = pm.load_data()
    W, R, pred, n_ind = build_weights(events, rets)
    Rv = R.to_numpy(dtype=float)
    first = np.flatnonzero(np.abs(W['Stratified']).sum(1) > 0)[0]
    dates = R.index[first + 1:]

    rows, nets = {}, {}
    for name, Wm in W.items():
        r, traded = pm.backtest(Wm, Rv)
        gross = np.concatenate([[np.nan], np.abs(Wm).sum(1)[:-1]])[first + 1:]
        label = 'Stratified' if name == 'Stratified' else f'MVO, {name}'
        rows[label], nets[label] = pm.stats(r[first + 1:], traded[first + 1:], gross)

    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    print(f'PEAD (sue3), {dates[0].date()} to {dates[-1].date()}, daily re-estimation, {pm.COV_WINDOW}-day window')
    print(f'Fundamental factors: {n_ind:.0f} industries (2-digit SIC) on average + {", ".join(STYLES)}. '
          f'Hybrid adds {N_STAT_FACTORS} statistical factors.\n')
    print(table.round(3).to_string())

    bias = pd.DataFrame({m: {'predicted vol': pred[('MVO', m)],
                             'realized vol': table.loc[f'MVO, {m}', ('gross', 'vol_ann')]} for m in MODELS}).T
    bias['bias ratio'] = bias['realized vol'] / bias['predicted vol']
    strat = pd.Series({m: pred[('Stratified', m)] for m in MODELS})
    print('\nMVO portfolios: realized / predicted volatility under each estimate')
    print(bias.round(3).to_string())
    print(f'\nStratified portfolio, realized vol {table.loc["Stratified", ("gross", "vol_ann")]:.3f}; '
          'predicted by each estimate:')
    print(strat.round(3).to_string())

    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    plt.figure(figsize=(16, 10))
    for (name, x), c in zip(nets.items(), ['grey', 'lightblue', 'skyblue', 'navy', 'orchid', 'olive', 'firebrick']):
        x = pd.Series(x, index=dates)
        plt.plot(x.cumsum(), color=c, linewidth=3,
                 label=f'{name} (Sharpe {rows[name][(f"net {pm.COST_BPS} bps", "sharpe")]:.2f})')
    plt.axhline(0, color='grey', linewidth=1)
    plt.title(f'PEAD long-short portfolios, net of {pm.COST_BPS} bps\nCumulative return', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Cumulative return', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.savefig(PLOT)
    print(f'\nPlot saved to {PLOT}')
    return table, bias


if __name__ == '__main__':
    main()
