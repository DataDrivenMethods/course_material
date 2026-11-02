"""Daily return forecast from fundamental, price-volume and implied-volatility features,
traded as a stratified (quintile) portfolio or by MVO with a two-stage covariance estimate.
The forecast, the covariance estimate and the portfolio are all updated every day.

Usage:
    python fundamentals_mvo.py stratified
    python fundamentals_mvo.py mvo
Each run prints the performance tables and saves, in plot/,
    fundamentals_<mode>_quintiles.png    quintile portfolios (by forecast, or by MVO weight)
    fundamentals_<mode>_equity.png       cumulative return, gross and net of costs
    fundamentals_<mode>_performance.csv  the performance table
The features, forecasts and daily covariance estimates are cached in data/ on the first run
(--refresh rebuilds them).

A rebuild of IEORE4576_2025/Notes/Week_13/fundamentals_mvo.py without its look-ahead
(there the signal was trained on targets up to ret_t and then scored on ret_t).

Universe: point-in-time S&P 500 members (crsp.msp500list), 2007-2023.

Features of name i at the close of day t (each z-scored across the members on t; a
missing value is set to 0, the cross-sectional mean):
  price-volume (CRSP dsf, through t)
    r1      return on t                                   (short-term reversal)
    r5      sum of returns t-4..t
    r21     sum of returns t-20..t
    mom     sum of returns t-251..t-21
    rv      21-day realized volatility
    turn    log mean daily turnover (vol/shrout), 21 days
    aturn   log turnover on t minus log mean turnover over t-20..t (abnormal volume)
    illiq   log mean of |ret| / dollar volume, 21 days (Amihud)
  implied volatility (OptionMetrics vsurfd, 30-day surface, as of the close of t-1, because
  options close 15 minutes after stocks)
    iv      at-the-money implied volatility, mean of the 50-delta call and put
    cpspr   50-delta call minus 50-delta put implied volatility
    skew    25-delta put minus at-the-money implied volatility
    ivrv    iv minus rv
    div     change in iv over the previous 5 days
  fundamentals (Compustat fundq, quarters with report date rdq before t)
    ep      ttm earnings / mcap
    roa     ttm earnings / assets
    dearn   change in ttm earnings from a year earlier / assets
    sg      ttm sales growth over a year

Forecast: pooled OLS of the de-meaned next-day return on the features, fitted each day t
on the TRAIN_DAYS pairs (features on s, return on s + 1) with s + 1 <= t. The forecast
Z_t beta is in return units; mu_t is its exponential moving average with half-life
HALFLIFE days.

Portfolios, set at the close of t and earning the return of t + 1:
  Stratified  long the top forecast quintile, short the bottom, equal-weighted, $1 per leg.
  MVO         max w'mu - lambda w'Sigma w subject to 1'w = 0, in closed form
                  w proportional to Sigma^-1 mu - (b/a) Sigma^-1 1
              lambda set so the predicted volatility is TARGET_VOL.
              Sigma = B diag(F) B' + D is the two-stage PCA estimate from the trailing
              COV_WINDOW daily returns (n x T, ending on t), applied with the Woodbury identity.

Each day both portfolios trade a fraction TRADE_RATE of the way from the drifted holdings
to the target above. Costs are COST_BPS per dollar traded.

The return of each portfolio on t + 1 is split into a factor part w' B B' r and a residual
part w' (I - B B') r, with B the estimated factors on t.

Compustat values are the current (possibly restated) vintage.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
FUND_DATA = os.path.join(HERE, 'data', 'sp500_fundamentals.pkl')      # members, daily ret/mcap, fundq
PV_DATA = os.path.join(HERE, 'data', 'sp500_dsf.pkl')                 # daily prc, vol, shrout
OPT_DATA = os.path.join(HERE, 'data', 'sp500_ivol.pkl')               # OptionMetrics 30-day surface
FORECAST_DATA = os.path.join(HERE, 'data', 'sp500_forecast.pkl')     # cache of forecasts and covariance estimates
PLOT_DIR = os.path.join(HERE, 'plot')

BEGIN, END = '2007-01-01', '2023-12-31'
PV = ['r1', 'r5', 'r21', 'mom', 'rv', 'turn', 'aturn', 'illiq']
OPT = ['iv', 'cpspr', 'skew', 'ivrv', 'div']
FUND = ['ep', 'roa', 'dearn', 'sg']
FEATURES = PV + OPT + FUND
TRAIN_DAYS = 252
HALFLIFE = 5                    # days, smoothing of the forecast
TRADE_RATE = 0.2                # fraction of the way to the target traded each day
MAX_STALE = 200
N_FACTORS = 10
COV_WINDOW = 252
FIRST = 2 * COV_WINDOW          # first trading day: one year for momentum, one for training
MIN_OBS = 0.9
MIN_NAMES = 100
TARGET_VOL = 0.10
COST_BPS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def wrds_connect():
    import wrds
    return wrds.Connection(wrds_username='cpehlivanian')


def load_data():
    """Members, daily CRSP (ret, prc, vol, shrout), Compustat quarters and implied vols."""
    d = pd.read_pickle(FUND_DATA)
    members, fq = d['members'], d['fq']
    permnos = tuple(int(p) for p in members.permno.unique())
    if not os.path.exists(PV_DATA):
        conn = wrds_connect()
        dsf = conn.raw_sql(f"""select permno, date, ret, abs(prc) as prc, vol, shrout from crsp.dsf
                               where permno in {permnos} and date between '{BEGIN}' and '{END}'""",
                           date_cols=['date'])
        conn.close()
        pd.to_pickle(dsf, PV_DATA)
        print('dsf', len(dsf), flush=True)
    if not os.path.exists(OPT_DATA):
        conn = wrds_connect()
        link = conn.raw_sql(f"""select secid, permno, sdate, edate from wrdsapps.opcrsphist
                                where score = 1 and permno in {permnos}""", date_cols=['sdate', 'edate'])
        secids = tuple(int(s) for s in link.secid.unique())
        parts = []
        for y in range(int(BEGIN[:4]), int(END[:4]) + 1):
            parts.append(conn.raw_sql(f"""select secid, date, delta, impl_volatility as iv from optionm.vsurfd{y}
                                          where days = 30 and delta in (50, -50, -25) and secid in {secids}""",
                                      date_cols=['date']))
            print('options', y, len(parts[-1]), flush=True)
        conn.close()
        opt = pd.concat(parts).merge(link, on='secid')
        opt = opt.loc[(opt.date >= opt.sdate) & (opt.date <= opt.edate), ['permno', 'date', 'delta', 'iv']]
        pd.to_pickle(opt, OPT_DATA)
    return members, pd.read_pickle(PV_DATA), fq, pd.read_pickle(OPT_DATA)


def wide(df, col, index, columns):
    out = df.pivot_table(index='date', columns='permno', values=col)
    return out.reindex(index=index, columns=columns).astype(float).to_numpy()


def membership(members, dates, names):
    """dates x names boolean: member of the S&P 500 on that date."""
    M = np.zeros((len(dates), len(names)), bool)
    for p, s, e in members[['permno', 'start', 'ending']].itertuples(index=False):
        j = names.get_indexer([int(p)])[0]
        if j >= 0:
            M[(dates >= s) & (dates <= e), j] = True
    return M


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def rolling_sum(X, w):
    """Sum over rows t-w+1..t (NaN treated as 0); NaN for the first w-1 rows."""
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(np.nan_to_num(X), 0)])
    out = np.full(X.shape, np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def rolling_mean(X, w):
    X = np.where(np.isfinite(X), X, np.nan)
    n = rolling_sum(np.isfinite(X).astype(float), w)
    return rolling_sum(X, w) / np.where(n > 0.5 * w, n, np.nan)


def lag(X, k):
    return np.vstack([np.full((k, X.shape[1]), np.nan), X[:-k]])


def price_volume_features(Rv, P, V, S):
    with np.errstate(divide='ignore', invalid='ignore'):
        turn = np.where(V > 0, V / S, np.nan)
        dvol = np.where(V > 0, P * V, np.nan)
        m1, m2 = rolling_mean(Rv, 21), rolling_mean(Rv ** 2, 21)
        mt = rolling_mean(turn, 21)
        return {'r1': Rv,
                'r5': rolling_sum(Rv, 5),
                'r21': rolling_sum(Rv, 21),
                'mom': lag(rolling_sum(Rv, 231), 21),
                'rv': np.sqrt(np.maximum(m2 - m1 ** 2, 0) * 252),
                'turn': np.log(mt),
                'aturn': np.log(turn) - np.log(mt),
                'illiq': np.log(rolling_mean(np.abs(Rv) / dvol, 21) * 1e6)}


def option_features(opt, dates, names, rv):
    """Implied-volatility features, lagged one day."""
    g = {k: wide(opt.loc[opt.delta == k], 'iv', dates, names) for k in (50, -50, -25)}
    iv = (g[50] + g[-50]) / 2
    out = {'iv': iv, 'cpspr': g[50] - g[-50], 'skew': g[-25] - iv, 'ivrv': iv - rv, 'div': iv - lag(iv, 5)}
    return {k: lag(v, 1) for k, v in out.items()}


def fundamental_features(fq, dates, names, mcap):
    fq = fq.dropna(subset=['rdq']).copy()
    fq['permno'] = fq.permno.astype(int)
    for c in ['atq', 'saleq', 'ibq']:
        fq[c] = fq[c].astype(float)
    fq = fq.sort_values(['permno', 'datadate']).drop_duplicates(['permno', 'datadate'], keep='last')
    g = fq.groupby('permno')
    full = (fq.datadate - g.datadate.shift(3)).dt.days.between(250, 290)
    for c in ['ibq', 'saleq']:
        fq[c + '_ttm'] = g[c].transform(lambda x: x.rolling(4).sum()).where(full)
    g = fq.groupby('permno')
    year = (fq.datadate - g.datadate.shift(4)).dt.days.between(350, 380)
    fq['ibq_ttm_lag4'] = g.ibq_ttm.shift(4).where(year)
    fq['saleq_ttm_lag4'] = g.saleq_ttm.shift(4).where(year)
    fq['avail'] = fq.rdq + pd.Timedelta(days=1)
    fq = fq.loc[fq.permno.isin(names)]

    panel = pd.DataFrame({'date': np.repeat(dates, len(names)), 'permno': np.tile(names, len(dates))})
    panel = pd.merge_asof(panel.sort_values('date'), fq.sort_values('avail'),
                          left_on='date', right_on='avail', by='permno')
    panel = panel.set_index(['date', 'permno']).reindex(pd.MultiIndex.from_product([dates, names]))
    fresh = ((panel.index.get_level_values(0) - panel.avail).dt.days < MAX_STALE).to_numpy()
    shape = (len(dates), len(names))
    col = lambda c: np.where(fresh, panel[c].to_numpy(dtype=float), np.nan).reshape(shape)
    mc = mcap / 1000                                                    # CRSP $000s -> Compustat $mm
    at = col('atq')
    with np.errstate(divide='ignore', invalid='ignore'):
        out = {'ep': col('ibq_ttm') / mc,
               'roa': col('ibq_ttm') / at,
               'dearn': (col('ibq_ttm') - col('ibq_ttm_lag4')) / at,
               'sg': col('saleq_ttm') / col('saleq_ttm_lag4') - 1}
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def zscore(X, M):
    """Winsorize at 1/99 percentiles and z-score each row over members; non-members NaN, missing 0."""
    X = np.where(M & np.isfinite(X), X, np.nan)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lo, hi = np.nanpercentile(X, [1, 99], axis=1, keepdims=True)
            X = np.clip(X, lo, hi)
            X = (X - np.nanmean(X, 1, keepdims=True)) / np.nanstd(X, 1, keepdims=True)
    return np.where(M, np.nan_to_num(X), np.nan)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def ols_forecast(Z, Rv, M):
    """mu[t] = Z[t] beta_t, beta_t = OLS on pairs (Z[s], de-meaned R[s+1]) for s = t-TRAIN_DAYS .. t-1."""
    n_dates, _, p = Z.shape
    XtX, Xty = np.zeros((n_dates, p, p)), np.zeros((n_dates, p))
    for s in range(n_dates - 1):
        ok = M[s] & np.isfinite(Z[s]).all(1) & np.isfinite(Rv[s + 1])
        if ok.sum() >= MIN_NAMES:
            X, y = Z[s, ok], Rv[s + 1, ok] - Rv[s + 1, ok].mean()
            XtX[s], Xty[s] = X.T @ X, X.T @ y
    cXtX, cXty = np.cumsum(XtX, 0), np.cumsum(Xty, 0)
    mu, betas = np.full(Z.shape[:2], np.nan), {}
    for t in range(TRAIN_DAYS + 1, n_dates):
        A = cXtX[t - 1] - cXtX[t - TRAIN_DAYS - 1]
        b = cXty[t - 1] - cXty[t - TRAIN_DAYS - 1]
        if np.linalg.matrix_rank(A) < p:
            continue
        beta = np.linalg.solve(A, b)
        betas[t] = beta
        ok = M[t] & np.isfinite(Z[t]).all(1)
        mu[t, ok] = Z[t, ok] @ beta
    return mu, pd.DataFrame(betas, index=FEATURES).T


def smooth(mu, halflife):
    """Exponential moving average of each name's forecast over the days it has one."""
    a = 1 - 0.5 ** (1 / halflife)
    out, s = np.full(mu.shape, np.nan), np.zeros(mu.shape[1])
    for t in range(len(mu)):
        ok = np.isfinite(mu[t])
        s = np.where(ok, (1 - a) * s + a * np.nan_to_num(mu[t]), s)
        out[t] = np.where(ok, s, np.nan)
    return out


# ---------------------------------------------------------------------------
# Covariance estimate, MVO and backtest
# ---------------------------------------------------------------------------

def two_stage_cov(X, k):
    """Sigma = B diag(F) B' + diag(D) from X (n x T, de-meaned): top k SVD factors plus residual variances."""
    T = X.shape[1]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    B, F = U[:, :k], s[:k] ** 2 / (T - 1)
    E = X - B @ (s[:k, None] * Vt[:k])
    return B, F, (E ** 2).sum(1) / (T - 1)


def factor_solve(B, F, D, V):
    """Sigma^-1 V by the Woodbury identity."""
    DV, DB = V / D[:, None], B / D[:, None]
    return DV - DB @ np.linalg.solve(np.diag(1 / F) + B.T @ DB, B.T @ DV)


def factor_var(B, F, D, w):
    return ((B.T @ w) ** 2 * F).sum() + (w ** 2 * D).sum()


def stratified_weights(mu):
    q = pd.qcut(mu, 5, labels=False, duplicates='drop')
    w = np.zeros(len(mu))
    top, bot = q == q.max(), q == q.min()
    w[top], w[bot] = 1 / top.sum(), -1 / bot.sum()
    return w


def mvo_weights(mu, B, F, D):
    X = factor_solve(B, F, D, np.column_stack([mu, np.ones(len(mu))]))
    w = X[:, 0] - X[:, 0].sum() / X[:, 1].sum() * X[:, 1]
    return w * TARGET_VOL / np.sqrt(252 * factor_var(B, F, D, w))


def backtest(targets, Rv):
    """Daily returns of positions set to targets[t] at the close of t, drifting with prices.

    Returns daily returns, dollars traded (charged the following day) and gross position.
    """
    n_dates, n_names = Rv.shape
    h = np.zeros(n_names)
    r, traded, gross = np.zeros(n_dates), np.zeros(n_dates), np.zeros(n_dates)
    for t in range(n_dates):
        x = np.nan_to_num(Rv[t])
        r[t] = h @ x
        gross[t] = np.abs(h).sum()
        h = h * (1 + x)
        if t in targets:
            traded[t] = np.abs(targets[t] - h).sum()
            h = targets[t].copy()
    return r, np.concatenate([[0.0], traded[:-1]]), gross


def stats(r, traded, gross):
    out = {}
    for name, x in [('gross', r), (f'net {COST_BPS} bps', r - COST_BPS * 1e-4 * traded)]:
        out[(name, 'mean_ann')] = x.mean() * 252
        out[(name, 'vol_ann')] = x.std() * np.sqrt(252)
        out[(name, 'sharpe')] = x.mean() / x.std() * np.sqrt(252)
    on = gross > 0
    out[('', 'gross_lev')] = gross[on].mean()
    out[('', 'turnover/day')] = 0.5 * traded[on].sum() / gross[on].sum()
    return out


# ---------------------------------------------------------------------------
# Data, forecast and daily covariance estimates (cached)
# ---------------------------------------------------------------------------

def build(members, dsf, fq, opt):
    dsf['permno'] = dsf.permno.astype(int)
    opt['permno'] = opt.permno.astype(int)
    dates = pd.DatetimeIndex(sorted(dsf.date.unique()))
    names = pd.Index(sorted(dsf.permno.unique()))
    Rv = wide(dsf, 'ret', dates, names)
    P, V, S = (wide(dsf, c, dates, names) for c in ['prc', 'vol', 'shrout'])
    M = membership(members, dates, names) & np.isfinite(Rv)
    raw = price_volume_features(Rv, P, V, S)
    raw.update(option_features(opt, dates, names, raw['rv']))
    raw.update(fundamental_features(fq, dates, names, P * S))
    Z = np.stack([zscore(raw[f], M) for f in FEATURES], -1)
    return dates, names, Rv, M, Z


def prepare(refresh=False):
    """Returns, membership, forecasts and the daily covariance estimates, cached in FORECAST_DATA."""
    if refresh or not os.path.exists(FORECAST_DATA):
        dates, names, Rv, M, Z = build(*load_data())
        raw_mu, betas = ols_forecast(Z, Rv, M)
        mu = smooth(raw_mu, HALFLIFE)
        covs = {}
        for t in range(FIRST, len(dates) - 1):
            win = Rv[t - COV_WINDOW + 1:t + 1].T                           # n x T, returns up to t
            ok = M[t] & (np.isfinite(win).mean(1) >= MIN_OBS) & np.isfinite(mu[t])
            if ok.sum() < MIN_NAMES:
                continue
            idx = np.flatnonzero(ok)
            X = win[ok] - np.nanmean(win[ok], axis=1, keepdims=True)
            covs[t] = (idx,) + two_stage_cov(np.nan_to_num(X), N_FACTORS)
        pd.to_pickle({'dates': dates, 'names': names, 'Rv': Rv, 'raw_mu': raw_mu, 'mu': mu,
                      'betas': betas, 'covs': covs}, FORECAST_DATA)
    return pd.read_pickle(FORECAST_DATA)


# ---------------------------------------------------------------------------
# Portfolio construction and evaluation
# ---------------------------------------------------------------------------

def run_portfolio(mode, d):
    """Daily targets for one mode, with the quantity used to sort names into quintiles.

    The sort variable is the forecast mu for 'stratified' and the MVO weight for 'mvo'
    (names grouped by what MVO actually holds).
    """
    dates, names, Rv, mu, covs = d['dates'], d['names'], d['Rv'], d['mu'], d['covs']
    targets, sortvar, split, pred = {}, {}, [], []
    held = np.zeros(len(names))
    for t, (idx, B, F, D) in covs.items():
        m = mu[t, idx]
        w = stratified_weights(pd.Series(m)) if mode == 'stratified' else mvo_weights(m, B, F, D)
        sortvar[t] = (idx, m if mode == 'stratified' else w)
        pred.append(np.sqrt(252 * factor_var(B, F, D, w)))
        full = np.zeros(len(names))
        full[idx] = w
        held = TRADE_RATE * full + (1 - TRADE_RATE) * held * (1 + np.nan_to_num(Rv[t]))
        targets[t] = held
        r1 = np.nan_to_num(Rv[t + 1, idx])
        f = held[idx] @ B @ (B.T @ r1)                                  # projection on the estimated factors
        split.append((dates[t + 1], f, held[idx] @ r1 - f))
    split = pd.DataFrame(split, columns=['date', 'factor', 'residual']).set_index('date')
    return targets, sortvar, split, np.mean(pred)


def quintile_returns(sortvar, Rv, dates):
    """Daily next-day return of each equal-weighted quintile, minus the mean over all names."""
    out = {}
    for t, (idx, s) in sortvar.items():
        r = Rv[t + 1, idx]
        ok = np.isfinite(r)
        q = pd.qcut(pd.Series(s[ok]).rank(method='first'), 5, labels=False).to_numpy()
        out[dates[t + 1]] = [r[ok][q == k].mean() - r[ok].mean() for k in range(5)]
    return pd.DataFrame(out, index=[f'Q{k + 1}' for k in range(5)]).T


def ann(x):
    return {'mean_ann': x.mean() * 252, 'vol_ann': x.std() * np.sqrt(252),
            'sharpe': x.mean() / x.std() * np.sqrt(252)}


def performance(mode, d):
    dates, Rv = d['dates'], d['Rv']
    targets, sortvar, split, pred_vol = run_portfolio(mode, d)
    r, traded, gross = backtest(targets, Rv)
    start = min(targets) + 1
    idx = dates[start:]
    pnl = pd.DataFrame({'gross': r[start:], f'net {COST_BPS} bps': r[start:] - COST_BPS * 1e-4 * traded[start:]},
                       index=idx)
    on = gross[start:] > 0
    summary = pd.DataFrame({c: ann(pnl[c]) for c in pnl}).T
    summary = pd.concat([summary, pd.DataFrame({c: ann(split[c]) for c in split}).T])
    summary.index = ['gross', f'net {COST_BPS} bps', 'factor part', 'residual part']
    info = pd.Series({'gross leverage': gross[start:][on].mean(),
                      'turnover / day': 0.5 * traded[start:][on].sum() / gross[start:][on].sum(),
                      'predicted vol': pred_vol,
                      'realized vol': pnl['gross'].std() * np.sqrt(252)})
    yearly = pnl.groupby(pnl.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
    quint = quintile_returns(sortvar, Rv, dates)
    return pnl, summary, info, yearly, quint


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_quintiles(quint, mode, path):
    sortname = 'forecast' if mode == 'stratified' else 'MVO weight'
    colors = ['firebrick', 'coral', 'grey', 'skyblue', 'navy']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
    for c, col in zip(quint.columns, colors):
        a1.plot(quint[c].cumsum(), color=col, linewidth=2.5, label=c)
    a1.axhline(0, color='black', linewidth=1)
    a1.set_title(f'Cumulative return of quintiles of the {sortname}\n(equal-weighted, minus the cross-sectional mean)',
                 fontsize=16)
    a1.set_xlabel('Date', fontsize=14)
    a1.set_ylabel('Cumulative return', fontsize=14)
    a1.legend(fontsize=12, title=f'Q5 = highest {sortname}')
    m = quint.mean() * 252
    se = quint.std() * np.sqrt(252) / np.sqrt(len(quint) / 252)
    a2.bar(m.index, m, yerr=2 * se, color=colors, capsize=6)
    a2.axhline(0, color='black', linewidth=1)
    a2.set_title('Mean annualized return\n(bars: plus or minus 2 standard errors)', fontsize=16)
    a2.tick_params(labelsize=12)
    fig.suptitle(f'{mode.upper()}: quintiles, {quint.index[0].year}-{quint.index[-1].year}', fontsize=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity(pnl, summary, mode, path):
    fig, ax = plt.subplots(figsize=(16, 9))
    for c, col in zip(pnl.columns, ['navy', 'coral']):
        ax.plot(pnl[c].cumsum(), color=col, linewidth=3,
                label=f'{c} (Sharpe {summary.loc[c, "sharpe"]:.2f}, vol {summary.loc[c, "vol_ann"]:.1%})')
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_title(f'{mode.upper()}: long-short portfolio, daily rebalancing\nCumulative return (sum of daily returns)',
                 fontsize=20)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Cumulative return', fontsize=16)
    ax.legend(loc='best', fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(mode='stratified', refresh=False):
    d = prepare(refresh)
    dates, Rv = d['dates'], d['Rv']
    pnl, summary, info, yearly, quint = performance(mode, d)

    print(f'{mode.upper()}: {pnl.index[0].date()} to {pnl.index[-1].date()}, daily forecast and daily '
          f'two-stage covariance (k = {N_FACTORS}, {COV_WINDOW} days)')
    print(f'Forecast: OLS on {len(FEATURES)} features, {TRAIN_DAYS}-day window, half-life {HALFLIFE} days. '
          f'Trading {TRADE_RATE:.0%} of the way to target each day, costs {COST_BPS} bps.\n')
    for name, x in [('raw', d['raw_mu']), ('smoothed', d['mu'])]:
        ic = np.array([spearmanr(x[t, ok], Rv[t + 1, ok])[0] for t in d['covs']
                       for ok in [np.isfinite(x[t]) & np.isfinite(Rv[t + 1])]])
        print(f'{name:8s} forecast daily rank IC: mean {ic.mean():.4f}, t {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}')
    print('\nPerformance')
    print(summary.round(3).to_string())
    print()
    print(info.round(3).to_string())
    print('\nQuintiles (annualized mean return minus cross-sectional mean)')
    print((quint.mean() * 252).round(3).to_frame('mean_ann').T.to_string())
    print('\nSharpe by year')
    print(yearly.round(2).T.to_string())

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = {'quintiles': os.path.join(PLOT_DIR, f'fundamentals_{mode}_quintiles.png'),
           'equity': os.path.join(PLOT_DIR, f'fundamentals_{mode}_equity.png'),
           'table': os.path.join(PLOT_DIR, f'fundamentals_{mode}_performance.csv')}
    plot_quintiles(quint, mode, out['quintiles'])
    plot_equity(pnl, summary, mode, out['equity'])
    pd.concat([summary, info.to_frame('value')], axis=0).to_csv(out['table'])
    print('\nSaved:', *out.values(), sep='\n  ')
    return pnl, summary, info, yearly, quint


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=['stratified', 'mvo'])
    parser.add_argument('--refresh', action='store_true', help='rebuild the cached features and forecasts')
    args = parser.parse_args()
    main(args.mode, args.refresh)
