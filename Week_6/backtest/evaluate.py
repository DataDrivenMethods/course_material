"""Information coefficients of a signal against forward returns.

Timing (SPEC Section 1): the signal on day d is traded at the close of d, so it is
paired with returns from d+1 on. Everything here works on date x permno
matrices on the CRSP return calendar; "fwd_h on row d" is the compounded return
over d+1..d+h, and the single-day return at lag k on row d is R[d+k].
check_timing() verifies this pairing with synthetic signals.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .returns import wide_returns

MIN_NAMES_IC = 20       # minimum stock count for a daily IC


def n_col_for(col: str) -> str:
    """Story-count column paired with a signal column: sent_5 -> n_5, S -> n."""
    return 'n' if col == 'S' else 'n_' + col.split('_')[-1]


# ---------------------------------------------------------------- matrices

def wide_signal(signal: pd.DataFrame, col: str, news_only: bool = True,
                n_col: str | None = None, index=None, columns=None) -> pd.DataFrame:
    """date x permno matrix of `col`; NaN where n_col == 0 if news_only."""
    s = signal
    if news_only:
        s = s[s[n_col or n_col_for(col)] > 0]
    X = s.pivot(index='date', columns='permno', values=col)
    if index is not None:
        extra = X.index.difference(index)
        if len(extra):
            raise ValueError(f'{len(extra)} signal dates are not trading days, e.g. {extra[0]}')
        X = X.reindex(index=index)
    if columns is not None:
        X = X.reindex(columns=columns)
    return X.astype('float64')


def fwd_matrix(R: pd.DataFrame, alive: pd.DataFrame, h: int) -> pd.DataFrame:
    """Row d: compounded return over d+1..d+h; NaN if the stock is not alive on d."""
    L = np.log1p(R.clip(lower=-0.999999))
    F = np.expm1(L.rolling(h, min_periods=h).sum().shift(-h))
    return F.where(alive)


def lag_matrix(R: pd.DataFrame, alive: pd.DataFrame, k: int) -> pd.DataFrame:
    """Row d: single-day return on d+k; NaN if the stock is not alive on d."""
    return R.shift(-k).where(alive)


def rowwise_corr(X: pd.DataFrame, Y: pd.DataFrame, method: str = 'spearman',
                 min_obs: int = MIN_NAMES_IC) -> pd.Series:
    """Cross-sectional correlation of X and Y on each row (pairwise-complete)."""
    Y = Y.reindex_like(X)
    mask = X.notna().to_numpy() & Y.notna().to_numpy()
    x = np.where(mask, X.to_numpy(), np.nan)
    y = np.where(mask, Y.to_numpy(), np.nan)
    if method == 'spearman':
        x = pd.DataFrame(x).rank(axis=1).to_numpy()       # average ranks for ties
        y = pd.DataFrame(y).rank(axis=1).to_numpy()
    elif method != 'pearson':
        raise ValueError(method)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        xm = x - np.nanmean(x, axis=1, keepdims=True)
        ym = y - np.nanmean(y, axis=1, keepdims=True)
        r = np.nansum(xm * ym, 1) / np.sqrt(np.nansum(xm ** 2, 1) * np.nansum(ym ** 2, 1))
    n = mask.sum(1)
    r[n < min_obs] = np.nan
    return pd.Series(r, index=X.index)


# ---------------------------------------------------------------- public API

def forward_returns(returns: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """[date, permno, fwd_h...]: compounded return from close d to close d+h.

    Only stock-days on which the stock trades (alive) are returned. After a
    delisting the stock earns 0 (cash), so fwd_h spanning a delisting includes
    the delisting return and nothing after it.
    """
    R, _, alive = wide_returns(returns)
    cols = {f'fwd_{h}': fwd_matrix(R, alive, h).stack() for h in horizons}
    out = pd.DataFrame(cols)
    out.index.names = ['date', 'permno']
    return out.reset_index()


def daily_ic(signal: pd.DataFrame, col: str, fwd: pd.DataFrame, horizon: int,
             method: str = 'spearman', news_only: bool = True) -> pd.Series:
    """One cross-sectional IC per signal date between col on d and fwd_{horizon} on d.

    fwd: long output of forward_returns (or a date x permno matrix of fwd_h).
    """
    if 'permno' in fwd.columns:
        F = fwd.pivot(index='date', columns='permno', values=f'fwd_{horizon}')
    else:
        F = fwd
    X = wide_signal(signal, col, news_only, index=F.index, columns=F.columns)
    ic = rowwise_corr(X, F, method)
    d = pd.DatetimeIndex(signal['date'].unique())
    return ic.reindex(d.sort_values()).rename(f'ic_{horizon}')


def nw_se(x: pd.Series, lags: int) -> float:
    """Newey-West standard error of the mean of x."""
    x = x.dropna()
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(
        cov_type='HAC', cov_kwds={'maxlags': max(int(lags), 0)})
    return float(res.bse[0])


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """mean, std, IR (daily and annualized), NW t-stat (lags = horizon - 1), hit rate,
    n_days, by-year means."""
    x = ic.dropna()
    mean, std = x.mean(), x.std()
    se = nw_se(x, horizon - 1)
    out = {'horizon': horizon, 'mean': mean, 'std': std, 'ir': mean / std,
           'ir_ann': mean / std * np.sqrt(252 / horizon),
           'nw_se': se, 'nw_t': mean / se, 'hit_rate': (x > 0).mean(), 'n_days': len(x)}
    for y, m in x.groupby(x.index.year).mean().items():
        out[f'mean_{y}'] = m
    return out


def ic_decay(signal: pd.DataFrame, col: str, returns: pd.DataFrame,
             lags: range = range(1, 21), method: str = 'spearman',
             news_only: bool = True) -> pd.DataFrame:
    """IC of the signal on d against the single-day return on d+k (lag) and the
    compounded return over d+1..d+k (cum). NW s.e.: lag 0 for single-day,
    k - 1 for cumulative (overlapping) returns."""
    R, _, alive = wide_returns(returns)
    X = wide_signal(signal, col, news_only, index=R.index, columns=R.columns)
    days = X.index.isin(pd.DatetimeIndex(signal['date'].unique()))
    rows = []
    for k in lags:
        lag = rowwise_corr(X, lag_matrix(R, alive, k), method)[days].dropna()
        cum = rowwise_corr(X, fwd_matrix(R, alive, k), method)[days].dropna()
        rows.append({'k': k, 'ic_lag': lag.mean(), 'se_lag': nw_se(lag, 0),
                     'ic_cum': cum.mean(), 'se_cum': nw_se(cum, k - 1)})
    out = pd.DataFrame(rows).set_index('k')
    out['t_lag'] = out['ic_lag'] / out['se_lag']
    out['t_cum'] = out['ic_cum'] / out['se_cum']
    return out


def check_timing(returns: pd.DataFrame, start=None, end=None) -> dict:
    """Off-by-one test of the signal/return pairing through the public functions.

    A signal equal to the day-d return is known at the close of d, so its IC with
    fwd_1 (return on d+1) must be ~0 (only short-term reversal). A signal equal to
    the day d+1 return is look-ahead by construction and must have IC ~1.
    """
    R, _, alive = wide_returns(returns)
    days = R.loc[start:end].index

    def as_signal(M):
        s = M.loc[days].where(alive.loc[days]).stack().rename('sent_1').reset_index()
        s.columns = ['date', 'permno', 'sent_1']
        s['n_1'] = 1
        return s

    fwd = fwd_matrix(R, alive, 1)
    ic_now = daily_ic(as_signal(R), 'sent_1', fwd, 1).mean()
    ic_next = daily_ic(as_signal(R.shift(-1)), 'sent_1', fwd, 1).mean()
    assert abs(ic_now) < 0.1, f'timing: IC of same-day return = {ic_now:.3f}, expected ~0'
    assert ic_next > 0.99, f'timing: IC of next-day return = {ic_next:.3f}, expected ~1'
    return {'ic_same_day_return': ic_now, 'ic_next_day_return': ic_next}
