"""Quantile portfolio simulation.

Portfolio formed at the close of d from the signal on d, earns returns on d+1, ...
Weights live in date x permno matrices on the CRSP return calendar: W[d] is the
target held from the close of d, so the return on day t is sum(W[t-1] * R[t]).
"""

import numpy as np
import pandas as pd

from sentlib.portfolio import assign_quantiles

from .returns import wide_returns


def tradable_rows(signal: pd.DataFrame, alive: pd.DataFrame) -> np.ndarray:
    """Boolean mask of signal rows whose stock trades on that day (can be bought at the close)."""
    ri = alive.index.get_indexer(signal['date'])
    ci = alive.columns.get_indexer(signal['permno'])
    ok = (ri >= 0) & (ci >= 0)
    ok[ok] = alive.to_numpy()[ri[ok], ci[ok]]
    return ok


def quantile_portfolios(signal: pd.DataFrame, col: str, returns: pd.DataFrame,
                        n_quantiles: int = 5, holding: int = 1, weight: str = 'equal',
                        news_only: bool = True, cost_bps: float = 0.0) -> dict:
    """Daily returns of signal-sorted quantile portfolios and the long-short (q - 1).

    holding h > 1: average of h overlapping sub-portfolios formed on d-1..d-h
    (Jegadeesh-Titman), each held at its target weights.
    weight 'cap': prevcap on d, i.e. market cap at the close of d-1.
    Stocks that delist while held earn the delisting return, then 0 (cash).
    Names with a missing return on a day are dropped and the rest renormalized.

    Turnover: one-way, 0.5 * sum|W_target[d] - W_drift[d]|, where W_drift is
    yesterday's weights after today's returns. Trading costs are cost_bps per
    dollar traded, i.e. cost_bps * sum|dW| (= 2 x one-way turnover) per leg,
    charged to the next day's return. Long-short net pays both legs.

    Returns dict: quantiles, quantiles_net, long_short, long_short_net (Series),
    turnover (one-way, per quantile and 'ls' = mean of the two legs),
    n_names (per formation date), quantile_labels (Series aligned with signal).
    """
    R, cap, alive = wide_returns(returns)
    sig = signal[tradable_rows(signal, alive)]
    q = assign_quantiles(sig, col, n_quantiles, news_only=news_only)
    Q = (sig[['date', 'permno']].assign(q=q.astype('float64'))
         .pivot(index='date', columns='permno', values='q')
         .reindex(index=R.index, columns=R.columns))
    formed = Q.notna().any(axis=1)               # formation dates
    form_days = np.flatnonzero(formed.to_numpy())
    first, last = form_days[0], form_days[-1]
    # Return days t with all h sub-portfolios formed on signal dates
    t_idx = np.arange(first + holding, min(last + 1, len(R) - 1) + 1)
    days = R.index[t_idx]

    Rv = R.to_numpy()
    valid = ~np.isnan(Rv)
    Rz = np.where(valid, Rv, 0.0)
    gross, traded, names = {}, {}, {}
    for j in range(1, n_quantiles + 1):
        M = (Q.to_numpy() == j)
        if weight == 'equal':
            base = M.astype('float64')
        elif weight == 'cap':
            base = np.where(M, np.nan_to_num(cap.to_numpy(), nan=0.0), 0.0)
        else:
            raise ValueError(weight)
        tot = base.sum(1, keepdims=True)
        W = np.divide(base, tot, out=np.zeros_like(base), where=tot > 0)
        if holding > 1:
            W = pd.DataFrame(W).rolling(holding, min_periods=1).sum().to_numpy() / holding
        # Return on day t from weights held since the close of t-1
        Wp = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
        Wv = np.where(valid, Wp, 0.0)
        wsum = Wv.sum(1)
        r = np.divide((Wv * Rz).sum(1), wsum, out=np.full(len(wsum), np.nan), where=wsum > 0)
        # Drifted weights at the close of t, then the trade to W[t]
        drift = Wp * (1 + Rz) / (1 + np.nan_to_num(r))[:, None]
        trade = np.abs(W - drift).sum(1)              # sum |dW| at the close of t
        trade_prev = np.concatenate([[np.nan], trade[:-1]])  # charged on day t+1
        gross[j] = r[t_idx]
        traded[j] = trade_prev[t_idx]
        names[j] = M.sum(1)

    qret = pd.DataFrame({f'q{j}': gross[j] for j in gross}, index=days)
    qtrade = pd.DataFrame({f'q{j}': traded[j] for j in traded}, index=days)
    c = cost_bps * 1e-4
    top, bot = f'q{n_quantiles}', 'q1'
    ls = (qret[top] - qret[bot]).rename('long_short')
    ls_net = (ls - c * (qtrade[top] + qtrade[bot])).rename('long_short_net')
    turnover = 0.5 * qtrade
    turnover['ls'] = 0.5 * (turnover[top] + turnover[bot])
    n_names = pd.DataFrame({f'q{j}': names[j] for j in names}, index=R.index)[formed]
    return {'quantiles': qret, 'quantiles_net': qret - c * qtrade,
            'long_short': ls, 'long_short_net': ls_net, 'turnover': turnover,
            'n_names': n_names, 'cost_bps': cost_bps, 'holding': holding,
            'quantile_labels': q}


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def performance_table(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """One row per column: annualized return (252 x mean, and CAGR), vol, Sharpe
    (252 days, no risk-free rate), max drawdown, hit rate, n_days."""
    rows = {}
    for c, r in daily_returns.items():
        r = r.dropna()
        rows[c] = {'ann_ret': r.mean() * 252,
                   'cagr': (1 + r).prod() ** (252 / len(r)) - 1,
                   'ann_vol': r.std() * np.sqrt(252),
                   'sharpe': r.mean() / r.std() * np.sqrt(252),
                   'max_dd': max_drawdown(r),
                   'hit_rate': (r > 0).mean(),
                   'n_days': len(r)}
    return pd.DataFrame(rows).T
