"""Cross-sectional quantile assignment for signal sorts."""

import numpy as np
import pandas as pd


ZERO_TOL = 1e-9     # |signal| below this counts as zero sentiment


def default_n_col(col: str) -> str:
    """Story-count column paired with a signal column: sent_5 -> n_5, S -> n."""
    if col == 'S':
        return 'n'
    return 'n_' + col.split('_')[-1]


def assign_quantiles(signal: pd.DataFrame, col: str, n_quantiles: int = 5,
                     news_only: bool = True, n_col: str | None = None,
                     min_names: int | None = None, ties: str = 'random',
                     seed: int = 0) -> pd.Series:
    """Quantile of `col` within each date, split by sign: 1 = most negative,
    n_quantiles = most positive.

    Per date, among eligible names (col not NaN, and n_col > 0 if news_only):
      * col == 0 (|col| < ZERO_TOL) goes to the middle quantile (n_quantiles odd) or is excluded
        (n_quantiles even), so stocks without sentiment never fill the extremes;
      * col < 0 is split into equal-count buckets 1..k, most negative in 1;
      * col > 0 is split into equal-count buckets n_quantiles-k+1..n_quantiles,
        most positive in n_quantiles;
    where k = n_quantiles // 2. With 5 quantiles: negatives -> Q1, Q2; zeros -> Q3;
    positives -> Q4, Q5. Ties among nonzero values are broken by ties='random'
    (a seeded uniform key drawn in (date, permno) order, so deterministic across
    runs) or ties='first' (row order).
    Returns an Int64 Series aligned with `signal`; <NA> for excluded rows and
    for dates with fewer than min_names (default 10 * n_quantiles) eligible names.
    """
    if n_quantiles < 2:
        raise ValueError('n_quantiles must be at least 2')
    if ties not in ('random', 'first'):
        raise ValueError(f"ties must be 'random' or 'first', got {ties!r}")
    if min_names is None:
        min_names = 10 * n_quantiles
    eligible = signal[col].notna()
    if news_only:
        n_col = n_col or default_n_col(col)
        eligible &= signal[n_col] > 0

    keys = ['date', 'permno'] if 'permno' in signal.columns else ['date']
    sub = signal.loc[eligible, keys + ([col] if col not in keys else [])]
    sub = sub.sort_values(keys, kind='stable')
    tiebreak = (np.random.default_rng(seed).random(len(sub)) if ties == 'random'
                else np.arange(len(sub), dtype='float64'))
    x = sub[col].to_numpy()
    # -1, 0, +1; averages like (0.1 + 0.2 - 0.3) / 3 leave ~1e-17, treated as 0
    side = np.where(np.abs(x) < ZERO_TOL, 0, np.sign(x)).astype('int64')
    # Sort by date, side, signal, tie key; rank = position within (date, side)
    order = np.lexsort((tiebreak, x, side, sub['date'].to_numpy()))
    sub, side = sub.iloc[order], side[order]
    grp = [sub['date'].to_numpy(), side]
    rank = sub.groupby(grp).cumcount().to_numpy() + 1.0
    count = sub.groupby(grp)[col].transform('count').to_numpy()
    total = sub.groupby('date')[col].transform('count').to_numpy()

    # Equal-count buckets within each side: rank r of c names -> floor((r-1) * k / c)
    k = n_quantiles // 2
    b = np.floor((rank - 1) * k / count).astype('int64')
    q = np.where(side < 0, b + 1, np.where(side > 0, n_quantiles - k + 1 + b, (n_quantiles + 1) // 2))
    keep = (total >= min_names) & ~((side == 0) & (n_quantiles % 2 == 0))

    out = pd.Series(pd.NA, index=signal.index, dtype='Int64', name=f'q_{col}')
    out.loc[sub.index[keep]] = q[keep]
    return out
