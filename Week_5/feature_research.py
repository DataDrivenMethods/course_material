"""
feature_research.py

Information-coefficient (IC) and mutual-information analysis for
panel data produced by build_daily_panel().

Usage
-----
    from daily_fundamental_signals import build_daily_panel, load_panel
    from feature_research import compute_ics, compute_mutual_information

    panel, fcols, ffreq = load_panel('2015-01-01', '2020-12-31',
                                      days_ahead=[1, 5, 10, 20])

    ic = compute_ics(panel, fcols)
    mi = compute_mutual_information(panel, fcols)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _fwd_cols(panel):
    """Return sorted list of fwd_*d_ret columns present in the panel."""
    return sorted(
        [c for c in panel.columns if c.startswith('fwd_') and c.endswith('d_ret')],
        key=lambda c: int(c.split('_')[1].rstrip('d')),
    )


# ──────────────────────────────────────────────────────────────────────
# Information Coefficients (Spearman rank correlation)
# ──────────────────────────────────────────────────────────────────────

def compute_ics(panel, feature_cols, fwd_ret_cols=None, by_date=False):
    """
    Cross-sectional Spearman IC of each feature against each forward return.

    Parameters
    ----------
    panel : pd.DataFrame
        From build_daily_panel (must contain feature and fwd_*d_ret columns).
    feature_cols : list[str]
        Feature column names.
    fwd_ret_cols : list[str] or None
        Forward-return columns to correlate against.
        If None, auto-detects all fwd_*d_ret columns.
    by_date : bool
        If False (default), compute a single pooled IC per (feature, horizon).
        If True, compute IC per date and return the daily series plus a
        summary with mean, std, t-stat, and hit-rate.

    Returns
    -------
    If by_date=False:
        pd.DataFrame — index: feature, columns: fwd_ret horizon.
    If by_date=True:
        (daily_ics, summary) where
            daily_ics : dict[str, pd.DataFrame]  keyed by fwd_ret col,
                each DataFrame has index=date, columns=features.
            summary : pd.DataFrame with MultiIndex (horizon, stat)
                and columns=features.
    """
    if fwd_ret_cols is None:
        fwd_ret_cols = _fwd_cols(panel)
    if not fwd_ret_cols:
        raise ValueError("No fwd_*d_ret columns found in panel")

    df = panel[['permno', 'date'] + feature_cols + fwd_ret_cols].copy()

    if not by_date:
        # ── Pooled IC (pairwise complete observations) ──
        results = {}
        for fwd in fwd_ret_cols:
            ics = {}
            for feat in feature_cols:
                pair = df[[feat, fwd]].dropna()
                if len(pair) < 10:
                    ics[feat] = np.nan
                else:
                    rho, _ = spearmanr(pair[feat], pair[fwd])
                    ics[feat] = rho
            results[fwd] = ics
        return pd.DataFrame(results, index=feature_cols)

    # ── Daily cross-sectional IC ──
    dates = sorted(df['date'].unique())
    daily_ics = {fwd: {} for fwd in fwd_ret_cols}

    for dt in dates:
        snap = df[df['date'] == dt]
        if len(snap) < 5:
            continue
        for fwd in fwd_ret_cols:
            row = {}
            y = snap[fwd].values
            for feat in feature_cols:
                x = snap[feat].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 5:
                    row[feat] = np.nan
                else:
                    rho, _ = spearmanr(x[mask], y[mask])
                    row[feat] = rho
            daily_ics[fwd][dt] = row

    # Convert to DataFrames
    for fwd in fwd_ret_cols:
        daily_ics[fwd] = pd.DataFrame(daily_ics[fwd]).T
        daily_ics[fwd].index.name = 'date'

    # Summary statistics
    summary_parts = []
    for fwd in fwd_ret_cols:
        ic_df = daily_ics[fwd]
        mean = ic_df.mean()
        std = ic_df.std()
        n = ic_df.count()
        tstat = mean / (std / np.sqrt(n))
        hit = (ic_df > 0).mean()
        part = pd.DataFrame({
            'mean_ic': mean, 'std_ic': std,
            'tstat': tstat, 'hit_rate': hit,
        })
        part.index.name = 'feature'
        summary_parts.append(part.assign(horizon=fwd))

    summary = pd.concat(summary_parts).set_index('horizon', append=True)
    summary = summary.reorder_levels(['horizon', 'feature'])

    return daily_ics, summary


# ──────────────────────────────────────────────────────────────────────
# Mutual Information
# ──────────────────────────────────────────────────────────────────────

def compute_mutual_information(panel, feature_cols, fwd_ret_cols=None,
                               n_neighbors=5, random_state=42,
                               max_rows=200_000):
    """
    Mutual information between each feature and each forward return.

    Uses sklearn's mutual_info_regression (k-NN estimator, Kraskov et al.).

    Parameters
    ----------
    panel : pd.DataFrame
    feature_cols : list[str]
    fwd_ret_cols : list[str] or None
        If None, auto-detects all fwd_*d_ret columns.
    n_neighbors : int
        Number of neighbours for MI estimator (higher = smoother, slower).
    random_state : int
    max_rows : int
        If the panel exceeds this many rows after dropna, subsample
        to keep runtime reasonable.

    Returns
    -------
    pd.DataFrame — index: feature, columns: fwd_ret horizon.
        Values are MI in nats (non-negative; 0 = independent).
    """
    if fwd_ret_cols is None:
        fwd_ret_cols = _fwd_cols(panel)
    if not fwd_ret_cols:
        raise ValueError("No fwd_*d_ret columns found in panel")

    df = panel[feature_cols + fwd_ret_cols].copy()
    # Drop rows missing the target, but fill feature NaNs with median
    # (MI needs a complete matrix; listwise deletion wastes too much data)
    df = df.dropna(subset=fwd_ret_cols)
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
        print(f"  MI: subsampled to {max_rows:,} rows")

    X = df[feature_cols].values

    results = {}
    for fwd in fwd_ret_cols:
        y = df[fwd].values
        mi = mutual_info_regression(
            X, y, n_neighbors=n_neighbors, random_state=random_state,
        )
        results[fwd] = dict(zip(feature_cols, mi))
        print(f"  MI vs {fwd}: done")

    return pd.DataFrame(results, index=feature_cols)
