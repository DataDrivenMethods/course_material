"""Daily stock-level signal from scored stories: aggregation, windows, membership, I/O."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .timing import assign_trading_day
from .wrds_io import DATA_DIR, membership_spells


def window_columns(windows: list[int]) -> list[str]:
    return [c for L in sorted(windows) for c in (f'sent_{L}', f'n_{L}')]


def aggregate_daily(stories: pd.DataFrame, score_col: str, link: pd.DataFrame,
                    dedupe_story: bool = False) -> pd.DataFrame:
    """Per (permno, date): S = mean story score, n = number of stories.

    stories need [date, rp_entity_id, rp_story_id, score_col]; rows with a
    missing date or score are dropped. dedupe_story=True keeps the first row per
    (rp_story_id, rp_entity_id). Entities map to permnos only where the link is
    valid on that date; several entities on one permno are combined story-weighted.
    """
    s = stories[['date', 'rp_entity_id', 'rp_story_id', score_col]].dropna(subset=['date', score_col])
    if dedupe_story:
        s = s.drop_duplicates(['rp_story_id', 'rp_entity_id'])
    daily = (s.groupby(['rp_entity_id', 'date'], observed=True)[score_col]
              .agg(S='mean', n='count').reset_index())

    daily = daily.merge(link, on='rp_entity_id', how='inner')
    daily = daily[(daily['date'] >= daily['link_start']) & (daily['date'] <= daily['link_end'])]
    daily['S_x_n'] = daily['S'] * daily['n']
    daily = daily.groupby(['permno', 'date'], as_index=False)[['S_x_n', 'n']].sum()
    daily['S'] = daily['S_x_n'] / daily['n']
    daily = daily.astype({'permno': 'int64', 'n': 'int64'})
    return daily[['permno', 'date', 'S', 'n']]


def build_panel(daily: pd.DataFrame, permnos: list[int], calendar: pd.DatetimeIndex,
                windows: list[int]) -> pd.DataFrame:
    """Full permno x calendar grid with S, n (0 without news) and sent_L, n_L.

    sent_L(d) = S_d + ... + S_{d-L+1} over trading days; NaN until L days exist.
    """
    grid = pd.MultiIndex.from_product([sorted(permnos), pd.DatetimeIndex(calendar)],
                                      names=['permno', 'date'])
    panel = (daily.set_index(['permno', 'date'])[['S', 'n']]
                  .reindex(grid, fill_value=0).sort_index())
    # Wide (date x permno) so each rolling sum is one vectorized call
    S = panel['S'].unstack('permno')
    n = panel['n'].unstack('permno').astype('float64')
    for L in sorted(windows):
        panel[f'sent_{L}'] = S.rolling(L, min_periods=L).sum().unstack()
        panel[f'n_{L}'] = n.rolling(L, min_periods=L).sum().unstack()
    return panel.reset_index()


def restrict_to_membership(panel: pd.DataFrame, membership: pd.DataFrame,
                           start: str, end: str) -> pd.DataFrame:
    """Keep (permno, date) while the permno is a member and start <= date <= end.

    Returns the SPEC section 3 columns sorted by (date, permno).
    """
    windows = sorted(int(c.split('_')[1]) for c in panel.columns if c.startswith('sent_'))
    members = membership_spells(membership)
    p = panel[(panel['date'] >= pd.Timestamp(start)) & (panel['date'] <= pd.Timestamp(end))]
    sig = p.merge(members, on='permno', how='inner')
    sig = sig[(sig['date'] >= sig['effective_from']) & (sig['date'] <= sig['effective_to'])]
    if sig.duplicated(['date', 'permno']).any():
        raise ValueError('overlapping membership windows give duplicate (date, permno) rows')

    cols = ['date', 'permno', 'permco', 'ticker', 'S', 'n'] + window_columns(windows)
    sig = sig[cols].sort_values(['date', 'permno']).reset_index(drop=True)
    if sig[window_columns(windows)].isna().any().any():
        raise ValueError('incomplete windows inside [start, end]: start the calendar earlier')
    return sig.astype({'permno': 'int64', 'permco': 'int64', 'S': 'float64', 'n': 'int64',
                       **{f'sent_{L}': 'float64' for L in windows},
                       **{f'n_{L}': 'int64' for L in windows}})


def make_signal(stories: pd.DataFrame, score_col: str, link: pd.DataFrame,
                membership: pd.DataFrame, calendar: pd.DatetimeIndex,
                windows: list[int], start: str, end: str,
                cutoff: str = '15:30', dedupe_story: bool = False) -> pd.DataFrame:
    """Stories with timestamp_utc -> member stock-day signal (SPEC section 3).

    The calendar must start at least 2*max(windows)+10 calendar days before start.
    """
    s = stories.copy()
    s['date'] = assign_trading_day(s['timestamp_utc'], calendar, cutoff)
    daily = aggregate_daily(s, score_col, link, dedupe_story=dedupe_story)
    permnos = sorted(membership.dropna(subset=['effective_from'])['permno'].unique().tolist())
    panel = build_panel(daily, permnos, calendar, windows)
    return restrict_to_membership(panel, membership, start, end)


def save_signal(signal: pd.DataFrame, tag: str, name: str, params: dict,
                data_dir: Path = DATA_DIR) -> Path:
    """Write data/{tag}_{name}_signal.parquet and a .json sidecar with params and counts."""
    data_dir = Path(data_dir)
    path = data_dir / f'{tag}_{name}_signal.parquet'
    signal.to_parquet(path, index=False)
    windows = sorted(int(c.split('_')[1]) for c in signal.columns if c.startswith('sent_'))
    meta = {
        'tag': tag, 'name': name, 'file': path.name,
        'created': datetime.now().isoformat(timespec='seconds'),
        'params': params,
        'windows': windows,
        'rows': int(len(signal)),
        'permnos': int(signal['permno'].nunique()),
        'dates': int(signal['date'].nunique()),
        'first_date': str(signal['date'].min().date()),
        'last_date': str(signal['date'].max().date()),
        'news_share': {f'n_{L}': float((signal[f'n_{L}'] > 0).mean()) for L in windows},
        'stories_total_L1': int(signal['n'].sum()),
    }
    with open(path.with_suffix('.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


def load_signal(path: str | Path) -> pd.DataFrame:
    """Read a signal file: .parquet (SPEC section 3) or legacy .h5 (key 'sentiment')."""
    path = Path(path)
    if path.suffix == '.parquet':
        sig = pd.read_parquet(path)
    elif path.suffix in ('.h5', '.hdf', '.hdf5'):
        sig = pd.read_hdf(path, key='sentiment')
    else:
        raise ValueError(f'unknown signal file type: {path}')
    sig['date'] = pd.to_datetime(sig['date'])
    return sig.sort_values(['date', 'permno']).reset_index(drop=True)
