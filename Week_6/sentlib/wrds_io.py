"""WRDS connection, universe membership and the CRSP trading calendar."""

import os
import re
from pathlib import Path

import pandas as pd
import wrds

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# CRSP CIZ stock library: 'crsp_q_stock' (quarterly update, the default) or 'crsp_a_stock'
# (annual). Set the CRSP_LIB environment variable to switch. The legacy tables
# (crsp.dsi, crsp.stocknames, crsp.dsedist) stop at 2024-12-31 in both, so only CIZ tables are used.
CRSP_LIB = os.environ.get('CRSP_LIB', 'crsp_q_stock')


def connect(username: str | None = None) -> wrds.Connection:
    """WRDS connection (credentials from ~/.pgpass)."""
    return wrds.Connection(wrds_username=username or os.environ.get('WRDS_USERNAME', 'cpehlivanian'))


UPDATE_PERIODS = ('year', 'month', 'day')


def universe_tag(top_n: int, start: str, end: str, update_period: str = 'year') -> str:
    """File tag used by Big_10_Index.py: 'top1000_2020-01-01_2024-12-31' for yearly
    lists, 'top1000m_...' for monthly and 'top1000d_...' for daily."""
    if update_period not in UPDATE_PERIODS:
        raise ValueError(f'update_period must be one of {UPDATE_PERIODS}')
    start = pd.Timestamp(start).strftime('%Y-%m-%d')
    end = pd.Timestamp(end).strftime('%Y-%m-%d')
    suffix = '' if update_period == 'year' else update_period[0]
    return f'top{top_n}{suffix}_{start}_{end}'


def load_membership(tag: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """PIT membership written by Big_10_Index.py, without the final (NaT) list."""
    path = Path(data_dir) / f'{tag}_membership.h5'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found: run Big_10_Index.py for this universe first')
    membership = pd.read_hdf(path, key='membership')
    # The last year-end list has no holding window yet
    return membership.dropna(subset=['effective_from']).reset_index(drop=True)


def membership_spells(membership: pd.DataFrame) -> pd.DataFrame:
    """Merge back-to-back holding windows of a share class into one spell.

    Monthly or daily lists give one row per stock per period; a stock that stays
    in the list keeps one row as long as its permco and ticker do not change.
    Returns [permno, permco, ticker, effective_from, effective_to].
    """
    key = ['permno', 'permco', 'ticker']
    m = (membership.dropna(subset=['effective_from'])
                   .sort_values(['permno', 'effective_from']).reset_index(drop=True))
    prev = m.groupby('permno')[key + ['effective_to']].shift()
    cont = ((m[key] == prev[key]).all(axis=1)
            & (m['effective_from'] == prev['effective_to'] + pd.Timedelta(days=1)))
    spells = (m.assign(spell=(~cont).cumsum())
               .groupby('spell').agg(permno=('permno', 'first'), permco=('permco', 'first'),
                                     ticker=('ticker', 'first'),
                                     effective_from=('effective_from', 'min'),
                                     effective_to=('effective_to', 'max')))
    return spells.reset_index(drop=True)


def member_days(membership: pd.DataFrame, calendar: pd.DatetimeIndex,
                start: str, end: str) -> pd.DataFrame:
    """One row per member stock-day: [date, permno, permco, ticker], START <= date <= END."""
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    rows = []
    for m in membership_spells(membership).itertuples(index=False):
        d = days[(days >= m.effective_from) & (days <= m.effective_to)]
        rows.append(pd.DataFrame({'date': d, 'permno': m.permno, 'permco': m.permco, 'ticker': m.ticker}))
    out = pd.concat(rows, ignore_index=True)
    out = out.astype({'permno': 'int64', 'permco': 'int64'})
    return out.sort_values(['date', 'permno']).reset_index(drop=True)


def find_universe(top_n: int, start: str, end: str, update_period: str = 'year',
                  data_dir: Path = DATA_DIR) -> str:
    """Tag of a Big_10_Index.py build whose membership covers start..end.

    The exact build (same dates) is used if it exists, otherwise the narrowest build
    whose range contains start..end. Each list depends only on its selection date, so a
    window of a longer build is the same universe. Coverage is checked on the file's
    contents, not only its name (a build can end early if CRSP data stopped)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    exact = universe_tag(top_n, start, end, update_period)
    prefix = exact.split('_')[0]                       # 'top1000', 'top1000m' or 'top1000d'
    pattern = re.compile(rf'{prefix}_(\d{{4}}-\d\d-\d\d)_(\d{{4}}-\d\d-\d\d)_membership\.h5')
    candidates = []
    for path in Path(data_dir).glob(f'{prefix}_*_membership.h5'):
        m = pattern.fullmatch(path.name)
        if m and pd.Timestamp(m[1]) <= start and pd.Timestamp(m[2]) >= end:
            tag = path.name.removesuffix('_membership.h5')
            candidates.append((tag != exact, pd.Timestamp(m[2]) - pd.Timestamp(m[1]), tag))
    for _, _, tag in sorted(candidates):
        m = load_membership(tag, data_dir)
        first, last = m['effective_from'].min(), m['effective_to'].max()
        # Lists are held through the last trading day on or before end (up to 4 days earlier)
        if first <= start and last >= end - pd.Timedelta(days=4):
            return tag
        print(f'Skipping {tag}: its lists only cover {first:%Y-%m-%d}..{last:%Y-%m-%d}')
    raise FileNotFoundError(f'no {prefix}_*_membership.h5 in {data_dir} covers '
                            f'{start:%Y-%m-%d}..{end:%Y-%m-%d}: run Big_10_Index.py for a range that does')


def load_universe(top_n: int, start: str, end: str, update_period: str = 'year',
                  data_dir: Path = DATA_DIR) -> tuple[str, pd.DataFrame]:
    """(tag, membership) for start..end: the build from find_universe, keeping only
    the lists held during start..end, which is what a build of exactly start..end holds."""
    tag = find_universe(top_n, start, end, update_period, data_dir)
    m = load_membership(tag, data_dir)
    m = m[(m['effective_to'] >= pd.Timestamp(start)) & (m['effective_from'] <= pd.Timestamp(end))]
    return tag, m.reset_index(drop=True)


def trading_calendar(conn, start: str, end: str) -> pd.DatetimeIndex:
    """CRSP trading days (CIZ daily index file) between start and end, inclusive."""
    cal = conn.raw_sql(f"""
        SELECT dlycaldt FROM {CRSP_LIB}.wrds_dailyindexret_query
        WHERE dlycaldt BETWEEN '{start}' AND '{end}'
        ORDER BY dlycaldt
    """)
    return pd.DatetimeIndex(pd.to_datetime(cal['dlycaldt']), name='date')
