"""CRSP daily total returns for the backtest (CIZ / v2 daily stock file).

Pulled from <CRSP_LIB>.wrds_dsfv2_query (sentlib.wrds_io.CRSP_LIB, default crsp_q_stock).

Delisting returns in crsp_a_stock.wrds_dsfv2_query (verified 2026-10 on the
194 members of top1000_2020-01-01_2024-12-31 that delisted before 2025-12-31):
  * The delisting return is NOT folded into dlyret of the last trading day.
    CIZ adds one extra row dated the next trading day (dlydelflg = 'Y',
    dlyprcflg 'DA'/'DP', dlyretdurflg 'DD'/'P2'/'MR') whose dlyret equals the
    legacy crsp.dsedelist dlret. Examples: CERN 10909 last trade 2022-06-07,
    'Y' row 2022-06-08 dlyret 0.000843 (= dsedelist dlret); FRC 12448 last
    trade 2023-04-28, 'Y' row 2023-05-01 dlyret -0.9048.
  * Two cases have a missing delisting return (dlyretmissflg 'DG'): SIVB 11786
    and SBNY 90090, both 2023-03-13. dlyprc on that row is the first OTC price
    (0.40 vs 39.37 for SIVB), so we fill dlyret = dlyprc / dlyprevprc - 1
    (about -99%); otherwise a failed bank would silently earn 0.
  * We keep the 'Y' row as an ordinary return on its date (moved to the next
    trading day if it is not one), so a position held into a delisting earns it.
After a stock's last row its return is 0 (proceeds held as cash) for
compounding; gaps before that stay NaN. The table holds one row per
(permno, dlycaldt) except for a handful of duplicates from the distribution
join, which we drop.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sentlib.wrds_io import CRSP_LIB

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PAD_DAYS = 45      # calendar days pulled past the signal end, for forward returns


def default_cache_path(permnos, start: str, end: str, data_dir: Path = DATA_DIR) -> Path:
    h = hashlib.sha1(','.join(map(str, sorted(set(permnos)))).encode()).hexdigest()[:8]
    lib = '' if CRSP_LIB == 'crsp_a_stock' else f'_{CRSP_LIB}'    # annual caches keep their old names
    return data_dir / f'crsp_returns_{start}_{end}_{h}{lib}.parquet'


def get_returns(conn, permnos: list[int], start: str, end: str,
                cache_path: Path | None = None, pad_days: int = PAD_DAYS,
                refresh: bool = False) -> pd.DataFrame:
    """[date, permno, ret, prevcap, delist] for start..end + pad_days, cached to parquet.

    `end` is the signal end; the pull runs pad_days calendar days past it.
    delist = True on the CIZ delisting row (see module docstring).
    """
    start = pd.Timestamp(start).strftime('%Y-%m-%d')
    end_pull = (pd.Timestamp(end) + pd.Timedelta(days=pad_days)).strftime('%Y-%m-%d')
    permnos = sorted({int(p) for p in permnos})
    if cache_path is None:
        cache_path = default_cache_path(permnos, start, end_pull)
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    plist = ','.join(map(str, permnos))
    raw = conn.raw_sql(f"""
        SELECT permno, dlycaldt, dlydelflg, dlyret, dlyprc, dlyprevprc, dlyprevcap
        FROM {CRSP_LIB}.wrds_dsfv2_query
        WHERE permno IN ({plist}) AND dlycaldt BETWEEN '{start}' AND '{end_pull}'
    """)
    raw['dlycaldt'] = pd.to_datetime(raw['dlycaldt'])
    raw = raw.drop_duplicates(['permno', 'dlycaldt'])
    for c in ['dlyret', 'dlyprc', 'dlyprevprc', 'dlyprevcap']:
        raw[c] = raw[c].astype('float64')
    raw['delist'] = raw['dlydelflg'].eq('Y').fillna(False).astype(bool)

    # Missing delisting return with a post-delisting price: use the price change
    fix = raw['delist'] & raw['dlyret'].isna() & (raw['dlyprc'] > 0) & (raw['dlyprevprc'] > 0)
    raw.loc[fix, 'dlyret'] = raw.loc[fix, 'dlyprc'] / raw.loc[fix, 'dlyprevprc'] - 1

    # Trading calendar = dates with regular rows; delisting rows go to the next trading day
    cal = pd.DatetimeIndex(np.sort(raw.loc[~raw['delist'], 'dlycaldt'].unique()))
    pos = cal.searchsorted(raw['dlycaldt'])
    ok = pos < len(cal)
    raw = raw[ok].copy()
    raw['date'] = cal[pos[ok]]

    out = (raw.rename(columns={'dlyret': 'ret', 'dlyprevcap': 'prevcap'})
              [['date', 'permno', 'ret', 'prevcap', 'delist']]
              .sort_values(['delist', 'date', 'permno'])
              .drop_duplicates(['date', 'permno'])           # regular row wins a collision
              .sort_values(['date', 'permno']).reset_index(drop=True))
    out['permno'] = out['permno'].astype('int64')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f'Saved {len(out):,} return rows ({out.permno.nunique()} permnos, '
          f'{out.date.min():%Y-%m-%d}..{out.date.max():%Y-%m-%d}) to {cache_path}')
    return out


def wide_returns(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot to date x permno: (R, prevcap, alive).

    R: daily return, 0 after the stock's last row (cash), NaN in earlier gaps.
    alive: True from a stock's first to last regular (non-delisting) trading day,
    i.e. days on which it can be bought at the close.
    """
    R = returns.pivot(index='date', columns='permno', values='ret').sort_index()
    cap = returns.pivot(index='date', columns='permno', values='prevcap').reindex_like(R)
    reg = returns[~returns['delist']]
    pos = pd.Series(np.arange(len(R)), index=R.index)
    first = reg.groupby('permno')['date'].min().map(pos).reindex(R.columns)
    last = reg.groupby('permno')['date'].max().map(pos).reindex(R.columns)
    last_any = returns.groupby('permno')['date'].max().map(pos).reindex(R.columns)
    t = np.arange(len(R))[:, None]
    alive = pd.DataFrame((t >= first.to_numpy()[None, :]) & (t <= last.to_numpy()[None, :]),
                         index=R.index, columns=R.columns)
    after = t > last_any.to_numpy()[None, :]
    R = R.mask(after, 0.0)
    return R, cap, alive
