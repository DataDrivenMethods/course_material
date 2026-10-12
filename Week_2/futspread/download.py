"""WRDS downloader: trdstrm.wrds_contract_info + trdstrm.wrds_fut_contract."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from . import store
from .universe import universe

PRICE_COLS = ['open', 'high', 'low', 'settlement', 'volume', 'oi']


def connect(username: str):
    import wrds
    return wrds.Connection(wrds_username=username)


def _contract_year(contrdate: str, lasttrddate) -> float:
    """MMYY -> full contract year.

    The century is ambiguous in MMYY, and the contract year is NOT always the
    last-trading-date year: CL Jan-2025 stops trading 2024-12-19.  Pick the
    century that puts the contract closest to its own last trading date.
    """
    if not isinstance(contrdate, str) or len(contrdate) != 4:
        return np.nan
    if pd.isna(lasttrddate):
        return np.nan
    yy = int(contrdate[2:])
    ref = pd.Timestamp(lasttrddate).year
    return float(min((1900 + yy, 2000 + yy, 2100 + yy),
                     key=lambda c: abs(c - ref)))


# Series whose vendor first-notice date is missing on older rows and whose
# exchange rule is simple enough to reconstruct: CBOT/KCBT grains, FND = last
# business day of the month before the delivery month.  Holidays are ignored,
# so a rule-derived FND can sit one day late; rows carry fnd_is_rule=True.
_FND_RULE = {3247, 3272, 3376, 456, 3256, 3262, 3268, 3623}


def _rule_fnd(contrcode, month, year):
    if contrcode not in _FND_RULE or pd.isna(month) or pd.isna(year):
        return pd.NaT
    first = pd.Timestamp(int(year), int(month), 1)
    return first - pd.offsets.BDay(1)


def fetch_contract_info(conn, contrcodes, min_ltd='2000-01-01') -> pd.DataFrame:
    """Contract metadata for the given series, with derived month/year/LTD.

    Older rows of some series (ZW before 2007, ZL before 2007, HE before 2001)
    have bars but every date field except startdate is null.  The last bar
    date stands in for lasttrddate on those (ltd_source='last_bar'); the
    filter on min_ltd uses the same fallback so they are not dropped.
    """
    codes = ','.join(str(int(c)) for c in contrcodes)
    df = conn.raw_sql(f"""
        SELECT ci.futcode, ci.contrcode, ci.dsmnem, ci.contrname, ci.contrdate,
               ci.exchtickersymb, ci.startdate, ci.lasttrddate,
               ci.firstnoticedate, ci.sttlmntdate, ci.trdstatcode,
               ci.unitdesc, ci.isocurrcode, lb.last_bar
        FROM trdstrm.wrds_contract_info ci
        LEFT JOIN (
            SELECT fc.futcode, MAX(fc.date_) AS last_bar
            FROM trdstrm.wrds_fut_contract fc
            JOIN trdstrm.wrds_contract_info c2 ON c2.futcode = fc.futcode
            WHERE c2.contrcode IN ({codes}) AND c2.lasttrddate IS NULL
            GROUP BY fc.futcode
        ) lb ON lb.futcode = ci.futcode
        WHERE ci.contrcode IN ({codes})
          AND COALESCE(ci.lasttrddate, lb.last_bar) >= '{min_ltd}'
        ORDER BY ci.contrcode, COALESCE(ci.lasttrddate, lb.last_bar)
    """, date_cols=['startdate', 'lasttrddate', 'firstnoticedate',
                    'sttlmntdate', 'last_bar'])

    df['futcode'] = df['futcode'].astype('int64')
    df['contrcode'] = df['contrcode'].astype('int64')
    df['ltd_source'] = np.where(df['lasttrddate'].notna(), 'vendor', 'last_bar')
    df['lasttrddate'] = df['lasttrddate'].fillna(df['last_bar'])
    df = df.drop(columns='last_bar')

    df['contract_month'] = pd.to_numeric(
        df['contrdate'].str.slice(0, 2), errors='coerce')
    df['contract_year'] = [
        _contract_year(cd, ltd)
        for cd, ltd in zip(df['contrdate'], df['lasttrddate'])]

    rule = pd.Series([_rule_fnd(c, m, y) for c, m, y in
                      zip(df['contrcode'], df['contract_month'],
                          df['contract_year'])], index=df.index)
    df['fnd_is_rule'] = df['firstnoticedate'].isna() & rule.notna()
    df['firstnoticedate'] = df['firstnoticedate'].fillna(rule)

    # LTD for a seasonal study is the last day the position can be held
    # without taking delivery: the earlier of last trading day and first
    # notice day.  For metals and grains FND leads LTD by up to a month.
    fnd = df['firstnoticedate']
    df['ltd'] = df['lasttrddate'].where(fnd.isna(), np.minimum(
        df['lasttrddate'], fnd.fillna(pd.Timestamp.max)))
    df['ltd_is_fnd'] = fnd.notna() & (fnd < df['lasttrddate'])

    return df.dropna(subset=['contract_month', 'contract_year', 'ltd'])


def fetch_series_span(conn, contrcodes) -> pd.DataFrame:
    """Full-history span per series, ignoring `min_ltd`.

    `contracts.parquet` is truncated at min_ltd, so it cannot say how far back
    WRDS actually goes.  Null lasttrddate falls back to the last bar, as in
    fetch_contract_info.
    """
    codes = ','.join(str(int(c)) for c in contrcodes)
    df = conn.raw_sql(f"""
        SELECT ci.contrcode,
               MIN(COALESCE(ci.lasttrddate, lb.last_bar)) AS first_ltd,
               MAX(COALESCE(ci.lasttrddate, lb.last_bar)) AS last_ltd,
               COUNT(*) AS n_contracts
        FROM trdstrm.wrds_contract_info ci
        LEFT JOIN (
            SELECT fc.futcode, MAX(fc.date_) AS last_bar
            FROM trdstrm.wrds_fut_contract fc
            JOIN trdstrm.wrds_contract_info c2 ON c2.futcode = fc.futcode
            WHERE c2.contrcode IN ({codes}) AND c2.lasttrddate IS NULL
            GROUP BY fc.futcode
        ) lb ON lb.futcode = ci.futcode
        WHERE ci.contrcode IN ({codes})
        GROUP BY ci.contrcode
    """, date_cols=['first_ltd', 'last_ltd'])
    df['contrcode'] = df['contrcode'].astype('int64')
    df['n_contracts'] = df['n_contracts'].astype('int64')
    return df


def fetch_prices(conn, futcodes) -> pd.DataFrame:
    """Every daily bar for the given contracts (one series' futcodes)."""
    # DISTINCT is load-bearing.  wrds_fut_contract holds several rows per
    # (futcode, date) that are identical in every OHLCV field and differ only
    # in the undocumented column `p` - up to 5 copies for some series (KE, BRN,
    # G, ES, ZN), none for others (GC, NG).  Since `p` is not selected, the
    # projected rows are exact duplicates and DISTINCT collapses them at the
    # server, cutting the transfer by up to 5x.
    df = conn.raw_sql(f"""
        SELECT DISTINCT fc.futcode, fc.date_ AS date, fc.open_ AS open,
               fc.high, fc.low, fc.settlement, fc.volume,
               fc.openinterest AS oi
        FROM trdstrm.wrds_fut_contract fc
        WHERE fc.futcode IN ({','.join(str(int(f)) for f in futcodes)})
        ORDER BY fc.futcode, fc.date_
    """, date_cols=['date'])
    df['futcode'] = df['futcode'].astype('int64')
    # Belt and braces: a residual duplicate would blow up the leg join later.
    return df.drop_duplicates(['futcode', 'date']).reset_index(drop=True)


def download(username: str, root=None, tickers=None, sectors=None,
             min_ltd='2000-01-01', verbose=True) -> pd.DataFrame:
    """Pull metadata + prices for the universe and persist the raw tree."""
    uni = universe(sectors=sectors, tickers=tickers)
    if uni.empty:
        raise ValueError('no series selected')

    conn = connect(username)
    try:
        info = fetch_contract_info(conn, uni['contrcode'], min_ltd)

        # A partial download must not evict series pulled earlier, or their
        # already-built combos become unrebuildable.  Refresh only the
        # contrcodes in this run and keep everything else.
        path = store.contracts_path(root)
        if path.exists():
            prior = store.read(path)
            prior = prior[~prior['contrcode'].isin(info['contrcode'].unique())]
            info = pd.concat([prior, info], ignore_index=True)
            info = info.sort_values(['contrcode', 'lasttrddate'])
            # series pulled before these columns existed
            info['ltd_source'] = info['ltd_source'].fillna('vendor')
            info['fnd_is_rule'] = info['fnd_is_rule'].fillna(False).astype(bool)
        store.write(info, path)
        if verbose:
            print(f'contracts: {len(info):,} rows '
                  f'({info["contrcode"].nunique()} series) -> {path}')

        span = fetch_series_span(conn, uni['contrcode'])
        path = store.series_path(root)
        if path.exists():
            prior = store.read(path)
            prior = prior[~prior['contrcode'].isin(span['contrcode'])]
            span = pd.concat([prior, span], ignore_index=True)
        store.write(span.sort_values('contrcode'), path)

        for row in uni.itertuples():
            fut = info.loc[info['contrcode'] == row.contrcode, 'futcode']
            px = fetch_prices(conn, fut)
            path = store.write(px, store.prices_path(row.contrcode, root))
            if verbose:
                span = (f"{px['date'].min():%Y-%m-%d}..{px['date'].max():%Y-%m-%d}"
                        if len(px) else 'empty')
                print(f'  {row.ticker:<4s} {row.name[:34]:<34s} '
                      f'{len(px):>9,} bars  {span}')
    finally:
        conn.close()
    return info


def fetch_catalog(conn, search=None, min_contracts=1, active_since=None,
                  prefixes=None) -> pd.DataFrame:
    """Every contract SERIES in trdstrm, for discovering what to add.

    This is the whole feed, not the curated universe: ~3,700 series, mostly
    non-US (European single-stock futures dominate).  `search` matches
    contrname or ticker, case-insensitively.
    """
    where = ['1=1']
    if search:
        terms = [search] if isinstance(search, str) else list(search)
        ors = ' OR '.join(
            f"strpos(upper(contrname), '{t.upper()}') > 0 "
            f"OR upper(exchtickersymb) = '{t.upper()}'" for t in terms)
        where.append(f'({ors})')
    if prefixes:
        pl = ','.join(f"'{p}'" for p in prefixes)
        where.append(f'left(dsmnem, 1) IN ({pl})')
    # active_since asks "is this series still listing", which is a property
    # of the series (max lasttrddate), not of each contract.  Filtering rows
    # on it would shrink n_contracts and silently defeat min_contracts.
    having = [f'count(*) >= {int(min_contracts)}']
    if active_since:
        having.append(f"max(lasttrddate) >= '{active_since}'")

    df = conn.raw_sql(f"""
        SELECT contrcode,
               min(exchtickersymb) AS ticker,
               min(contrname)      AS name,
               min(left(dsmnem,1)) AS pfx,
               count(*)            AS n_contracts,
               min(lasttrddate)    AS first_ltd,
               max(lasttrddate)    AS last_ltd,
               sum(CASE WHEN firstnoticedate IS NOT NULL THEN 1 ELSE 0 END)
                                   AS n_fnd
        FROM trdstrm.wrds_contract_info
        WHERE {' AND '.join(where)}
        GROUP BY contrcode
        HAVING {' AND '.join(having)}
        ORDER BY count(*) DESC
    """, date_cols=['first_ltd', 'last_ltd'])
    df['contrcode'] = df['contrcode'].astype('int64')
    return df


def catalog(username, search=None, min_contracts=1, active_since=None,
            prefixes=None) -> pd.DataFrame:
    conn = connect(username)
    try:
        df = fetch_catalog(conn, search, min_contracts, active_since, prefixes)
    finally:
        conn.close()
    from .universe import UNIVERSE
    df['in_universe'] = df['contrcode'].isin(UNIVERSE['contrcode'])
    return df
