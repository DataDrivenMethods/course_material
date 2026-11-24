"""Turn raw per-contract bars into stacked combo instances.

One parquet per aggregate combo (e.g. GC[X][Z1]), holding every yearly
instance stacked, aligned on a trading-day offset to LTD.
"""
from __future__ import annotations

import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import store
from .codes import (MAX_YEAR_OFFSET, NUM_TO_MONTH_CODE, ComboSpec, Leg,
                    parse_combo, resolve_leg_years)
from .universe import contrcode_for

LEG_FIELDS = ['open', 'high', 'low', 'settlement', 'volume', 'oi']


# Building every combo for a ticker means hundreds of calls against the same
# parquet.  Re-reading and re-grouping NG's 780k bars 432 times dominates the
# run, so keep the last two series in memory - the CLI walks ticker by ticker,
# so two is enough to avoid any thrash.
_CACHE_SIZE = 2
_PX_CACHE = OrderedDict()
_INFO_CACHE = OrderedDict()


def _cached(cache, key, build):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    val = build()
    cache[key] = val
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return val


def clear_caches():
    _PX_CACHE.clear()
    _INFO_CACHE.clear()


def load_prices(contrcode: int, root=None) -> dict:
    """futcode -> bars, for one series."""
    def build():
        px = store.read(store.prices_path(contrcode, root))
        return {int(k): v for k, v in px.groupby('futcode')}
    return _cached(_PX_CACHE, (int(contrcode), str(root)), build)


def load_contracts(ticker: str, root=None) -> pd.DataFrame:
    return _cached(_INFO_CACHE, (ticker.upper(), str(root)),
                   lambda: _load_contracts(ticker, root))


def _load_contracts(ticker: str, root=None) -> pd.DataFrame:
    info = store.read(store.contracts_path(root))
    info = info[info['contrcode'] == contrcode_for(ticker)].copy()
    info['contract_year'] = info['contract_year'].astype(int)
    info['contract_month'] = info['contract_month'].astype(int)
    # A (year, month) can appear more than once after a re-listing; keep the
    # contract that actually traded longest.
    info = (info.sort_values(['contract_year', 'contract_month', 'startdate'])
                .drop_duplicates(['contract_year', 'contract_month'],
                                 keep='first'))
    return info.set_index(['contract_year', 'contract_month'], drop=False)


def traded_months(info: pd.DataFrame, since_year=2010, min_count=4) -> list:
    """Month codes the series actually lists, from observed contracts."""
    recent = info[info['contract_year'] >= since_year]
    counts = recent['contract_month'].value_counts()
    return [NUM_TO_MONTH_CODE[m] for m in sorted(counts[counts >= min_count].index)]


def month_liquidity(ticker: str, root=None, since_year=2015,
                    metric='oi') -> pd.DataFrame:
    """Per contract-month liquidity, from actual bars.

    For each contract take its peak open interest (or volume), then the median
    of that across contracts of the same month.  `share` normalises against
    the busiest month.  Existence and liquidity are very different things: gold
    lists all twelve months, but the serial months sit around 0.007 of the
    Dec contract while the six real ones sit above 0.12.
    """
    info = load_contracts(ticker, root)
    px = load_prices(contrcode_for(ticker), root)
    info = info[info['contract_year'] >= since_year]

    rows = []
    for r in info.itertuples():
        d = px.get(int(r.futcode))
        if d is None or d.empty or metric not in d:
            continue
        col = d[metric].to_numpy(dtype=float)
        if not np.isfinite(col).any():
            continue
        rows.append((int(r.contract_month), float(np.nanmax(col)), len(d)))
    if not rows:
        return pd.DataFrame(columns=['n', 'med_peak', 'med_days', 'share',
                                     'code'])

    f = pd.DataFrame(rows, columns=['month', 'peak', 'ndays'])
    g = f.groupby('month').agg(n=('month', 'size'),
                               med_peak=('peak', 'median'),
                               med_days=('ndays', 'median'))
    g['share'] = g['med_peak'] / g['med_peak'].max()
    g['code'] = [NUM_TO_MONTH_CODE[m] for m in g.index]
    return g


def liquid_months(ticker: str, root=None, min_share=0.05, since_year=2015,
                  metric='oi') -> list:
    """Month codes that actually trade, not merely exist.

    0.05 separates cleanly in every series checked: gold's weakest real month
    (V, October) sits at 0.126 while its busiest serial reaches only 0.009;
    silver splits 0.89+ against 0.015; crude, where all twelve months are
    genuine, floors at 0.67 and nothing is excluded.
    """
    g = month_liquidity(ticker, root, since_year, metric)
    if g.empty:
        return traded_months(load_contracts(ticker, root))
    keep = sorted(g[g['share'] >= min_share].index)
    return [NUM_TO_MONTH_CODE[m] for m in keep]


def liquidity_table(tickers=None, root=None, min_share=0.05, since_year=2015,
                    metric='oi', refresh=False) -> pd.DataFrame:
    """Liquid / illiquid month split for many series, cached to disk.

    Scanning every series means loading every price parquet, so the result is
    cached at raw/liquidity.parquet.  Pass refresh=True after a re-download.
    """
    from .universe import universe
    path = store.liquidity_path(root)
    cached = pd.DataFrame()
    if path.exists() and not refresh:
        cached = store.read(path)

    want = [t.upper() for t in (tickers if tickers is not None
                                else universe()['ticker'])]
    have = set(cached['ticker']) if len(cached) else set()
    todo = [t for t in want if t not in have]

    rows = []
    for tk in todo:
        try:
            g = month_liquidity(tk, root, since_year, metric)
        except Exception:
            continue
        if g.empty:
            continue
        liq = [NUM_TO_MONTH_CODE[m] for m in sorted(g[g['share'] >= min_share].index)]
        ill = [NUM_TO_MONTH_CODE[m] for m in sorted(g[g['share'] < min_share].index)]
        rows.append({'ticker': tk, 'liquid': ' '.join(liq),
                     'illiquid': ' '.join(ill), 'n_liquid': len(liq),
                     'n_illiquid': len(ill),
                     'weakest_liquid_share': float(
                         g[g['share'] >= min_share]['share'].min()) if liq else float('nan')})
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        store.write(cached, path)
    if cached.empty:
        return cached
    return cached[cached['ticker'].isin(want)].reset_index(drop=True)


def enumerate_specs(ticker: str, months, kind='spread',
                    max_offset=MAX_YEAR_OFFSET, consecutive_only=False):
    """Every combo of the given kind over `months`.

    Equal months are kept: GC[Z][Z0] is the Dec/Dec 12-month spread, since
    offset 0 resolves to the next occurrence after the front leg.
    """
    offs = range(max_offset + 1)
    out = []
    if kind == 'spread':
        for m1, m2 in itertools.product(months, months):
            for o in offs:
                out.append(ComboSpec(ticker, (Leg(m1, 0), Leg(m2, o))))
    elif kind == 'butterfly':
        for m1, m2, m3 in itertools.product(months, months, months):
            if consecutive_only:
                i = months.index
                if not (i(m2) - i(m1)) % len(months) == 1:
                    continue
                if not (i(m3) - i(m2)) % len(months) == 1:
                    continue
            for o2, o3 in itertools.product(offs, offs):
                out.append(ComboSpec(ticker,
                                     (Leg(m1, 0), Leg(m2, o2), Leg(m3, o3))))
    else:
        raise ValueError(f'kind must be spread or butterfly, got {kind!r}')
    return out


def instance_legs(info: pd.DataFrame, spec: ComboSpec, front_year: int):
    """Metadata rows for each leg, or None if any leg does not exist."""
    years = resolve_leg_years(front_year, spec.legs)
    rows = []
    for leg, yr in zip(spec.legs, years):
        key = (yr, leg.month_num)
        if key not in info.index:
            return None
        rows.append(info.loc[key])
    return rows


def build_instance(spec: ComboSpec, legs, px_by_futcode, lookback_years=2):
    """One yearly instance as a daily series, or None if unusable.

    The combo dies with its front leg, so LTD is the front leg's
    min(last trading day, first notice day), and history starts
    `lookback_years` before that.
    """
    ltd = pd.Timestamp(legs[0]['ltd'])
    start = ltd - pd.DateOffset(years=lookback_years)

    frames = []
    for i, leg in enumerate(legs, start=1):
        px = px_by_futcode.get(int(leg['futcode']))
        if px is None or px.empty:
            return None
        px = px[(px['date'] >= start) & (px['date'] <= ltd)]
        px = px.dropna(subset=['settlement'])
        if px.empty:
            return None
        px = px.set_index('date')[LEG_FIELDS].add_prefix(f'l{i}_')
        frames.append(px)

    df = pd.concat(frames, axis=1, join='inner').sort_index()
    if df.empty:
        return None

    # The combo price.  Only settlement is differenced - a spread has no
    # meaningful open/high/low, so the raw leg fields are carried through
    # untouched rather than combined.
    sett = np.zeros(len(df))
    for i, w in enumerate(spec.weights, start=1):
        sett += w * df[f'l{i}_settlement'].to_numpy()
    df.insert(0, 'settlement', sett)

    # Liquidity of the combo is bounded by its thinnest leg.
    n = len(spec.legs)
    df['volume'] = df[[f'l{i}_volume' for i in range(1, n + 1)]].min(axis=1)
    df['oi'] = df[[f'l{i}_oi' for i in range(1, n + 1)]].min(axis=1)

    df = df.reset_index().rename(columns={'index': 'date'})

    # Alignment: trading-day offset, 0 on the last observed bar at or before
    # LTD.  cal_to_ltd is the calendar-day equivalent, kept so a window can be
    # specified either way.
    df['td_to_ltd'] = np.arange(len(df)) - (len(df) - 1)
    df['cal_to_ltd'] = -(ltd - df['date']).dt.days

    df.insert(0, 'instance', int(legs[0]['contract_year']))
    df.insert(1, 'combo', spec.code)
    df.insert(2, 'ticker', spec.ticker)
    df['ltd'] = ltd
    df['ltd_is_fnd'] = bool(legs[0]['ltd_is_fnd'])
    df['fnd_is_rule'] = bool(legs[0].get('fnd_is_rule', False))
    df['ltd_source'] = str(legs[0].get('ltd_source', 'vendor'))
    # >0 means the data stops before LTD, so offset 0 is not really LTD.
    df['ltd_gap_days'] = int((ltd - df['date'].iloc[-1]).days)
    for i, leg in enumerate(legs, start=1):
        df[f'l{i}_futcode'] = int(leg['futcode'])
        df[f'l{i}_year'] = int(leg['contract_year'])
        df[f'l{i}_month'] = int(leg['contract_month'])
    return df


def build_combo(spec, root=None, lookback_years=2, min_year=None,
                max_year=None, min_obs=20, persist=True):
    """Build and optionally persist every instance of one aggregate combo."""
    if isinstance(spec, str):
        spec = parse_combo(spec)
    info = load_contracts(spec.ticker, root)
    px_by_futcode = load_prices(contrcode_for(spec.ticker), root)

    years = sorted(info['contract_year'].unique())
    if min_year is not None:
        years = [y for y in years if y >= min_year]
    if max_year is not None:
        years = [y for y in years if y <= max_year]

    out = []
    for yr in years:
        legs = instance_legs(info, spec, int(yr))
        if legs is None:
            continue
        inst = build_instance(spec, legs, px_by_futcode, lookback_years)
        if inst is None or len(inst) < min_obs:
            continue
        out.append(inst)

    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    if persist:
        store.write(df, store.combo_path(spec, root))
    return df
