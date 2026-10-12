"""Curated US futures universe.

`contrcode` is the key, NOT `exchtickersymb`.  The Refinitiv/Datastream feed
behind trdstrm carries ~3,200 ticker symbols against ~3,700 contract series,
so tickers collide, and several US series exist twice - once as a pit/electronic
"Composite" and once as an "Electronic"-only series with a shorter history.
The composite is preferred here wherever both exist; every contrcode below was
checked against trdstrm.wrds_contract_info for contract count, date span and
first-notice-date coverage.
"""
from __future__ import annotations

import pandas as pd

# contrcode, ticker, name, sector, exchange, lot_size, unit
# lot_size is the contract size in `unit`; trdstrm carries only the unit
# name (unitdesc), not the size.  Index: USD per index point.  Rates: face.
_ROWS = [
    # ---- energy (NYMEX) -------------------------------------------------
    (1986, 'CL',  'Crude Oil (Light Sweet)',    'energy',    'NYMEX',       1000, 'bbl'),
    (2060, 'NG',  'Natural Gas (Henry Hub)',    'energy',    'NYMEX',      10000, 'MMBtu'),
    (2029, 'HO',  'Heating Oil / ULSD',         'energy',    'NYMEX',      42000, 'gal'),
    (2091, 'RB',  'Gasoline RBOB',              'energy',    'NYMEX',      42000, 'gal'),
    # ---- metals (COMEX) -------------------------------------------------
    (2020, 'GC',  'Gold (100 oz)',              'metals',    'COMEX',        100, 'oz'),
    (2108, 'SI',  'Silver (5000 oz)',           'metals',    'COMEX',       5000, 'oz'),
    (2026, 'HG',  'Copper (High Grade)',        'metals',    'COMEX',      25000, 'lb'),
    (2074, 'PL',  'Platinum',                   'metals',    'NYMEX',         50, 'oz'),
    (2065, 'PA',  'Palladium',                  'metals',    'NYMEX',        100, 'oz'),
    # ---- grains & oilseeds (CBOT) ---------------------------------------
    (3247, 'ZC',  'Corn',                       'grains',    'CBOT',        5000, 'bu'),
    (3272, 'ZW',  'Wheat (SRW, composite)',     'grains',    'CBOT',        5000, 'bu'),
    (3376, 'ZS',  'Soybeans (composite)',       'grains',    'CBOT',        5000, 'bu'),
    ( 456, 'ZL',  'Soybean Oil',                'grains',    'CBOT',       60000, 'lb'),
    (3256, 'ZM',  'Soybean Meal',               'grains',    'CBOT',         100, 'short ton'),
    (3262, 'ZO',  'Oats (composite)',           'grains',    'CBOT',        5000, 'bu'),
    (3268, 'ZR',  'Rough Rice (composite)',     'grains',    'CBOT',        2000, 'cwt'),
    (3623, 'KE',  'HRW Wheat (Kansas City)',    'grains',    'KCBT',        5000, 'bu'),
    # ---- softs (ICE US, carried under the NYMEX prefix in this feed) -----
    (2104, 'SB',  "Sugar #11",                  'softs',     'ICE-US',    112000, 'lb'),
    (2038, 'KC',  "Coffee 'C'",                 'softs',     'ICE-US',     37500, 'lb'),
    (1980, 'CC',  'Cocoa',                      'softs',     'ICE-US',        10, 'metric ton'),
    (1992, 'CT',  'Cotton #2',                  'softs',     'ICE-US',     50000, 'lb'),
    (2036, 'OJ',  'Orange Juice (FCOJ-A)',      'softs',     'ICE-US',     15000, 'lb'),
    # ---- livestock (CME) ------------------------------------------------
    (2675, 'LE',  'Live Cattle (composite)',    'livestock', 'CME',        40000, 'lb'),
    (2676, 'HE',  'Lean Hogs (composite)',      'livestock', 'CME',        40000, 'lb'),
    (3250, 'GF',  'Feeder Cattle (composite)',  'livestock', 'CME',        50000, 'lb'),
    # ---- FX (CME) -------------------------------------------------------
    (2683, '6E',  'Euro FX',                    'fx',        'CME',       125000, 'EUR'),
    (2846, '6J',  'Japanese Yen',               'fx',        'CME',     12500000, 'JPY'),
    (2776, '6C',  'Canadian Dollar',            'fx',        'CME',       100000, 'CAD'),
    (2807, '6S',  'Swiss Franc',                'fx',        'CME',       125000, 'CHF'),
    # ---- ICE Europe energy ----------------------------------------------
    (1575, 'BRN', 'Brent Crude (ICE)',          'energy',    'ICE-EU',      1000, 'bbl'),
    (1576, 'G',   'Gas Oil (ICE, low sulphur)', 'energy',    'ICE-EU',       100, 'metric ton'),
    # ---- refining cracks, listed as instruments in their own right ------
    (4349, 'ARE', 'RBOB Crack Spread (NYMEX)',  'cracks',    'NYMEX',       1000, 'bbl'),
    (3963, 'RBB', 'RBOB vs Brent Crack',        'cracks',    'NYMEX',       1000, 'bbl'),
    (3943, 'HOB', 'Heating Oil vs Brent Crack', 'cracks',    'NYMEX',       1000, 'bbl'),
    (3886, 'GZ',  'Gas Oil Crack Spread',       'cracks',    'NYMEX',       1000, 'bbl'),
    # ---- equity index (CME) ---------------------------------------------
    (1381, 'ES',  'E-mini S&P 500',             'equity',    'CME',           50, 'USD/pt'),
    ( 323, 'NQ',  'E-mini Nasdaq 100',          'equity',    'CME',           20, 'USD/pt'),
    (4396, 'RTY', 'E-mini Russell 2000',        'equity',    'CME',           50, 'USD/pt'),
    # ---- US treasuries (CBOT) -------------------------------------------
    (3377, 'ZT',  '2Y T-Note',                  'rates',     'CBOT',      200000, 'USD face'),
    (2680, 'ZF',  '5Y T-Note (composite)',      'rates',     'CBOT',      100000, 'USD face'),
    (2682, 'ZN',  '10Y T-Note (composite)',     'rates',     'CBOT',      100000, 'USD face'),
    (3271, 'ZB',  '30Y T-Bond',                 'rates',     'CBOT',      100000, 'USD face'),
    (2685, 'UB',  'Ultra T-Bond (composite)',   'rates',     'CBOT',      100000, 'USD face'),
    # ---- dairy / lumber (CME) -------------------------------------------
    (3255, 'DC',  'Milk Class III (composite)', 'dairy',     'CME',       200000, 'lb'),
    ( 361, 'LBS', 'Lumber, random length',      'lumber',    'CME',       110000, 'board ft'),
    (4755, 'LBR', 'Lumber (2022 respec)',       'lumber',    'CME',        27500, 'board ft'),
]

UNIVERSE = pd.DataFrame(
    _ROWS, columns=['contrcode', 'ticker', 'name', 'sector', 'exchange',
                   'lot_size', 'unit'])


def universe(sectors=None, tickers=None) -> pd.DataFrame:
    """The curated universe, optionally filtered by sector or ticker."""
    u = UNIVERSE
    if sectors:
        u = u[u['sector'].isin(list(sectors))]
    if tickers:
        u = u[u['ticker'].isin([t.upper() for t in tickers])]
    return u.reset_index(drop=True)


def contrcode_for(ticker: str) -> int:
    hit = UNIVERSE[UNIVERSE['ticker'] == ticker.upper()]
    if hit.empty:
        raise KeyError(f'{ticker!r} is not in the curated universe; '
                       'add its contrcode to futspread/universe.py')
    return int(hit['contrcode'].iloc[0])
