"""Seasonal futures spread / butterfly backtest harness.

    from futspread import build_combo, load_combo, condense, plot_combo

    build_combo('GC[X][Z1]')                       # once, after download
    df = load_combo('GC[X][Z1]')
    cond, wide = condense(df, window=(-40, -15))
    plot_combo(df, window=(-40, -15))
"""
from .codes import (ComboSpec, Leg, parse_combo, parse_safe_code,
                    resolve_leg_years)
from .universe import UNIVERSE, contrcode_for, universe
from .download import download
from .instances import (build_combo, clear_caches, enumerate_specs,
                        liquid_months, load_contracts, load_prices,
                        month_liquidity, traded_months)
from .condense import condense, load_combo, overlay, stats, summary
from .plotting import plot_combo

__all__ = ['ComboSpec', 'Leg', 'parse_combo', 'parse_safe_code',
           'resolve_leg_years',
           'UNIVERSE', 'universe', 'contrcode_for', 'download', 'build_combo',
           'enumerate_specs', 'load_contracts', 'load_prices',
           'clear_caches', 'traded_months', 'liquid_months', 'month_liquidity', 'load_combo',
           'overlay', 'condense', 'summary', 'stats', 'plot_combo']
