"""Shared library for the news-sentiment signal pipeline (see SPEC.md)."""

from .portfolio import assign_quantiles
from .ravenpack import STORY_COLUMNS, StoryFilter, apply_filter, build_link, fetch_stories
from .signal import (aggregate_daily, build_panel, load_signal, make_signal,
                     restrict_to_membership, save_signal)
from .timing import assign_trading_day
from .wrds_io import (CRSP_LIB, DATA_DIR, connect, find_universe, load_membership, load_universe,
                      member_days, membership_spells, trading_calendar, universe_tag)

__all__ = ['assign_quantiles', 'STORY_COLUMNS', 'StoryFilter', 'apply_filter', 'build_link',
           'fetch_stories', 'aggregate_daily', 'build_panel', 'load_signal', 'make_signal',
           'restrict_to_membership', 'save_signal', 'assign_trading_day', 'CRSP_LIB', 'DATA_DIR',
           'connect', 'find_universe', 'load_membership', 'load_universe', 'member_days', 'membership_spells', 'trading_calendar',
           'universe_tag']
