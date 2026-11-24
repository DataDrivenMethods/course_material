"""Evaluation and simulation layer for news-sentiment signals (see SPEC.md, Section 5)."""

from .returns import get_returns, wide_returns
from .evaluate import forward_returns, daily_ic, ic_summary, ic_decay, check_timing
from .simulate import quantile_portfolios, performance_table

__all__ = ['get_returns', 'wide_returns', 'forward_returns', 'daily_ic', 'ic_summary',
           'ic_decay', 'check_timing', 'quantile_portfolios', 'performance_table']
