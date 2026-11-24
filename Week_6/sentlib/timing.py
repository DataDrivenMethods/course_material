"""Map story timestamps to the trading day on which they are first tradable."""

import numpy as np
import pandas as pd


def assign_trading_day(timestamps_utc: pd.Series, calendar: pd.DatetimeIndex,
                       cutoff: str = '15:30') -> pd.Series:
    """Trading day d such that the story is before cutoff (New York time) on d.

    A story at or after the cutoff moves to the next calendar day, then rolls
    forward to the first trading day on or after it (weekends, holidays).
    Naive timestamps are taken as UTC. Returns NaT where that day is after
    the last calendar date. Same index as the input.
    """
    ts = pd.to_datetime(timestamps_utc)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    ny = ts.dt.tz_convert('America/New_York').dt.tz_localize(None)
    day = ny.dt.normalize()
    cut = pd.Timedelta(cutoff + ':00' if cutoff.count(':') == 1 else cutoff)
    after_cutoff = (ny - day) >= cut
    candidate = day + pd.to_timedelta(after_cutoff.astype(int), unit='D')

    calendar = pd.DatetimeIndex(calendar)
    pos = calendar.searchsorted(candidate.to_numpy(), side='left')
    valid = (pos < len(calendar)) & candidate.notna().to_numpy()
    out = np.full(len(ts), np.datetime64('NaT'), dtype='datetime64[ns]')
    out[valid] = calendar.to_numpy()[pos[valid]]
    return pd.Series(out, index=timestamps_utc.index, name='date')
