"""Overlay every instance of a combo on a common LTD-relative axis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import store
from .codes import parse_combo


def load_combo(spec, root=None) -> pd.DataFrame:
    if isinstance(spec, str):
        spec = parse_combo(spec)
    return store.read(store.combo_path(spec, root))


def full_window(df, axis='td_to_ltd', min_coverage=0.9, max_ltd_gap_days=7):
    """The widest window that most instances actually reach.

    Instances of one combo are usually the same length, but a re-listing or a
    data gap can leave one much shorter.  Taking the union would rebase that
    short instance far to the right of the others and make the overlay
    incomparable, so the deep end is the offset that `min_coverage` of
    instances still cover.
    """
    d = df
    if 'ltd_gap_days' in d.columns and max_ltd_gap_days is not None:
        d = d[d['ltd_gap_days'] <= max_ltd_gap_days]
    if d.empty:
        raise ValueError('no instances left after the LTD-gap filter')
    starts = d.groupby('instance')[axis].min()
    lo = int(np.ceil(starts.quantile(min_coverage)))
    hi = int(d[axis].max())
    return lo, hi


def overlay(df, window=(-40, -15), axis='td_to_ltd', price='settlement',
            require_full=True, min_coverage=0.9, max_ltd_gap_days=7):
    """Wide matrix of rebased instances: index = offset, columns = year.

    Each instance is rebased to 0 at the first observation inside the window,
    so the value at any offset is the P&L, in price units, of one combo held
    from the start of the window.  Levels are not comparable across years -
    a 2008 crude spread and a 2016 one live on different scales - but P&L
    from a common entry is exactly what the strategy earns.
    """
    lo, hi = window
    if lo > hi:
        raise ValueError(f'window {window} is inverted')

    # An instance whose data stops well before its LTD has offset 0 sitting
    # somewhere in the middle of its life, not at expiry.  Overlaying it would
    # shift every other instance against it.  Live and not-yet-expired
    # contracts are exactly this case, so they are dropped by default.
    if 'ltd_gap_days' in df.columns and max_ltd_gap_days is not None:
        df = df[df['ltd_gap_days'] <= max_ltd_gap_days]
        if df.empty:
            return pd.DataFrame()

    sub = df[(df[axis] >= lo) & (df[axis] <= hi)]
    if sub.empty:
        return pd.DataFrame()

    wide = sub.pivot_table(index=axis, columns='instance', values=price)
    wide = wide.reindex(range(int(lo), int(hi) + 1))

    if require_full:
        span = hi - lo + 1
        keep = wide.notna().sum() >= min_coverage * span
        wide = wide.loc[:, keep]
    if wide.empty:
        return wide

    base = wide.apply(lambda c: c.loc[c.first_valid_index()]
                      if c.first_valid_index() is not None else np.nan)
    return wide - base


def condense(df, window=(-40, -15), axis='td_to_ltd', price='settlement',
             require_full=True, min_coverage=0.9, max_ltd_gap_days=7):
    """The average instance, plus dispersion across instances."""
    wide = overlay(df, window, axis, price, require_full, min_coverage,
                   max_ltd_gap_days)
    if wide.empty:
        return pd.DataFrame(), wide
    out = pd.DataFrame({
        'mean': wide.mean(axis=1),
        'median': wide.median(axis=1),
        'std': wide.std(axis=1, ddof=1),
        'n': wide.notna().sum(axis=1),
    })
    out['se'] = out['std'] / np.sqrt(out['n'].where(out['n'] > 0))
    out['p_up'] = (wide > 0).sum(axis=1) / wide.notna().sum(axis=1)
    out.index.name = axis
    return out, wide


def summary(df, window=(-40, -15), axis='td_to_ltd', price='settlement',
            require_full=True, min_coverage=0.9,
            max_ltd_gap_days=7) -> pd.DataFrame:
    """Per-instance P&L over the window, and the aggregate statistics."""
    wide = overlay(df, window, axis, price, require_full, min_coverage,
                   max_ltd_gap_days)
    if wide.empty:
        return pd.DataFrame()
    pnl = wide.apply(lambda c: c.loc[c.last_valid_index()]
                     if c.last_valid_index() is not None else np.nan)
    per = pd.DataFrame({'instance': pnl.index, 'pnl': pnl.to_numpy()})
    per['win'] = per['pnl'] > 0
    return per.reset_index(drop=True)


def stats(per: pd.DataFrame) -> dict:
    """Aggregate the per-instance P&L into the numbers worth quoting."""
    if per.empty:
        return {}
    x = per['pnl'].dropna()
    n = len(x)
    sd = x.std(ddof=1)
    return {
        'n': n,
        'mean': x.mean(),
        'median': x.median(),
        'std': sd,
        'hit_rate': float((x > 0).mean()),
        'min': x.min(),
        'max': x.max(),
        # Across independent yearly instances, not a time series - this is a
        # one-sample t on n observations, so n is small and the t is fragile.
        't_stat': float(x.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
    }
