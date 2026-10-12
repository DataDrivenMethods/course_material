"""Overlay and condensed plots for one aggregate combo."""
from __future__ import annotations

import numpy as np

from .condense import condense, stats, summary


def plot_combo(df, window=(-40, -15), axis='td_to_ltd', price='settlement',
               separate=True, require_full=True, min_coverage=0.9,
               title=None, outfile=None, figsize=(12, 8)):
    """All instances overlaid, plus the condensed average.

    separate=True puts the average in its own panel underneath; False draws
    it on top of the overlay in bold.
    """
    import matplotlib.pyplot as plt

    cond, wide = condense(df, window, axis, price, require_full, min_coverage)
    if wide.empty:
        raise ValueError('no instances survived the window / coverage filter')

    combo = df['combo'].iloc[0] if 'combo' in df else ''
    per = summary(df, window, axis, price, require_full, min_coverage)
    st = stats(per)

    nrows = 2 if separate else 1
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True,
                             squeeze=False)
    ax = axes[0][0]

    cmap = plt.get_cmap('viridis')
    years = list(wide.columns)
    for k, yr in enumerate(years):
        ax.plot(wide.index, wide[yr], lw=1.0, alpha=0.75,
                color=cmap(k / max(len(years) - 1, 1)), label=str(yr))
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_ylabel(f'{price} P&L from entry')
    ax.grid(alpha=0.25)
    if len(years) <= 18:
        ax.legend(ncol=max(1, len(years) // 9), fontsize=7, frameon=False)

    head = title or f'{combo}   window [{window[0]}, {window[1]}] {axis}'
    if st:
        head += (f'\nn={st["n"]}  mean={st["mean"]:.4g}  '
                 f'median={st["median"]:.4g}  hit={st["hit_rate"]:.0%}  '
                 f't={st["t_stat"]:.2f}')
    ax.set_title(head, fontsize=10)

    ax2 = axes[1][0] if separate else ax
    ax2.plot(cond.index, cond['mean'], color='crimson', lw=2.2, label='mean')
    ax2.plot(cond.index, cond['median'], color='navy', lw=1.4, ls='--',
             label='median')
    ax2.fill_between(cond.index, cond['mean'] - cond['se'],
                     cond['mean'] + cond['se'], color='crimson', alpha=0.18,
                     label='+/- 1 s.e.')
    ax2.axhline(0, color='0.4', lw=0.8)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_ylabel('average P&L')
    ax2.set_xlabel(f'{axis}  (0 = LTD = min(last trade, first notice))')

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=140, bbox_inches='tight')
    return fig, (cond, wide, per, st)
