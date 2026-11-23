#!/usr/bin/env python
# coding: utf-8

# Backtest of news-sentiment signals: IC by horizon, IC decay, quantile portfolios.
#
# Usage:
#     python run_backtest.py --signals data/top1000_2020-01-01_2024-12-31_rp_noprice_signal.parquet
#     python run_backtest.py --signals data/*_signal.parquet --col sent_5 --holding 5 --all-stocks
#     python run_backtest.py --signals ... --show          # also open the plots in a window
#     python run_backtest.py --signals ... --start 2023-01-01 --end 2024-12-31   # test period only
#
# Timing: the signal on day d is traded at the close of d and earns returns from
# d+1 on (SPEC Section 1). Synthetic-signal checks run first and stop the script
# if the pairing is off by a day.
# Outputs go to data/backtest/{signal name}/: CSV tables and PNG plots, with file
# names ending in {col}_{news|all}_q{quantiles}_h{holding}_{weight}.

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wrds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest.evaluate import n_col_for
from backtest import (check_timing, daily_ic, forward_returns, get_returns, ic_decay,
                      ic_summary, performance_table, quantile_portfolios)

DATA_DIR = HERE / 'data'
OUT_DIR = DATA_DIR / 'backtest'
TAG_RE = re.compile(r'top\d+[md]?_\d{4}-\d\d-\d\d_\d{4}-\d\d-\d\d')

parser = argparse.ArgumentParser(description='IC and quantile-portfolio backtest of sentiment signals.')
parser.add_argument('--signals', nargs='+', required=True, help='Signal files (.parquet, or legacy .h5)')
parser.add_argument('--col', default='sent_1', help='Signal column (default sent_1)')
parser.add_argument('--quantiles', type=int, default=5)
parser.add_argument('--holding', type=int, default=1, help='Holding period in days (overlapping portfolios)')
parser.add_argument('--weight', choices=['equal', 'cap'], default='equal')
parser.add_argument('--cost-bps', type=float, nargs='+', default=[0, 5], help='Costs per dollar traded')
parser.add_argument('--all-stocks', action='store_true', help='Include stock-days without news (news_only off)')
parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20])
parser.add_argument('--method', choices=['spearman', 'pearson'], default='spearman')
parser.add_argument('--max-lag', type=int, default=20, help='IC decay lags 1..max-lag')
parser.add_argument('--refresh-returns', action='store_true', help='Re-pull CRSP returns')
parser.add_argument('--show', action='store_true', help='Open the plots in a window at the end')
parser.add_argument('--start', help='First signal date to evaluate (default: first in the files)')
parser.add_argument('--end', help='Last signal date to evaluate (default: last in the files)')
args = parser.parse_args()
NEWS_ONLY = not args.all_stocks
MODE = 'all' if args.all_stocks else 'news'
SUFFIX = f'{args.col}_{MODE}_q{args.quantiles}_h{args.holding}_{args.weight}'
NQ = args.quantiles
QNAME = {5: 'quintile', 10: 'decile', 4: 'quartile', 3: 'tercile'}.get(NQ, 'quantile')

if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ## Printing helpers

def heading(text, char='='):
    print(f'\n{text}\n{char * len(text)}')


def pct(x, d=1):
    return '' if pd.isna(x) else f'{x:.{d}%}'


def num(x, d=2):
    return '' if pd.isna(x) else f'{x:.{d}f}'


def show_table(df, title, note=None):
    # First column left-aligned, the rest right-aligned, 3 spaces apart
    print(f'\n{title}')
    if note:
        print(f'  {note}')
    cols = [[str(c)] + [str(v) for v in df[c]] for c in df.columns]
    widths = [max(len(v) for v in col) for col in cols]
    for i in range(len(df) + 1):
        cells = [cols[0][i].ljust(widths[0])] + [cols[j][i].rjust(widths[j]) for j in range(1, len(cols))]
        print('  ' + '   '.join(cells))


def rel(path):
    try:
        return Path(path).relative_to(HERE)
    except ValueError:
        return path


def col_description(col):
    if col == 'S':
        return "the day's mean story sentiment"
    L = col.split('_')[-1]
    return f'daily mean story sentiment, summed over the last {L} trading day{"s" if L != "1" else ""}'


def load_signal(path):
    try:
        from sentlib.signal import load_signal as lib_load
        return lib_load(path)
    except ImportError:
        path = Path(path)
        sig = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_hdf(path, 'sentiment')
        sig['date'] = pd.to_datetime(sig['date'])
        return sig.sort_values(['date', 'permno']).reset_index(drop=True)


def signal_name(path):
    """data/top1000_..._rp_signal.parquet -> rp; legacy ..._sentiment.h5 -> sentiment."""
    stem = TAG_RE.sub('', Path(path).stem).strip('_')
    return re.sub(r'_signal$', '', stem) or Path(path).stem


def membership_permnos(path):
    m = TAG_RE.search(Path(path).name)
    f = DATA_DIR / f'{m.group(0)}_membership.h5' if m else None
    if f is None or not f.exists():
        return []
    mem = pd.read_hdf(f, 'membership')
    return mem.loc[mem['effective_from'].notna(), 'permno'].astype(int).tolist()


# ## Signals and returns

# Signals with the same name from different universes keep their universe tag
names = [signal_name(p) for p in args.signals]
names = [n if names.count(n) == 1 else f'{TAG_RE.search(Path(p).name).group(0)}_{n}'
         for n, p in zip(names, args.signals)]
if len(set(names)) < len(names):
    sys.exit(f'two signal files give the same name: {names}')
signals = {n: load_signal(p) for n, p in zip(names, args.signals)}

permnos = set()
for p, s in zip(args.signals, signals.values()):
    permnos |= set(membership_permnos(p)) | set(s['permno'].astype(int))
start = min(s.date.min() for s in signals.values()).strftime('%Y-%m-%d')
end = max(s.date.max() for s in signals.values()).strftime('%Y-%m-%d')
# Returns are pulled for the files' full range (same cache file); evaluation uses --start/--end
ret_start, ret_end = start, end
if args.start or args.end:
    lo, hi = pd.Timestamp(args.start or start), pd.Timestamp(args.end or end)
    signals = {n: s[(s.date >= lo) & (s.date <= hi)].reset_index(drop=True) for n, s in signals.items()}
    if any(s.empty for s in signals.values()):
        sys.exit(f'no signal dates between {lo:%Y-%m-%d} and {hi:%Y-%m-%d} in some file')
    start = min(s.date.min() for s in signals.values()).strftime('%Y-%m-%d')
    end = max(s.date.max() for s in signals.values()).strftime('%Y-%m-%d')

heading('NEWS-SENTIMENT BACKTEST')
holding = ('rebalanced daily, each held 1 day' if args.holding == 1 else
           f'held {args.holding} days (average of {args.holding} overlapping portfolios)')
settings = [
    ('Signal', f'{args.col}: {col_description(args.col)}'),
    ('Period', f'{start} to {end}' + (' (--start/--end)' if args.start or args.end else '')),
    ('Timing', 'signal on day d is traded at the close of d; first return earned is day d+1'),
    ('Stocks', 'stock-days with news only' if NEWS_ONLY else 'all member stock-days (no news = 0)'),
    ('Portfolios', f'{NQ} {QNAME}s by signal (Q1 lowest, Q{NQ} highest), '
                   f'{"equal" if args.weight == "equal" else "market-cap"} weight, {holding}'),
    ('Trading costs', ', '.join(f'{c:g}' for c in args.cost_bps) + ' bps per dollar traded'),
    ('IC', f'{args.method.capitalize()} correlation across stocks each day, averaged over days; '
           't-stats are Newey-West'),
]
for k, v in settings:
    print(f'  {k:<14}{v}')

print('\nSignals')
w = max(len(n) for n in signals) + 2
for (name, s), p in zip(signals.items(), args.signals):
    print(f'  {name:<{w}}{len(s):>10,} stock-days, {s.permno.nunique():,} stocks, '
          f'{s.date.min():%Y-%m-%d} to {s.date.max():%Y-%m-%d}, '
          f'{(s[n_col_for(args.col)] > 0).mean():.1%} with news   ({rel(p)})')

# Cached under data/; WRDS is only contacted when the cache is missing
from backtest.returns import PAD_DAYS, default_cache_path
end_pull = (pd.Timestamp(ret_end) + pd.Timedelta(days=PAD_DAYS)).strftime('%Y-%m-%d')
cache = default_cache_path(permnos, ret_start, end_pull)
conn = None
if args.refresh_returns or not cache.exists():
    conn = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME', 'cpehlivanian'))
returns = get_returns(conn, sorted(permnos), ret_start, ret_end, cache_path=cache, refresh=args.refresh_returns)
if conn is not None:
    conn.close()

# Look-ahead checks: same-day return as signal -> IC ~0 and long-short ~0;
# next-day return as signal -> IC ~1 and a huge long-short Sharpe
print('\nLook-ahead checks (stop the script if the signal/return pairing is off by a day)')
timing = check_timing(returns, start, end)
print(f"  Signal = day d return:    IC with day d+1 return {timing['ic_same_day_return']:7.3f}  "
      f"(should be near 0)   passed")
print(f"  Signal = day d+1 return:  IC with day d+1 return {timing['ic_next_day_return']:7.3f}  "
      f"(should be 1)        passed")

from backtest.returns import wide_returns
R, _, alive = wide_returns(returns)
for shift, ok, label, expect in [(0, lambda sr: abs(sr) < 3, 'day d return:   ', 'near 0)'),
                                 (-1, lambda sr: sr > 10, 'day d+1 return: ', 'large) ')]:
    x = R.shift(shift).where(alive).loc[start:end].stack().rename('sent_1').reset_index()
    x.columns = ['date', 'permno', 'sent_1']
    x['n_1'] = 1
    sr = performance_table(quantile_portfolios(x, 'sent_1', returns)['long_short'].to_frame()).loc['long_short', 'sharpe']
    assert ok(sr), f'portfolio timing test failed: Sharpe {sr:.2f} for return shift {shift}'
    print(f'  Sort on {label}  long-short Sharpe {sr:7.2f}           (should be {expect}  passed')

fwd = forward_returns(returns, sorted(set(args.horizons)))


# ## Plots

def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252)


def plot_equity(res_by_cost, name, out):
    res0 = res_by_cost[min(res_by_cost)]
    q = res0['quantiles']
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, q.shape[1]))
    for c, col in zip(shades, q.columns):
        ax[0].plot((1 + q[col].fillna(0)).cumprod(), color=c, lw=1.5,
                   label=f'{col.upper()}  Sharpe {sharpe(q[col]):.2f}')
    ax[0].set_title(f'{QNAME.capitalize()} portfolios before costs (Q1 = lowest signal)')
    ax[0].legend(frameon=False, fontsize=9, loc='upper left')
    ax[1].plot((1 + res0['long_short'].fillna(0)).cumprod(), color='#2a78d6', lw=1.5,
               label=f"before costs  Sharpe {sharpe(res0['long_short']):.2f}")
    for cost, color in zip(sorted(c for c in res_by_cost if c > 0), ['#eb6834', '#1baf7a', '#e87ba4']):
        ax[1].plot((1 + res_by_cost[cost]['long_short_net'].fillna(0)).cumprod(),
                   color=color, lw=1.5,
                   label=f"after {cost:g} bps costs  Sharpe {sharpe(res_by_cost[cost]['long_short_net']):.2f}")
    ax[1].set_title(f'Long Q{q.shape[1]}, short Q1')
    ax[1].legend(frameon=False, fontsize=9, loc='lower left')
    for a in ax:
        a.axhline(1, color='0.6', lw=0.8)
        a.grid(alpha=0.25)
        a.set_ylabel('growth of $1')
    fig.suptitle(f'{name}: equity curves')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    if not args.show:
        plt.close(fig)


def plot_ic_series(ic, name, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
    for c, col in zip(colors, ic.columns):
        h = col.replace('ic_', '')
        ax.plot(ic[col].rolling(63, min_periods=40).mean(), color=c, lw=1.5, label=f'{h}-day return')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_title(f'{name}: daily IC, rolling 3-month mean')
    ax.legend(frameon=False, fontsize=8, ncol=len(ic.columns))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    if not args.show:
        plt.close(fig)


def plot_decay(dec, name, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, kind, title in [(ax[0], 'lag', 'return on day d+k'),
                           (ax[1], 'cum', 'return from d to d+k')]:
        a.errorbar(dec.index, dec[f'ic_{kind}'], yerr=2 * dec[f'se_{kind}'], fmt='o-',
                   color='#2a78d6', ms=4, lw=1.5, capsize=2)
        a.axhline(0, color='0.4', lw=0.8)
        a.set_xlabel('k (trading days)')
        a.set_title(f'IC with the {title}, ±2 s.e.')
        a.grid(alpha=0.25)
    ax[0].set_ylabel('mean IC')
    fig.suptitle(f'{name}: IC decay')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    if not args.show:
        plt.close(fig)


# ## Per-signal evaluation

DECAY_SHOWN = [k for k in [1, 2, 3, 4, 5, 10, 15, 20] if k <= args.max_lag]
rows = []
for name, sig in signals.items():
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    # IC by horizon
    ic = pd.concat([daily_ic(sig, args.col, fwd, h, args.method, NEWS_ONLY) for h in args.horizons], axis=1)
    summ = pd.DataFrame([ic_summary(ic[f'ic_{h}'], h) for h in args.horizons]).set_index('horizon')
    dec = ic_decay(sig, args.col, returns, range(1, args.max_lag + 1), args.method, NEWS_ONLY)

    # Quantile portfolios (gross is the same for every cost)
    res = {c: quantile_portfolios(sig, args.col, returns, NQ, args.holding,
                                  args.weight, NEWS_ONLY, c) for c in args.cost_bps}
    r0 = res[args.cost_bps[0]]
    daily = r0['quantiles'].copy()
    daily['long_short'] = r0['long_short']
    for c in args.cost_bps:
        if c > 0:
            daily[f'long_short_net_{c:g}bps'] = res[c]['long_short_net']
    perf = performance_table(daily)
    to = r0['turnover'].mean()
    names = r0['n_names'].mean()
    perf['avg_turnover'] = [to['ls'] if i.startswith('long_short') else to[i] for i in perf.index]
    perf['avg_names'] = [names.get(i, np.nan) for i in perf.index]

    # Save
    files = {'equity': out / f'equity_{SUFFIX}.png', 'ic': out / f'ic_timeseries_{SUFFIX}.png',
             'decay': out / f'ic_decay_{SUFFIX}.png'}
    ic.to_csv(out / f'ic_daily_{SUFFIX}.csv')
    summ.to_csv(out / f'ic_summary_{SUFFIX}.csv')
    dec.to_csv(out / f'ic_decay_{SUFFIX}.csv')
    daily.join(r0['turnover'].add_prefix('turnover_')).to_csv(out / f'quantile_returns_{SUFFIX}.csv')
    perf.to_csv(out / f'performance_{SUFFIX}.csv')
    plot_equity(res, name, files['equity'])
    plot_ic_series(ic, name, files['ic'])
    plot_decay(dec, name, files['decay'])

    # Report
    heading(f'SIGNAL: {name}')

    years = [c for c in summ.columns if c.startswith('mean_')]
    t = pd.DataFrame({'Horizon': [f'{h} day{"s" if h > 1 else ""}' for h in summ.index],
                      'Mean IC': summ['mean'].map(lambda x: num(x, 4)),
                      't-stat': summ['nw_t'].map(num),
                      'Days IC > 0': summ['hit_rate'].map(pct)})
    for y in years:
        t[y.replace('mean_', '')] = summ[y].map(lambda x: num(x, 4)).values
    show_table(t, 'IC by horizon',
               'correlation of the signal on day d with the return from the close of d to the close of d+h; '
               'last columns: mean IC by year')

    d = dec.loc[DECAY_SHOWN]
    t = pd.DataFrame({'k': d.index,
                      'IC, return on d+k': d['ic_lag'].map(lambda x: num(x, 4)), 't-stat': d['t_lag'].map(num),
                      'IC, return d to d+k': d['ic_cum'].map(lambda x: num(x, 4)), 't-stat ': d['t_cum'].map(num)})
    show_table(t, 'IC decay', f'k = 1..{args.max_lag} trading days ahead; all k in the CSV and the plot')

    labels = {f'q{j}': f'Q{j}' + (' (lowest)' if j == 1 else ' (highest)' if j == NQ else '')
              for j in range(1, NQ + 1)}
    labels['long_short'] = f'Long Q{NQ}, short Q1'
    for c in args.cost_bps:
        if c > 0:
            labels[f'long_short_net_{c:g}bps'] = f'Long-short after {c:g} bps'
    t = pd.DataFrame({'Portfolio': [labels[i] for i in perf.index],
                      'Ann. return': perf['ann_ret'].map(pct),
                      'Ann. volatility': perf['ann_vol'].map(pct),
                      'Sharpe': perf['sharpe'].map(num),
                      'Max drawdown': perf['max_dd'].map(pct),
                      'Days up': perf['hit_rate'].map(pct),
                      'Turnover per day': perf['avg_turnover'].map(lambda x: pct(x, 0)),
                      'Stocks': perf['avg_names'].map(lambda x: '' if pd.isna(x) else f'{x:.0f}')})
    show_table(t, f'{QNAME.capitalize()} portfolios, returns from {r0["quantiles"].index[0]:%Y-%m-%d} to '
                  f'{r0["quantiles"].index[-1]:%Y-%m-%d}',
               'returns before costs unless stated; turnover = share of the portfolio traded each day (one way)')

    print('\nPlots')
    print(f'  Equity curves      {rel(files["equity"])}')
    print(f'  IC over time       {rel(files["ic"])}')
    print(f'  IC decay           {rel(files["decay"])}')
    print(f'Tables (CSV) are in {rel(out)}/')

    row = {'signal': name}
    for h in args.horizons:
        row[f'ic_{h}'] = summ.loc[h, 'mean']
        row[f't_{h}'] = summ.loc[h, 'nw_t']
    row['ls_sharpe'] = perf.loc['long_short', 'sharpe']
    row['ls_ann_ret'] = perf.loc['long_short', 'ann_ret']
    for c in args.cost_bps:
        if c > 0:
            row[f'ls_sharpe_{c:g}bps'] = perf.loc[f'long_short_net_{c:g}bps', 'sharpe']
    row['turnover'] = to['ls']
    row['names_per_q'] = names.mean()
    rows.append(row)

comp = pd.DataFrame(rows).set_index('signal')
comp_file = OUT_DIR / f'comparison_{SUFFIX}.csv'
comp.to_csv(comp_file)

if len(comp) > 1:
    heading('SUMMARY ACROSS SIGNALS')
    t = pd.DataFrame({'Signal': comp.index})
    for h in args.horizons:
        t[f'IC {h}d'] = comp[f'ic_{h}'].map(lambda x: num(x, 4)).values
        t[f't {h}d'] = comp[f't_{h}'].map(num).values
    t['L/S Sharpe'] = comp['ls_sharpe'].map(num).values
    for c in args.cost_bps:
        if c > 0:
            t[f'after {c:g} bps'] = comp[f'ls_sharpe_{c:g}bps'].map(num).values
    t['Turnover/day'] = comp['turnover'].map(lambda x: pct(x, 0)).values
    show_table(t, 'Mean IC and t-stat by horizon; long-short Sharpe before and after costs')
    print(f'Saved to {rel(comp_file)}')

if args.show:
    plt.show()
