#!/usr/bin/env python
# coding: utf-8

# # RavenPack News Sentiment Signal
# Point-in-Time (PIT) daily sentiment for the top-N U.S. companies by market cap
#
# Data: RavenPack Analytics 1.0, Dow Jones edition (ravenpack_dj.rpa_djpr_equities_YYYY).
# Universe: the PIT membership file written by Big_10_Index.py, so run that first:
#     python Big_10_Index.py --top-n 1000 --start 2020-01-01 --end 2024-12-31
#
# Usage:
#     python ravenpack_sentiment.py                                  # top 1000, 2020-01-01..2024-12-31
#     python ravenpack_sentiment.py --top-n 100 --start 2021-01-01 --end 2022-12-31
#
# Signal construction (one row per stock per trading day d):
#   1. Keep stories with relevance >= 90 (the company is central to the story),
#      a detected event (event_sentiment_score not null), and
#      event_similarity_days >= 1 (no similar event in the previous day, i.e.
#      the first story about the event, not a repeat).
#   2. A story counts toward the first trading day d whose 15:30 New York time
#      is after the story's timestamp. Stories after 15:30, or on weekends and
#      holidays, therefore go to the next trading day.
#   3. S_d = mean event_sentiment_score over the stock's stories on day d
#      (scale -1..+1, 0 is neutral), N_d = number of stories; S_d = 0 if none.
#   4. sent_L on day d = S_d + S_{d-1} + ... + S_{d-L+1}, for L in --windows.
#
# Timing: sent_L on day d uses only stories published before 15:30 ET on d,
# so it can be traded at the close of d. Pair it with returns from the close of
# d onward (the first return is day d+1's), never with day d's own return.

# In[1]:


import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wrds

parser = argparse.ArgumentParser(description='PIT RavenPack sentiment signal for a top-N market-cap universe.')
parser.add_argument('--top-n', type=int, default=1000, help='Universe size used in Big_10_Index.py (default 1000)')
parser.add_argument('--start', default='2020-01-01', help='First signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--end', default='2024-12-31', help='Last signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--windows', type=int, nargs='+', default=[1, 5, 20], help='Summation windows L in trading days')
parser.add_argument('--min-relevance', type=float, default=90, help='Minimum RavenPack relevance (0-100)')
parser.add_argument('--cutoff', default='15:30', help='New York time after which news counts toward the next trading day')
args = parser.parse_args()

N = args.top_n
START = pd.Timestamp(args.start).strftime('%Y-%m-%d')
END = pd.Timestamp(args.end).strftime('%Y-%m-%d')
WINDOWS = sorted(args.windows)
CUTOFF = pd.Timedelta(args.cutoff + ':00')

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
TAG = f'top{N}_{START}_{END}'
MEMBERSHIP_H5 = DATA_DIR / f'{TAG}_membership.h5'
SIGNAL_H5 = DATA_DIR / f'{TAG}_sentiment.h5'

if not MEMBERSHIP_H5.exists():
    raise SystemExit(f'{MEMBERSHIP_H5} not found: run Big_10_Index.py --top-n {N} --start {START} --end {END} first')

conn = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME', 'cpehlivanian'))


# ## Step 1: Universe and trading calendar

# In[2]:


# Membership: a share class (permno) is in the universe on day d if
# effective_from <= d <= effective_to (lists are chosen at each prior year-end).
membership = pd.read_hdf(MEMBERSHIP_H5, key='membership')
membership = membership.dropna(subset=['effective_from'])
permnos = sorted(membership['permno'].unique().tolist())
print(f'Universe: {len(permnos)} share classes ever in the top {N}')

# Trading calendar, starting early enough that the longest window is full on START
lookback_start = (pd.Timestamp(START) - pd.Timedelta(days=2 * max(WINDOWS) + 10)).strftime('%Y-%m-%d')
calendar = conn.raw_sql(f"""
    SELECT date FROM crsp.dsi
    WHERE date BETWEEN '{lookback_start}' AND '{END}'
    ORDER BY date
""")
trading_days = pd.DatetimeIndex(pd.to_datetime(calendar['date']))
print(f'Trading days {trading_days[0]:%Y-%m-%d}..{trading_days[-1]:%Y-%m-%d}: {len(trading_days)}')


# ## Step 2: Link RavenPack entities to CRSP permnos

# In[3]:


# RavenPack gives each company an rp_entity_id with dated 9-digit CUSIPs.
# CRSP stocknames gives each permno its dated 8-digit historical CUSIP (ncusip).
# A link is valid on the days where both date ranges overlap.
# One entity can map to several permnos (share classes, e.g. GOOG and GOOGL);
# each permno then gets the company's sentiment.
permnos_str = ', '.join(str(p) for p in permnos)
crsp_names = conn.raw_sql(f"""
    SELECT DISTINCT permno, permco, ncusip, namedt, nameenddt
    FROM crsp.stocknames
    WHERE permno IN ({permnos_str})
      AND ncusip IS NOT NULL
      AND nameenddt >= '{lookback_start}' AND namedt <= '{END}'
""")
rp_cusips = conn.raw_sql("""
    SELECT rp_entity_id, data_value AS cusip, range_start, range_end
    FROM ravenpack_common.rpa_entity_mappings
    WHERE data_type = 'CUSIP' AND entity_type = 'COMP'
""")
rp_cusips['ncusip'] = rp_cusips['cusip'].str[:8]

link = rp_cusips.merge(crsp_names, on='ncusip', how='inner')
for col in ['range_start', 'range_end', 'namedt', 'nameenddt']:
    link[col] = pd.to_datetime(link[col])
far_future = pd.Timestamp('2100-01-01')
link['link_start'] = link[['range_start', 'namedt']].max(axis=1)
link['link_end'] = link[['range_end', 'nameenddt']].fillna(far_future).min(axis=1)
link = link[link['link_start'] <= link['link_end']]
link = (link.groupby(['rp_entity_id', 'permno', 'permco'], as_index=False)
            .agg(link_start=('link_start', 'min'), link_end=('link_end', 'max')))
link[['permno', 'permco']] = link[['permno', 'permco']].astype('int64')

linked = set(link['permno'])
print(f'Linked {len(linked)} of {len(permnos)} permnos to {link["rp_entity_id"].nunique()} RavenPack entities')
unlinked = membership[~membership['permno'].isin(linked)][['permno', 'ticker']].drop_duplicates('permno')
if len(unlinked):
    print(f'Unlinked permnos ({len(unlinked)}), e.g.: {unlinked.head(10).to_dict("records")}')


# ## Step 3: Pull filtered RavenPack stories

# In[4]:


entities_str = ', '.join(f"'{e}'" for e in sorted(link['rp_entity_id'].unique()))
# Pull one day before lookback_start so stories after the cutoff that day are kept
pull_start = (pd.Timestamp(lookback_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
stories = []
for year in range(int(pull_start[:4]), int(END[:4]) + 1):
    df = conn.raw_sql(f"""
        SELECT timestamp_utc, rp_entity_id, rp_story_id, event_sentiment_score AS ess
        FROM ravenpack_dj.rpa_djpr_equities_{year}
        WHERE rp_entity_id IN ({entities_str})
          AND entity_type = 'COMP'
          AND relevance >= {args.min_relevance}
          AND event_sentiment_score IS NOT NULL
          AND event_similarity_days >= 1
          AND timestamp_utc >= '{pull_start}' AND timestamp_utc < '{END}'::date + 1
    """)
    print(f'{year}: {len(df):,} stories')
    stories.append(df)
stories = pd.concat(stories, ignore_index=True)
stories['ess'] = stories['ess'].astype(float)


# ## Step 4: Assign each story to a trading day

# In[5]:


# Convert to New York time; a story at or after the cutoff moves to the next
# calendar day, then rolls forward to the first trading day on or after it.
ts_ny = pd.to_datetime(stories['timestamp_utc']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
day = ts_ny.dt.tz_localize(None).dt.normalize()
after_cutoff = (ts_ny.dt.tz_localize(None) - day) >= CUTOFF
candidate = day + pd.to_timedelta(after_cutoff.astype(int), unit='D')
pos = trading_days.searchsorted(candidate.values, side='left')
in_range = pos < len(trading_days)
stories = stories[in_range].copy()
stories['date'] = trading_days[pos[in_range]]


# ## Step 5: Daily sentiment per company, then per permno

# In[6]:


daily = (stories.groupby(['rp_entity_id', 'date'])['ess']
                .agg(S='mean', n='count').reset_index())

# Map to permnos, keeping only links valid on that date
daily = daily.merge(link, on='rp_entity_id', how='inner')
daily = daily[(daily['date'] >= daily['link_start']) & (daily['date'] <= daily['link_end'])]
# If two entities map to the same permno on a day, combine them story-weighted
daily['S_x_n'] = daily['S'] * daily['n']
daily = daily.groupby(['permno', 'date'], as_index=False)[['S_x_n', 'n']].sum()
daily['S'] = daily['S_x_n'] / daily['n']

# Full grid permno x trading day, S = 0 and n = 0 on days without news
grid = pd.MultiIndex.from_product([permnos, trading_days], names=['permno', 'date'])
panel = (daily.set_index(['permno', 'date'])[['S', 'n']]
              .reindex(grid, fill_value=0).sort_index())

# Rolling sums over L trading days within each permno
for L in WINDOWS:
    g = panel.groupby(level='permno')
    panel[f'sent_{L}'] = g['S'].transform(lambda x: x.rolling(L, min_periods=L).sum())
    panel[f'n_{L}'] = g['n'].transform(lambda x: x.rolling(L, min_periods=L).sum())
panel = panel.reset_index()


# ## Step 6: Restrict to PIT membership and save

# In[7]:


# Keep (permno, date) only while the permno is in the universe and START <= date <= END
members = membership[['permno', 'permco', 'ticker', 'effective_from', 'effective_to']]
signal = panel[(panel['date'] >= START) & (panel['date'] <= END)].merge(members, on='permno', how='inner')
signal = signal[(signal['date'] >= signal['effective_from']) & (signal['date'] <= signal['effective_to'])]
signal = signal.drop(columns=['effective_from', 'effective_to'])
signal['linked'] = signal['permno'].isin(linked)

cols = ['date', 'permno', 'permco', 'ticker', 'linked', 'S', 'n'] + \
       [c for L in WINDOWS for c in (f'sent_{L}', f'n_{L}')]
signal = signal[cols].sort_values(['date', 'permno']).reset_index(drop=True)
signal = signal.astype({'permno': 'int64', 'permco': 'int64', 'n': 'int64', 'S': 'float64',
                        **{f'sent_{L}': 'float64' for L in WINDOWS},
                        **{f'n_{L}': 'int64' for L in WINDOWS}})

signal.to_hdf(SIGNAL_H5, key='sentiment', mode='w', format='table',
              data_columns=['date', 'permno'])
print(f'Saved {len(signal):,} rows ({signal["permno"].nunique()} permnos, '
      f'{signal["date"].nunique()} days) to {SIGNAL_H5}')


# ## Step 7: Coverage summary

# In[8]:


print('\nShare of member stock-days with at least one story in the window:')
for L in WINDOWS:
    print(f'  L={L:>2}: {(signal[f"n_{L}"] > 0).mean():.1%}   '
          f'(linked permnos only: {(signal.loc[signal["linked"], f"n_{L}"] > 0).mean():.1%})')
print('\nsent_L distribution on stock-days with news:')
print(pd.DataFrame({f'sent_{L}': signal.loc[signal[f'n_{L}'] > 0, f'sent_{L}'].describe()
                    for L in WINDOWS}).round(3))
by_year = signal.assign(year=signal['date'].dt.year).groupby('year')
print('\nBy year: members per day and share of stock-days with news (L=1):')
print(pd.DataFrame({'members_per_day': by_year.size() / by_year['date'].nunique(),
                    'news_share_L1': by_year['n'].apply(lambda n: (n > 0).mean())}).round(3))

conn.close()
