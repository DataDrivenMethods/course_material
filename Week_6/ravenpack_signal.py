#!/usr/bin/env python
# coding: utf-8

# # RavenPack News Sentiment Signal (sentlib version)
# Point-in-Time (PIT) daily sentiment for the top-N U.S. companies by market cap.
#
# Same construction as ravenpack_sentiment.py, built on the shared sentlib
# library so the FinBERT signal and the backtest use identical timing,
# aggregation and membership rules (see SPEC.md).
#
# Data: RavenPack Analytics 1.0, Dow Jones edition (ravenpack_dj.rpa_djpr_equities_YYYY).
# Universe: the PIT membership file written by Big_10_Index.py, so run that first.
#
# Usage:
#     python ravenpack_signal.py --start 2020-01-01 --end 2024-12-31                    # top 1000, L = 1, name rp
#     python ravenpack_signal.py --start 2020-01-01 --end 2024-12-31 --windows 1 5 20 --name rp_w
#     python ravenpack_signal.py --start 2020-01-01 --end 2024-12-31 --exclude-groups stock-prices --name rp_noprice
# --start and --end are required.
#
# Signal (one row per member stock per trading day d):
#   1. Stories with relevance >= --min-relevance, a detected event and
#      event_similarity_days >= --min-similarity-days (first story on the event).
#   2. A story counts toward the first trading day d whose --cutoff New York
#      time is after its timestamp (after the cutoff, weekends, holidays -> next day).
#   3. S_d = mean event_sentiment_score (-1..+1), n_d = story count, 0 if none.
#   4. sent_L(d) = S_d + ... + S_{d-L+1}, n_L(d) = stories in the window.
# Trade at the close of d; the first return paired with sent_L(d) is day d+1's.
#
# Stories (relevance >= 75, no other filters) are cached per year under
# data/rp_stories/{tag}/, so changing the filters does not hit WRDS again.

# In[1]:


import argparse

import pandas as pd

from sentlib import ravenpack as rp
from sentlib.signal import make_signal, save_signal
from sentlib.wrds_io import DATA_DIR, connect, load_universe, trading_calendar, universe_tag

parser = argparse.ArgumentParser(description='PIT RavenPack sentiment signal for a top-N market-cap universe.')
parser.add_argument('--top-n', type=int, default=1000, help='Universe size used in Big_10_Index.py (default 1000)')
parser.add_argument('--start', required=True, help='First signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--end', required=True, help='Last signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--update-period', '--update_period', dest='update_period', default='year',
                    choices=['year', 'month', 'day'], help='Universe update period used in Big_10_Index.py')
parser.add_argument('--windows', type=int, nargs='+', default=[1], help='Summation windows L in trading days (default 1)')
parser.add_argument('--min-relevance', type=float, default=90, help='Minimum RavenPack relevance (0-100)')
parser.add_argument('--min-similarity-days', type=float, default=1,
                    help='Minimum event_similarity_days (novelty); negative = no novelty filter')
parser.add_argument('--no-require-event', action='store_true',
                    help='Keep stories without a detected event (their score is missing, so they still do not count)')
parser.add_argument('--exclude-groups', nargs='*', default=[], help='RavenPack group values to drop, e.g. stock-prices')
parser.add_argument('--exclude-news-types', nargs='*', default=[], help='RavenPack news_type values to drop')
parser.add_argument('--max-entities-per-story', type=int, default=None,
                    help='Drop stories naming more universe companies than this')
parser.add_argument('--cutoff', default='15:30', help='New York time after which news counts toward the next trading day')
parser.add_argument('--name', default='rp', help='Signal name in the output file (default rp)')
parser.add_argument('--refresh-cache', action='store_true', help='Pull stories from WRDS even if cached')
args = parser.parse_args()

START = pd.Timestamp(args.start).strftime('%Y-%m-%d')
END = pd.Timestamp(args.end).strftime('%Y-%m-%d')
WINDOWS = sorted(set(args.windows))
TAG = universe_tag(args.top_n, START, END, args.update_period)     # names the output signal
SCORE_COL = 'event_sentiment_score'

story_filter = rp.StoryFilter(
    min_relevance=args.min_relevance,
    require_event=not args.no_require_event,
    min_similarity_days=None if args.min_similarity_days < 0 else args.min_similarity_days,
    exclude_groups=tuple(args.exclude_groups),
    exclude_news_types=tuple(args.exclude_news_types),
    max_entities_per_story=args.max_entities_per_story,
)

conn = connect()


# ## Step 1: Universe, calendar, entity link

# In[2]:


# Any Big_10_Index.py build covering START..END, cut to the lists held in START..END
UNIVERSE, membership = load_universe(args.top_n, START, END, args.update_period)
CACHE_DIR = DATA_DIR / 'rp_stories' / UNIVERSE
permnos = sorted(membership['permno'].unique().tolist())
print(f'Universe {UNIVERSE} (lists held {START}..{END}): {len(permnos)} share classes')

# Calendar starts early enough that the longest window is full on START
lookback_start = (pd.Timestamp(START) - pd.Timedelta(days=2 * max(WINDOWS) + 10)).strftime('%Y-%m-%d')
calendar = trading_calendar(conn, lookback_start, END)
print(f'Trading days {calendar[0]:%Y-%m-%d}..{calendar[-1]:%Y-%m-%d}: {len(calendar)}')

link = rp.build_link(conn, permnos, lookback_start, END)
linked = set(link['permno'])
print(f'Linked {len(linked)} of {len(permnos)} permnos to {link["rp_entity_id"].nunique()} RavenPack entities')
unlinked = membership[~membership['permno'].isin(linked)][['permno', 'ticker']].drop_duplicates('permno')
if len(unlinked):
    print(f'Unlinked permnos ({len(unlinked)}): {unlinked.head(10).to_dict("records")}')


# ## Step 2: Stories (cached), then filters

# In[3]:


# One day before the calendar start, so stories after the cutoff that day count
pull_start = (pd.Timestamp(lookback_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
stories = rp.fetch_stories(conn, link['rp_entity_id'].unique().tolist(), pull_start, END,
                           cache_dir=CACHE_DIR, refresh=args.refresh_cache)
conn.close()
print(f'Cached stories (relevance >= 75): {len(stories):,} rows')
stories = rp.apply_filter(stories, story_filter)
print(f'After filters {story_filter}: {len(stories):,} rows')


# ## Step 3: Signal and save

# In[4]:


signal = make_signal(stories, SCORE_COL, link, membership, calendar, WINDOWS, START, END,
                     cutoff=args.cutoff)
params = {'source': 'ravenpack_dj.rpa_djpr_equities', 'score_col': SCORE_COL,
          'top_n': args.top_n, 'update_period': args.update_period,
          'start': START, 'end': END, 'windows': WINDOWS,
          'cutoff': args.cutoff, 'filter': story_filter.__dict__ | {
              'exclude_groups': list(story_filter.exclude_groups),
              'exclude_news_types': list(story_filter.exclude_news_types)},
          'filtered_story_rows': int(len(stories)), 'linked_permnos': len(linked),
          'universe': UNIVERSE, 'universe_permnos': len(permnos), 'story_cache': str(CACHE_DIR)}
path = save_signal(signal, TAG, args.name, params)
print(f'Saved {len(signal):,} rows ({signal["permno"].nunique()} permnos, '
      f'{signal["date"].nunique()} days) to {path}')


# ## Step 4: Coverage summary

# In[5]:


is_linked = signal['permno'].isin(linked)
print('\nShare of member stock-days with at least one story in the window:')
for L in WINDOWS:
    print(f'  L={L:>2}: {(signal[f"n_{L}"] > 0).mean():.1%}   '
          f'(linked permnos only: {(signal.loc[is_linked, f"n_{L}"] > 0).mean():.1%})')
print('\nsent_L distribution on stock-days with news:')
print(pd.DataFrame({f'sent_{L}': signal.loc[signal[f'n_{L}'] > 0, f'sent_{L}'].describe()
                    for L in WINDOWS}).round(3))
by_year = signal.assign(year=signal['date'].dt.year).groupby('year')
print('\nBy year: members per day and share of stock-days with news (L=1):')
print(pd.DataFrame({'members_per_day': by_year.size() / by_year['date'].nunique(),
                    'news_share_L1': by_year['n'].apply(lambda n: (n > 0).mean())}).round(3))
