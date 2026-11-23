#!/usr/bin/env python
# coding: utf-8

# # FinBERT News Sentiment Signal
# Same construction as ravenpack_signal.py, but each story is scored from its
# headline with ProsusAI/finbert (finbert_score = p_pos - p_neg, -1..+1)
# instead of RavenPack's event_sentiment_score. Timing, aggregation and
# membership come from sentlib, so the two signals are directly comparable.
#
# Usage:
#     python finbert_signal.py --start 2020-01-01 --end 2024-12-31 --name finbert_event                                   # RavenPack production filter
#     python finbert_signal.py --start 2020-01-01 --end 2024-12-31 --exclude-groups stock-prices --name finbert_event_noprice
#     python finbert_signal.py --start 2020-01-01 --end 2024-12-31 --no-require-event --exclude-groups stock-prices --name finbert_all_noprice
#     python finbert_signal.py --start 2020-01-01 --end 2024-12-31 --exclude-groups stock-prices --model data/finbert_models/ft_fwd5 --name ft_fwd5
#     python finbert_signal.py --start 2020-01-01 --end 2024-12-31 --compare                                              # FinBERT vs RavenPack table
#
# --no-require-event keeps stories without a RavenPack event (FinBERT can score
# any headline). Those stories have no event_similarity_days, so the novelty
# filter is applied to stories with an event only. Note that RavenPack's `group`
# is also missing without an event, so --exclude-groups only drops event stories.
#
# A story tagged to one entity with several events has several rows; they share
# a headline, so one row per (story, entity) is kept (dedupe_story=True).
# Headline scores are cached in data/finbert_scores/{model}.parquet, or for a
# fine-tuned model (--model data/finbert_models/NAME, see finbert_finetune.py)
# in that folder. A fine-tuned model is out of sample only after its training end.

# In[1]:


import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finbert_score import DEFAULT_MODEL, default_cache_path, load_model, score_headlines
from sentlib import ravenpack as rp
from sentlib.signal import load_signal, make_signal, save_signal
from sentlib.wrds_io import DATA_DIR, connect, load_universe, trading_calendar, universe_tag

parser = argparse.ArgumentParser(description='PIT FinBERT headline sentiment signal for a top-N market-cap universe.')
parser.add_argument('--top-n', type=int, default=1000, help='Universe size used in Big_10_Index.py (default 1000)')
parser.add_argument('--start', required=True, help='First signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--end', required=True, help='Last signal date, inclusive (YYYY-MM-DD)')
parser.add_argument('--update-period', '--update_period', dest='update_period', default='year',
                    choices=['year', 'month', 'day'], help='Universe update period used in Big_10_Index.py')
parser.add_argument('--windows', type=int, nargs='+', default=[1], help='Summation windows L in trading days (default 1)')
parser.add_argument('--min-relevance', type=float, default=90, help='Minimum RavenPack relevance (0-100)')
parser.add_argument('--min-similarity-days', type=float, default=1,
                    help='Minimum event_similarity_days (novelty) for event stories; negative = no novelty filter')
parser.add_argument('--no-require-event', action='store_true',
                    help='Keep stories without a RavenPack event (they get a FinBERT score)')
parser.add_argument('--exclude-groups', nargs='*', default=[], help='RavenPack group values to drop, e.g. stock-prices')
parser.add_argument('--exclude-news-types', nargs='*', default=[], help='RavenPack news_type values to drop')
parser.add_argument('--max-entities-per-story', type=int, default=None,
                    help='Drop stories naming more universe companies than this')
parser.add_argument('--cutoff', default='15:30', help='New York time after which news counts toward the next trading day')
parser.add_argument('--name', default='finbert_event', help='Signal name in the output file (default finbert_event)')
parser.add_argument('--refresh-cache', action='store_true', help='Pull stories from WRDS even if cached')
parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Hugging Face model (default {DEFAULT_MODEL})')
parser.add_argument('--batch-size', type=int, default=256, help='Scoring batch size (default 256)')
parser.add_argument('--no-fp16', action='store_true', help='Score in fp32 on the GPU')
parser.add_argument('--compare', action='store_true',
                    help='Only compare saved FinBERT signals with the RavenPack ones (writes data/finbert_vs_rp.csv)')
args = parser.parse_args()

START = pd.Timestamp(args.start).strftime('%Y-%m-%d')
END = pd.Timestamp(args.end).strftime('%Y-%m-%d')
WINDOWS = sorted(set(args.windows))
TAG = universe_tag(args.top_n, START, END, args.update_period)     # names the output signal
SCORE_COL = 'finbert_score'
# Any Big_10_Index.py build covering START..END, cut to the lists held in START..END
UNIVERSE, membership = load_universe(args.top_n, START, END, args.update_period)
CACHE_DIR = DATA_DIR / 'rp_stories' / UNIVERSE
# A fine-tuned model (folder from finbert_finetune.py): absolute path, so its cache rows match across runs
TRAINING = None
if Path(args.model).is_dir():
    args.model = str(Path(args.model).resolve())
    info = Path(args.model) / 'training.json'
    TRAINING = json.loads(info.read_text()) if info.exists() else None
SCORE_CACHE = default_cache_path(args.model)
MIN_SIM = None if args.min_similarity_days < 0 else args.min_similarity_days


def make_filter(require_event, exclude_groups, min_similarity_days=MIN_SIM):
    # Without the event requirement, novelty is applied separately (event stories only)
    return rp.StoryFilter(
        min_relevance=args.min_relevance,
        require_event=require_event,
        min_similarity_days=min_similarity_days if require_event else None,
        exclude_groups=tuple(exclude_groups),
        exclude_news_types=tuple(args.exclude_news_types),
        max_entities_per_story=args.max_entities_per_story,
    )


def filter_stories(stories, f, min_similarity_days=MIN_SIM):
    out = rp.apply_filter(stories, f)
    if not f.require_event and min_similarity_days is not None:
        sim = out['event_similarity_days']
        out = out[(sim.isna() | (sim >= min_similarity_days)).to_numpy()].reset_index(drop=True)
    return out


def get_stories(conn, link_ids, lookback_start):
    # One day before the calendar start, so stories after the cutoff that day count
    pull_start = (pd.Timestamp(lookback_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    return rp.fetch_stories(conn, link_ids, pull_start, END, cache_dir=CACHE_DIR, refresh=args.refresh_cache)


lookback_start = (pd.Timestamp(START) - pd.Timedelta(days=2 * max(WINDOWS) + 10)).strftime('%Y-%m-%d')
conn = connect()
permnos = sorted(membership['permno'].unique().tolist())
link = rp.build_link(conn, permnos, lookback_start, END)
linked = set(link['permno'])
print(f'Universe {UNIVERSE} (lists held {START}..{END}): {len(permnos)} share classes, {len(linked)} linked to '
      f'{link["rp_entity_id"].nunique()} RavenPack entities')


# ## Comparison mode: FinBERT vs RavenPack

# In[2]:


def xs_corr(a, b, col='sent_1', n_col='n_1', news_only=True):
    # Per-date cross-sectional Pearson / Spearman of col between two signals, then averaged
    m = a[['date', 'permno', col, n_col]].merge(b[['date', 'permno', col, n_col]], on=['date', 'permno'])
    if news_only:
        m = m[(m[f'{n_col}_x'] > 0) & (m[f'{n_col}_y'] > 0)]
    g = m.groupby('date')[[f'{col}_x', f'{col}_y']]
    pear = g.apply(lambda d: d[f'{col}_x'].corr(d[f'{col}_y']) if len(d) >= 10 else np.nan)
    spear = g.apply(lambda d: d[f'{col}_x'].corr(d[f'{col}_y'], method='spearman') if len(d) >= 10 else np.nan)
    return {'xs_pearson_mean': pear.mean(), 'xs_spearman_mean': spear.mean(),
            'pooled_pearson': m[f'{col}_x'].corr(m[f'{col}_y']), 'stock_days': len(m),
            'days': int(pear.notna().sum())}


if args.compare:
    conn.close()
    t0 = time.time()
    rows = []

    # Story level, on the RavenPack production filter (one row per story-entity)
    stories = filter_stories(get_stories(None, link['rp_entity_id'].unique().tolist(), lookback_start),
                             make_filter(True, []))
    stories = stories.drop_duplicates(['rp_story_id', 'rp_entity_id']).reset_index(drop=True)
    stories = stories.join(score_headlines(stories['headline'], SCORE_CACHE, model_name=args.model))
    rows.append({'comparison': 'story: finbert_score vs event_sentiment_score (filter 1)',
                 'pearson': stories['finbert_score'].corr(stories['event_sentiment_score']),
                 'spearman': stories['finbert_score'].corr(stories['event_sentiment_score'], method='spearman'),
                 'n': len(stories)})

    # Signal level: cross-sectional correlation of sent_1, stock-days where both have news
    sig = {name: load_signal(DATA_DIR / f'{TAG}_{name}_signal.parquet')
           for name in ['rp', 'rp_noprice', 'finbert_event', 'finbert_event_noprice', 'finbert_all_noprice']}
    for fb, rpn in [('finbert_event', 'rp'), ('finbert_event_noprice', 'rp_noprice'),
                    ('finbert_all_noprice', 'rp_noprice')]:
        for news_only in (True, False):
            c = xs_corr(sig[fb], sig[rpn], news_only=news_only)
            rows.append({'comparison': f'signal sent_1: {fb} vs {rpn} '
                                       f'({"both have news" if news_only else "all member stock-days"})',
                         'pearson': c['xs_pearson_mean'], 'spearman': c['xs_spearman_mean'],
                         'pooled_pearson': c['pooled_pearson'], 'n': c['stock_days']})

    # Coverage: share of member stock-days with n_1 > 0
    for name, s in sig.items():
        rows.append({'comparison': f'coverage n_1>0: {name}', 'coverage': (s['n_1'] > 0).mean(),
                     'n': len(s), 'stories_L1': int(s['n_1'].sum()),
                     'mean_sent_1_news': s.loc[s['n_1'] > 0, 'sent_1'].mean()})

    table = pd.DataFrame(rows)
    out = DATA_DIR / 'finbert_vs_rp.csv'
    table.to_csv(out, index=False)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 90):
        print(table.round(4).to_string(index=False))
    print(f'Saved {out} ({time.time() - t0:.0f} s)')
    raise SystemExit


# ## Step 1: Calendar, stories (cached), filters

# In[3]:


t0 = time.time()
calendar = trading_calendar(conn, lookback_start, END)
print(f'Trading days {calendar[0]:%Y-%m-%d}..{calendar[-1]:%Y-%m-%d}: {len(calendar)}')
stories = get_stories(conn, link['rp_entity_id'].unique().tolist(), lookback_start)
conn.close()
print(f'Cached stories (relevance >= 75): {len(stories):,} rows')
story_filter = make_filter(not args.no_require_event, args.exclude_groups)
stories = filter_stories(stories, story_filter)
print(f'After filters {story_filter}: {len(stories):,} rows, {stories["rp_story_id"].nunique():,} stories')


# ## Step 2: FinBERT scores (cached by headline)

# In[4]:


if TRAINING:
    print(f"Fine-tuned model trained on {TRAINING['train_start']}..{TRAINING['train_end']} "
          f"({TRAINING['label']}): only signal dates after {TRAINING['train_end']} are out of sample")
model, tokenizer, device = load_model(args.model, fp16=not args.no_fp16)
scores = score_headlines(stories['headline'], SCORE_CACHE, model_name=args.model, batch_size=args.batch_size,
                         model=model, tokenizer=tokenizer, device=device)
stories[SCORE_COL] = scores[SCORE_COL]
print(f'Scored ({time.time() - t0:.0f} s so far); finbert_score mean {stories[SCORE_COL].mean():.3f}, '
      f'share >0 {(stories[SCORE_COL] > 0).mean():.1%}, <0 {(stories[SCORE_COL] < 0).mean():.1%}')


# ## Step 3: Signal and save

# In[5]:


signal = make_signal(stories, SCORE_COL, link, membership, calendar, WINDOWS, START, END,
                     cutoff=args.cutoff, dedupe_story=True)
params = {'source': 'ravenpack_dj.rpa_djpr_equities headlines', 'score_col': SCORE_COL,
          'model_name': args.model, 'fp16': not args.no_fp16, 'max_length': 128,
          'score_cache': str(SCORE_CACHE), 'dedupe_story': True,
          'model_training': None if TRAINING is None else {k: TRAINING[k] for k in (
              'train_start', 'train_end', 'validation_start', 'label', 'label_horizon', 'best_epoch')},
          'top_n': args.top_n, 'update_period': args.update_period,
          'start': START, 'end': END, 'windows': WINDOWS,
          'cutoff': args.cutoff, 'filter': story_filter.__dict__ | {
              'min_similarity_days': MIN_SIM,
              'novelty_applies_to': 'all stories' if story_filter.require_event else 'event stories only',
              'exclude_groups': list(story_filter.exclude_groups),
              'exclude_news_types': list(story_filter.exclude_news_types)},
          'filtered_story_rows': int(len(stories)), 'linked_permnos': len(linked),
          'universe': UNIVERSE, 'universe_permnos': len(permnos), 'story_cache': str(CACHE_DIR)}
path = save_signal(signal, TAG, args.name, params)
print(f'Saved {len(signal):,} rows ({signal["permno"].nunique()} permnos, '
      f'{signal["date"].nunique()} days) to {path} ({time.time() - t0:.0f} s)')


# ## Step 4: Coverage summary

# In[6]:


is_linked = signal['permno'].isin(linked)
print('\nShare of member stock-days with at least one story in the window:')
for L in WINDOWS:
    print(f'  L={L:>2}: {(signal[f"n_{L}"] > 0).mean():.1%}   '
          f'(linked permnos only: {(signal.loc[is_linked, f"n_{L}"] > 0).mean():.1%})')
print('\nsent_L distribution on stock-days with news:')
print(pd.DataFrame({f'sent_{L}': signal.loc[signal[f'n_{L}'] > 0, f'sent_{L}'].describe()
                    for L in WINDOWS}).round(3))
