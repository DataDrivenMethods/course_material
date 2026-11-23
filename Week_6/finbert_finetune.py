#!/usr/bin/env python
# coding: utf-8

# # Fine-tune FinBERT on forward abnormal returns
# Teaches ProsusAI/finbert which headlines are followed by drift. Each filtered
# RavenPack headline (same filters as finbert_signal.py) is labelled with its
# stock's abnormal return over d+1..d+h, where d is the trading day the story
# counts toward (first day whose --cutoff New York time is after it):
#   abnormal = compounded return d+1..d+h  -  equal-weighted mean over universe members on d
# --label-horizon 0 uses the concurrent day-d return instead.
# Labels are split into terciles on the training stories: bottom = negative,
# middle = neutral, top = positive, i.e. FinBERT's three classes, so the
# fine-tuned score p_pos - p_neg is read exactly like the original one.
#
# Leakage: only stories whose label window ends on or before --train-end are
# used. The last --val-frac of training dates is the validation set; training
# stories whose label window reaches past the validation start are dropped.
# Test on dates after --train-end only.
#
# Usage:
#     python finbert_finetune.py --start 2020-01-01 --end 2024-12-31 --update-period month \
#         --train-start 2020-01-01 --train-end 2022-12-31 --exclude-groups stock-prices --name ft_fwd5
#
# --start/--end/--update-period name the universe (as in the other scripts);
# --train-start/--train-end must lie inside it. The model goes to
# data/finbert_models/{name}/; use it with finbert_signal.py --model data/finbert_models/{name}.

# In[1]:


import argparse
import json
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from backtest.evaluate import fwd_matrix
from backtest.returns import get_returns, wide_returns
from finbert_score import DEFAULT_MODEL, _label_index, score_texts
from sentlib import ravenpack as rp
from sentlib.timing import assign_trading_day
from sentlib.wrds_io import DATA_DIR, connect, load_universe, member_days, trading_calendar

parser = argparse.ArgumentParser(description='Fine-tune FinBERT on forward abnormal returns of RavenPack headlines.')
parser.add_argument('--top-n', type=int, default=1000, help='Universe size used in Big_10_Index.py (default 1000)')
parser.add_argument('--start', help='Data window start (default --train-start); any Big_10_Index.py build covering it is used')
parser.add_argument('--end', help='Data window end (default --train-end)')
parser.add_argument('--update-period', '--update_period', dest='update_period', default='year',
                    choices=['year', 'month', 'day'], help='Universe update period used in Big_10_Index.py')
parser.add_argument('--train-start', required=True, help='First story date used for training, inclusive')
parser.add_argument('--train-end', required=True,
                    help='Last date any label return may use, inclusive; test only after this date')
parser.add_argument('--label-horizon', type=int, default=5,
                    help='h: label = abnormal return over d+1..d+h (default 5); 0 = day-d return')
parser.add_argument('--val-frac', type=float, default=0.1, help='Last share of training dates for validation')
parser.add_argument('--single-story-days', action='store_true',
                    help='Train only on stock-days with one story, so each label belongs to one headline')
parser.add_argument('--min-relevance', type=float, default=90, help='Minimum RavenPack relevance (0-100)')
parser.add_argument('--min-similarity-days', type=float, default=1,
                    help='Minimum event_similarity_days (novelty); negative = no novelty filter')
parser.add_argument('--no-require-event', action='store_true', help='Keep stories without a RavenPack event')
parser.add_argument('--exclude-groups', nargs='*', default=[], help='RavenPack group values to drop, e.g. stock-prices')
parser.add_argument('--cutoff', default='15:30', help='New York time after which news counts toward the next trading day')
parser.add_argument('--base-model', default=DEFAULT_MODEL, help=f'Model to start from (default {DEFAULT_MODEL})')
parser.add_argument('--name', required=True, help='Output folder name under data/finbert_models/')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-5, help='Peak learning rate (linear warmup then decay)')
parser.add_argument('--max-length', type=int, default=128, help='Token limit per headline (as in scoring)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

START = pd.Timestamp(args.start or args.train_start).strftime('%Y-%m-%d')
END = pd.Timestamp(args.end or args.train_end).strftime('%Y-%m-%d')
TRAIN_START, TRAIN_END = pd.Timestamp(args.train_start), pd.Timestamp(args.train_end)
H = args.label_horizon
OUT_DIR = DATA_DIR / 'finbert_models' / args.name
if not (pd.Timestamp(START) <= TRAIN_START < TRAIN_END <= pd.Timestamp(END)):
    raise SystemExit(f'training window {TRAIN_START:%Y-%m-%d}..{TRAIN_END:%Y-%m-%d} must lie inside the '
                     f'universe {START}..{END}')
if H < 0:
    raise SystemExit('--label-horizon must be >= 0')
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


# ## Step 1: Stories in the training window, with their trading day and permno

# In[2]:


t0 = time.time()
conn = connect()
# Any Big_10_Index.py build covering START..END, cut to the lists held in START..END
TAG, membership = load_universe(args.top_n, START, END, args.update_period)
permnos = sorted(membership['permno'].unique().tolist())
calendar = trading_calendar(conn, START, END)
link = rp.build_link(conn, permnos, START, END)

pull_start = (TRAIN_START - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
stories = rp.fetch_stories(conn, link['rp_entity_id'].unique().tolist(), pull_start,
                           TRAIN_END.strftime('%Y-%m-%d'), cache_dir=DATA_DIR / 'rp_stories' / TAG)
min_sim = None if args.min_similarity_days < 0 else args.min_similarity_days
story_filter = rp.StoryFilter(min_relevance=args.min_relevance, require_event=not args.no_require_event,
                              min_similarity_days=min_sim if not args.no_require_event else None,
                              exclude_groups=tuple(args.exclude_groups))
stories = rp.apply_filter(stories, story_filter)
if args.no_require_event and min_sim is not None:      # novelty for event stories only, as in finbert_signal.py
    sim = stories['event_similarity_days']
    stories = stories[(sim.isna() | (sim >= min_sim)).to_numpy()]
stories = stories.drop_duplicates(['rp_story_id', 'rp_entity_id']).reset_index(drop=True)
stories['date'] = assign_trading_day(stories['timestamp_utc'], calendar, args.cutoff)
stories = stories[(stories['date'] >= TRAIN_START) & (stories['date'] <= TRAIN_END)]

# Entity -> permno where the link is valid on d, then keep member stock-days
s = stories[['rp_story_id', 'rp_entity_id', 'date', 'headline']].merge(link, on='rp_entity_id')
s = s[(s['date'] >= s['link_start']) & (s['date'] <= s['link_end'])]
members = member_days(membership, calendar, TRAIN_START, TRAIN_END)
s = s.merge(members[['date', 'permno']], on=['date', 'permno'])
print(f'Universe {TAG}; story-entity rows {TRAIN_START:%Y-%m-%d}..{TRAIN_END:%Y-%m-%d} after filters: '
      f'{len(stories):,}, on member stock-days: {len(s.drop_duplicates(["rp_story_id", "rp_entity_id"])):,}')


# ## Step 2: Labels = forward abnormal return

# In[3]:


returns = get_returns(conn, permnos, calendar[0].strftime('%Y-%m-%d'), calendar[-1].strftime('%Y-%m-%d'))
conn.close()
R, _, alive = wide_returns(returns)
F = fwd_matrix(R, alive, H) if H > 0 else R.where(alive)
# Abnormal = minus the equal-weighted mean over members on d with a return
is_member = (members.assign(m=1).pivot(index='date', columns='permno', values='m')
                    .reindex(index=F.index, columns=F.columns).notna())
F = F.where(is_member)
A = F.sub(F.mean(axis=1), axis=0)
# Label window d+1..d+h must end on or before TRAIN_END
pos = pd.Series(np.arange(len(A)), index=A.index)
last_ok = pos[A.index <= TRAIN_END].max() - H
A = A[pos <= last_ok]

lab = A.stack().rename('label').reset_index()
lab.columns = ['date', 'permno', 'label']
s = s.merge(lab, on=['date', 'permno'])
# One row per story-entity (an entity with two share classes averages their labels)
s = (s.groupby(['rp_story_id', 'rp_entity_id', 'date'], as_index=False, observed=True)
      .agg(headline=('headline', 'first'), label=('label', 'mean')))
if args.single_story_days:
    one = s.groupby(['rp_entity_id', 'date'], observed=True)['rp_story_id'].transform('count') == 1
    s = s[one.to_numpy()]
s = s.reset_index(drop=True)

# Validation = last val_frac of dates; purge training labels that reach into it
dates = np.sort(s['date'].unique())
val_start = pd.Timestamp(dates[int(len(dates) * (1 - args.val_frac))])
is_val = s['date'] >= val_start
is_train = s['date'].map(pos).to_numpy() + H <= pos[val_start]
train, val = s[is_train].reset_index(drop=True), s[is_val.to_numpy()].reset_index(drop=True)

# Terciles from training labels -> negative / neutral / positive
q_lo, q_hi = train['label'].quantile([1 / 3, 2 / 3]).to_numpy()
CLASSES = np.array(['neg', 'neu', 'pos'])
for df in (train, val):
    df['cls'] = np.where(df['label'] < q_lo, 0, np.where(df['label'] > q_hi, 2, 1))

lbl = f'd+1..d+{H}' if H > 0 else 'day d'
print(f'Label: abnormal return over {lbl}; terciles at {q_lo:+.4f} / {q_hi:+.4f}')
print(f'Train {train["date"].min():%Y-%m-%d}..{train["date"].max():%Y-%m-%d}: {len(train):,} stories   '
      f'Validation {val["date"].min():%Y-%m-%d}..{val["date"].max():%Y-%m-%d}: {len(val):,} stories   '
      f'({time.time() - t0:.0f} s)')


# ## Step 3: Fine-tune

# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print('WARNING: no GPU, training on the CPU will be very slow')
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForSequenceClassification.from_pretrained(args.base_model).to(device)
idx = _label_index(model)
to_head = np.array([idx['neg'], idx['neu'], idx['pos']])     # our class -> model output column

enc = tokenizer(train['headline'].fillna('').astype(str).tolist(), truncation=True, max_length=args.max_length)
y_train = torch.tensor(to_head[train['cls'].to_numpy()])


def batches(order):
    for b in range(0, len(order), args.batch_size):
        rows = order[b:b + args.batch_size]
        x = tokenizer.pad({'input_ids': [enc['input_ids'][i] for i in rows],
                           'attention_mask': [enc['attention_mask'][i] for i in rows]}, return_tensors='pt')
        yield {k: v.to(device) for k, v in x.items()}, y_train[rows].to(device)


def evaluate(model, df):
    """Validation loss, accuracy and Spearman(score, abnormal return), story level and daily."""
    model.eval()
    p = score_texts(df['headline'].tolist(), model, tokenizer, device, max_length=args.max_length)
    probs = np.clip(p[['p_neg', 'p_neu', 'p_pos']].to_numpy(), 1e-12, 1)
    cls = df['cls'].to_numpy()
    sc = p['finbert_score'].to_numpy()
    daily = (pd.DataFrame({'date': df['date'].to_numpy(), 'x': sc, 'y': df['label'].to_numpy()})
               .groupby('date').filter(lambda g: len(g) >= 20)
               .groupby('date')[['x', 'y']].apply(lambda g: g['x'].corr(g['y'], method='spearman')))
    return {'loss': float(-np.log(probs[np.arange(len(cls)), cls]).mean()),
            'accuracy': float((probs.argmax(1) == cls).mean()),
            'spearman': float(pd.Series(sc).corr(df['label'].reset_index(drop=True), method='spearman')),
            'daily_ic': float(daily.mean())}


def show(epoch, m):
    print(f'  {epoch:<8}{m["loss"]:>8.4f}{m["accuracy"]:>10.1%}{m["spearman"]:>11.4f}{m["daily_ic"]:>11.4f}')


steps = args.epochs * int(np.ceil(len(train) / args.batch_size))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')

print(f'\nValidation ({len(val):,} stories; chance accuracy about 33%; '
      'Spearman = score vs abnormal return; daily IC = mean daily Spearman)')
print(f'  {"epoch":<8}{"loss":>8}{"accuracy":>10}{"Spearman":>11}{"daily IC":>11}')
history = [{'epoch': 0, **evaluate(model, val)}]
show('0 (base)', history[0])
best = history[0]
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)          # also drops the old model's score cache
for epoch in range(1, args.epochs + 1):
    model.train()
    for x, y in batches(rng.permutation(len(train))):
        with torch.autocast(**amp):
            loss = torch.nn.functional.cross_entropy(model(**x).logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    history.append({'epoch': epoch, **evaluate(model, val)})
    show(epoch, history[-1])
    # Keep the epoch with the best validation daily IC
    if history[-1]['daily_ic'] > best['daily_ic'] or best['epoch'] == 0:
        best = history[-1]
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)


# ## Step 4: Save settings

# In[5]:


meta = {'created': datetime.now().isoformat(timespec='seconds'), 'base_model': args.base_model,
        'universe': TAG, 'train_start': f'{TRAIN_START:%Y-%m-%d}', 'train_end': f'{TRAIN_END:%Y-%m-%d}',
        'validation_start': f'{val_start:%Y-%m-%d}', 'label_horizon': H,
        'label': f'abnormal return over {lbl} (minus equal-weighted universe mean)',
        'tercile_cuts': [float(q_lo), float(q_hi)], 'single_story_days': args.single_story_days,
        'filter': story_filter.__dict__ | {'exclude_groups': list(story_filter.exclude_groups),
                                           'exclude_news_types': list(story_filter.exclude_news_types)},
        'cutoff': args.cutoff, 'train_stories': len(train), 'val_stories': len(val),
        'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'max_length': args.max_length,
        'seed': args.seed, 'best_epoch': best['epoch'], 'history': history}
with open(OUT_DIR / 'training.json', 'w') as fh:
    json.dump(meta, fh, indent=2, default=str)
print(f'\nSaved epoch {best["epoch"]} to {OUT_DIR} ({time.time() - t0:.0f} s). '
      f'Test only on signal dates after {TRAIN_END:%Y-%m-%d}.')
