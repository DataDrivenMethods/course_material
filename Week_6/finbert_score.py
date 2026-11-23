#!/usr/bin/env python
# coding: utf-8

# # FinBERT headline scoring
# Scores news headlines with ProsusAI/finbert (positive / negative / neutral).
# The story score is finbert_score = p_pos - p_neg, in [-1, 1].
#
# Scores are cached per model as parquet, keyed by sha1 of the headline text,
# so each distinct headline is scored once across all runs.
#
# Usage (demo on a few headlines):
#     python finbert_score.py

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Quiet load: the only report item is the harmless unused 'position_ids' buffer
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = 'ProsusAI/finbert'
CACHE_DIR = HERE / 'data' / 'finbert_scores'
CACHE_COLUMNS = ['headline_sha1', 'headline', 'p_pos', 'p_neg', 'p_neu', 'finbert_score', 'model_name']


def load_model(model_name=DEFAULT_MODEL, device=None, fp16=True):
    # fp16 on GPU (checked against fp32: differences ~1e-3), fp32 on CPU
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if fp16 and device.type == 'cuda':
        model = model.half()
    model.to(device).eval()
    return model, tokenizer, device


def _label_index(model):
    # Map 'positive'/'negative'/'neutral' to output columns from the model config
    labels = {i: lab.lower() for i, lab in model.config.id2label.items()}
    idx = {}
    for i, lab in labels.items():
        for key in ('pos', 'neg', 'neu'):
            if lab.startswith(key):
                idx[key] = int(i)
    if len(idx) != 3:
        raise ValueError(f'unexpected labels in model config: {model.config.id2label}')
    return idx


@torch.inference_mode()
def score_texts(texts, model, tokenizer, device, batch_size=256, max_length=128):
    texts = ['' if t is None else str(t) for t in texts]
    idx = _label_index(model)
    probs = np.empty((len(texts), len(model.config.id2label)), dtype=np.float32)
    # Sort by length so each batch pads to a similar length; results go back in input order
    order = np.argsort([len(t) for t in texts], kind='stable')
    for b in range(0, len(texts), batch_size):
        rows = order[b:b + batch_size]
        enc = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt').to(device)
        logits = model(**enc).logits.float()
        probs[rows] = torch.softmax(logits, dim=-1).cpu().numpy()
    out = pd.DataFrame({'p_pos': probs[:, idx['pos']],
                        'p_neg': probs[:, idx['neg']],
                        'p_neu': probs[:, idx['neu']]}).astype('float64')
    out['finbert_score'] = out['p_pos'] - out['p_neg']
    return out


def headline_sha1(text):
    return hashlib.sha1(str(text).encode('utf-8')).hexdigest()


def default_cache_path(model_name=DEFAULT_MODEL):
    # A local (fine-tuned) model keeps its cache in its own folder, so retraining it starts afresh
    if Path(model_name).is_dir():
        return Path(model_name) / 'headline_scores.parquet'
    return CACHE_DIR / f"{model_name.replace('/', '__')}.parquet"


def score_headlines(headlines, cache_path=None, model_name=DEFAULT_MODEL, batch_size=256,
                    max_length=128, refresh=False, model=None, tokenizer=None, device=None):
    # headlines: pd.Series of text. Returns p_pos, p_neg, p_neu, finbert_score on headlines.index.
    cache_path = Path(cache_path) if cache_path is not None else default_cache_path(model_name)
    text = headlines.fillna('').astype(str)
    keys = text.map(headline_sha1)

    # Load cache (only rows from this model); refresh re-scores the requested headlines
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
        cache = cache[cache['model_name'] == model_name]
        if refresh:
            cache = cache[~cache['headline_sha1'].isin(keys)]
    else:
        cache = pd.DataFrame(columns=CACHE_COLUMNS)

    # Distinct headlines not yet scored
    uniq = pd.DataFrame({'headline_sha1': keys, 'headline': text}).drop_duplicates('headline_sha1')
    new = uniq[~uniq['headline_sha1'].isin(cache['headline_sha1'])]
    print(f'FinBERT: {len(text):,} headlines, {len(uniq):,} distinct, '
          f'{len(uniq) - len(new):,} cached, {len(new):,} to score')

    if len(new):
        if model is None:
            model, tokenizer, device = load_model(model_name, device)
        scores = score_texts(new['headline'].tolist(), model, tokenizer, device,
                             batch_size=batch_size, max_length=max_length)
        scored = pd.concat([new.reset_index(drop=True), scores], axis=1)
        scored['model_name'] = model_name
        cache = pd.concat([cache, scored[CACHE_COLUMNS]], ignore_index=True) if len(cache) else scored[CACHE_COLUMNS]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)

    # Align to input index
    lookup = cache.set_index('headline_sha1')[['p_pos', 'p_neg', 'p_neu', 'finbert_score']]
    out = lookup.reindex(keys.values).astype('float64')
    out.index = headlines.index
    return out


if __name__ == '__main__':
    demo = pd.Series([
        'Apple beats earnings estimates and raises full-year guidance',
        'Tesla shares plunge after recall of 100,000 vehicles',
        'Microsoft to hold annual shareholder meeting on December 7',
        'SHAREHOLDER ALERT: Law firm investigates XYZ Corp. on behalf of investors',
        'Apple beats earnings estimates and raises full-year guidance',
    ])
    model, tokenizer, device = load_model()
    print(model.config.id2label, device)
    res = score_texts(demo.tolist(), model, tokenizer, device)
    res.insert(0, 'headline', demo.values)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 80, 'display.max_columns', 10):
        print(res.round(3))
