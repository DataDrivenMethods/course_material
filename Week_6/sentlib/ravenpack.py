"""RavenPack (Dow Jones edition): entity-permno link, story pull with cache, local filters."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .wrds_io import CRSP_LIB

STORY_COLUMNS = ['rp_story_id', 'timestamp_utc', 'rp_entity_id', 'relevance',
                 'event_sentiment_score', 'event_similarity_days', 'group',
                 'category', 'news_type', 'source_name', 'headline']

# Always dropped: chart-pattern stories derived from prices, not news
ALWAYS_EXCLUDED_GROUPS = ('technical-analysis',)

# Low-cardinality text columns, stored as categoricals to save memory
_CATEGORICAL = ['group', 'category', 'news_type', 'source_name']


@dataclass(frozen=True)
class StoryFilter:
    """Story-level filters applied locally to the cached pull."""
    min_relevance: float = 90
    require_event: bool = True            # event_sentiment_score not null
    min_similarity_days: float | None = 1 # novelty; None = no novelty filter
    exclude_groups: tuple[str, ...] = ()  # RavenPack `group` values to drop
    exclude_news_types: tuple[str, ...] = ()
    max_entities_per_story: int | None = None  # drop stories naming more companies than this


def build_link(conn, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """Dated links [rp_entity_id, permno, permco, link_start, link_end].

    RavenPack gives each company dated 9-digit CUSIPs; CRSP stocknames gives each
    permno dated 8-digit historical CUSIPs (stocknames_v2.cusip, the legacy ncusip). A link is valid where both
    date ranges overlap. One entity can map to several permnos (share classes,
    e.g. GOOG and GOOGL); each permno then gets the company's sentiment.
    """
    permnos_str = ', '.join(str(int(p)) for p in permnos)
    crsp_names = conn.raw_sql(f"""
        SELECT DISTINCT permno, permco, cusip AS ncusip, namedt, nameenddt
        FROM {CRSP_LIB}.stocknames_v2
        WHERE permno IN ({permnos_str})
          AND cusip IS NOT NULL
          AND nameenddt >= '{start}' AND namedt <= '{end}'
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
    return link[['rp_entity_id', 'permno', 'permco', 'link_start', 'link_end']]


def entity_hash(entity_ids: list[str]) -> str:
    """First 10 hex chars of sha1 of the sorted entity ids joined by ','."""
    return hashlib.sha1(','.join(sorted(set(entity_ids))).encode()).hexdigest()[:10]


def _pull_year(conn, entity_ids: list[str], year: int, min_relevance: float) -> pd.DataFrame:
    entities_str = ', '.join(f"'{e}'" for e in sorted(set(entity_ids)))
    cols = ', '.join('"group"' if c == 'group' else c for c in STORY_COLUMNS)
    df = conn.raw_sql(f"""
        SELECT {cols}
        FROM ravenpack_dj.rpa_djpr_equities_{year}
        WHERE rp_entity_id IN ({entities_str})
          AND entity_type = 'COMP'
          AND relevance >= {min_relevance}
    """)
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[STORY_COLUMNS].copy()
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    for c in ['relevance', 'event_sentiment_score', 'event_similarity_days']:
        df[c] = df[c].astype('float64')
    for c in _CATEGORICAL:
        df[c] = df[c].astype('category')
    return df


def fetch_stories(conn, entity_ids: list[str], start: str, end: str,
                  cache_dir: Path, cache_min_relevance: float = 75,
                  refresh: bool = False) -> pd.DataFrame:
    """Stories for the entities with start <= timestamp_utc < end + 1 day.

    Pulls STORY_COLUMNS from ravenpack_dj.rpa_djpr_equities_{year} with
    entity_type = 'COMP' and relevance >= cache_min_relevance, no other filters
    (use apply_filter). Each full year is cached as
    cache_dir / f'stories_{year}_{entity_hash}.parquet', so any date range
    inside cached years is served without WRDS. A story with several events
    for one entity has one row per event. If conn is None, only the cache is used.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = entity_hash(entity_ids)
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    frames = []
    for year in range(lo.year, pd.Timestamp(end).year + 1):
        path = cache_dir / f'stories_{year}_{h}.parquet'
        df = pd.read_parquet(path) if path.exists() and not refresh else None
        # A cache built at a higher relevance floor cannot serve a lower one
        if df is not None and df.attrs.get('cache_min_relevance', 0) > cache_min_relevance:
            df = None
        if df is None:
            if conn is None:
                raise FileNotFoundError(f'{path} not usable and no WRDS connection given')
            df = _pull_year(conn, entity_ids, year, cache_min_relevance)
            df.attrs['cache_min_relevance'] = cache_min_relevance
            df.to_parquet(path, index=False)
            print(f'  RavenPack {year}: pulled {len(df):,} rows -> {path}')
        else:
            print(f'  RavenPack {year}: {len(df):,} rows from cache {path.name}')
        df = df[(df['timestamp_utc'] >= lo) & (df['timestamp_utc'] < hi)
                & (df['relevance'] >= cache_min_relevance)]
        frames.append(df)
    stories = pd.concat(frames, ignore_index=True)
    for c in _CATEGORICAL:      # concat of differing categoricals falls back to object
        stories[c] = stories[c].astype('category')
    return stories


def entities_per_story(stories: pd.DataFrame) -> pd.Series:
    """Distinct rp_entity_id per row's rp_story_id, aligned with `stories`."""
    return stories.groupby('rp_story_id', observed=True)['rp_entity_id'].transform('nunique')


def apply_filter(stories: pd.DataFrame, f: StoryFilter) -> pd.DataFrame:
    """Apply a StoryFilter to stories with STORY_COLUMNS.

    max_entities_per_story counts the distinct companies per story in the input
    (i.e. universe companies at the cache relevance floor), before other filters.
    Groups in ALWAYS_EXCLUDED_GROUPS are dropped regardless of the filter.
    """
    keep = stories['relevance'] >= f.min_relevance
    keep &= ~stories['group'].isin(ALWAYS_EXCLUDED_GROUPS)
    if f.require_event:
        keep &= stories['event_sentiment_score'].notna()
    if f.min_similarity_days is not None:
        keep &= stories['event_similarity_days'] >= f.min_similarity_days
    if f.exclude_groups:
        keep &= ~stories['group'].isin(f.exclude_groups)
    if f.exclude_news_types:
        keep &= ~stories['news_type'].isin(f.exclude_news_types)
    if f.max_entities_per_story is not None:
        keep &= entities_per_story(stories) <= f.max_entities_per_story
    return stories[keep.to_numpy()].reset_index(drop=True)
