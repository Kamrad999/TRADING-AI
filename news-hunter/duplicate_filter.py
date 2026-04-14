"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          MONSTER TRADING AI                                     ║
║                  DUPLICATE INTELLIGENCE LAYER  ·  v1.0  ELITE                   ║
║                                                                                  ║
║  duplicate_filter.py                                                             ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Institutional-grade duplicate clustering and confidence amplification.          ║
║                                                                                  ║
║  Pipeline position:                                                              ║
║      news_engine.py                                                              ║
║        → duplicate_filter.py          ← YOU ARE HERE                            ║
║          → fake_news_validator.py                                                ║
║            → signal_engine.py                                                    ║
║              → alert_router.py                                                   ║
║                                                                                  ║
║  MISSION                                                                         ║
║  ───────                                                                         ║
║  This is NOT a simple deduplicator.  It is a hedge-fund-grade news              ║
║  clustering engine that treats duplicate articles as a SIGNAL, not noise.       ║
║                                                                                  ║
║  When Reuters, Bloomberg, and CNBC all break the same CPI story within          ║
║  minutes, that is not redundancy — it is institutional confirmation.             ║
║  This module detects that confirmation and amplifies the surviving article's     ║
║  credibility accordingly.                                                        ║
║                                                                                  ║
║  DETECTION STRATEGY  (two-pass)                                                  ║
║  ──────────────────────────────                                                  ║
║  PASS 1  —  Exact Duplicate Detection   O(n)                                    ║
║    · Fingerprint via dedupe_hash (pre-computed by news_engine.py)               ║
║    · Fallback fingerprint via normalized link                                    ║
║    · Second fallback via normalized title                                        ║
║    · Groups articles that are byte-for-byte identical                            ║
║                                                                                  ║
║  PASS 2  —  Fuzzy Duplicate Clustering   O(k²) on survivors                    ║
║    · Applies difflib.SequenceMatcher on normalized titles                        ║
║    · Configurable similarity threshold (default 0.82)                            ║
║    · Clusters "same story, multiple outlets" — the most valuable signal         ║
║    · Skips pairs already merged in Pass 1                                        ║
║                                                                                  ║
║  CLUSTER INTELLIGENCE  (enrichment fields added per surviving article)           ║
║  ──────────────────────────────────────────────────────────────────────          ║
║  duplicate_count           int     Total articles in cluster (incl. self)        ║
║  source_cluster            list    All outlets that reported this story          ║
║  credibility_boost         float   Confidence multiplier from multi-source conf  ║
║  cluster_size_score        float   Normalized 0.0–1.0 cluster strength           ║
║  is_multi_source_confirm   bool    True if 2+ distinct sources cover the story   ║
║                                                                                  ║
║  PRESERVATION RULES  (which article survives from each cluster)                  ║
║  ──────────────────────────────────────────────────────────────                  ║
║  1. Highest priority source wins (priority 1 > priority 5)                       ║
║  2. Tie-break: earliest published_time                                           ║
║  3. Summary: longest summary from any member is preserved                        ║
║  4. Tags: union of all unique tags across the cluster                            ║
║  5. Sources: all outlet names merged into source_cluster                         ║
║                                                                                  ║
║  ARCHITECTURE PHILOSOPHY                                                         ║
║  ─────────────────────────                                                       ║
║  Pure functional pipeline.  Every function has ONE job.                          ║
║  Every function is independently testable and reusable.                          ║
║  Zero crashes from malformed input — full defensive coverage.                    ║
║                                                                                  ║
║  LAYER MAP                                                                       ║
║  ──────────                                                                      ║
║  1. PUBLIC API              — filter_duplicates() entry point                    ║
║  2. EXACT DEDUP PASS        — hash + link + title fingerprinting                 ║
║  3. FUZZY CLUSTER PASS      — difflib title similarity clustering                ║
║  4. CLUSTER RESOLUTION      — representative selection + metadata merge          ║
║  5. INTELLIGENCE ENRICHMENT — credibility boost, cluster scoring                 ║
║  6. NORMALIZATION HELPERS   — pure string-cleaning utilities                     ║
║  7. SAFETY HELPERS          — defensive wrappers, parse utilities                ║
║  8. METRICS & OBSERVABILITY — structured run reporting                           ║
║                                                                                  ║
║  PLUGIN BOUNDARY MAP                                                             ║
║  ─────────────────────                                                           ║
║  Upstream   : news_engine.py         → produces normalized article dicts        ║
║  Downstream : fake_news_validator.py → consumes enriched, de-duped articles     ║
║  Future     : sqlite_store.py        → cluster audit trail persistence           ║
║               signal_engine.py       → consumes credibility_boost field         ║
║               alert_router.py        → consumes is_multi_source_confirmation     ║
║                                                                                  ║
║  PERFORMANCE TARGETS                                                             ║
║  ─────────────────────                                                           ║
║  · 1000 articles processed in < 200ms                                           ║
║  · O(n) exact pass — hash-bucket grouping                                       ║
║  · O(k²) fuzzy pass on k survivors — k << n after exact pass                   ║
║  · Zero memory copies of article content — mutation is in-place on copies       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY IMPORTS  (zero third-party dependencies)
#
#  difflib   → SequenceMatcher for fuzzy title similarity scoring
#  datetime  → UTC timestamp parsing for chronological tie-breaking
#  time      → perf_counter for latency measurement
#  re        → title normalization (strip punctuation, collapse whitespace)
#  typing    → type hints for IDE support and future mypy enforcement
# ─────────────────────────────────────────────────────────────────────────────
import difflib
import re
import time
from datetime import datetime, timezone
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#
#  MODULE CONFIGURATION
#
#  All tunable parameters live here.  Never scatter magic numbers through
#  function bodies.  Future: load from config.yaml or environment variables.
#
# ══════════════════════════════════════════════════════════════════════════════

# Fuzzy similarity threshold for title clustering (0.0 – 1.0).
# 0.82 means titles must share ~82% of their character sequence to be clustered.
# Lower  → more aggressive merging (risk: over-collapsing distinct stories)
# Higher → more conservative merging (risk: missing same-story variants)
FUZZY_SIMILARITY_THRESHOLD: float = 0.82

# Minimum title token length to qualify for fuzzy comparison.
# Titles shorter than this (e.g. "[No Title]") are excluded from fuzzy pass.
MIN_TITLE_LENGTH_FOR_FUZZY: int = 12

# Maximum credibility boost multiplier from multi-source confirmation.
# A cluster of 5 sources will cap at this value.
MAX_CREDIBILITY_BOOST: float = 2.5

# Credibility boost increment per additional confirming source.
# Base boost = 1.0 (single source).  Each additional source adds this value.
CREDIBILITY_BOOST_PER_SOURCE: float = 0.30

# Cluster size above which cluster_size_score saturates to 1.0.
# Prevents unbounded scoring for extremely large clusters.
CLUSTER_SIZE_SATURATION: int = 8

# Verbosity levels (mirrors news_engine.py convention).
# 0 = silent  |  1 = summary only  |  2 = per-cluster  |  3 = full debug
VERBOSITY: int = 2

# Sentinel value used when dedupe_hash is absent — marks "no hash available".
_NO_HASH_SENTINEL: str = "__NO_HASH__"


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 1  —  PUBLIC API
#
#  Single entry point for the entire module.
#  Orchestrates the two-pass pipeline and returns enriched survivors.
#
# ══════════════════════════════════════════════════════════════════════════════

def filter_duplicates(articles: list[dict]) -> list[dict]:
    """
    PUBLIC ENTRY POINT
    ──────────────────
    Runs the full two-pass duplicate detection and cluster enrichment pipeline.

    Designed to be dropped directly into the post-pipeline plugin hook in
    news_engine.py:

        articles = filter_duplicates(articles)

    Pipeline stages:
        1. Validate and sanitize input list
        2. Pass 1 — Exact duplicate clustering (O(n) hash-bucket grouping)
        3. Pass 2 — Fuzzy duplicate clustering (O(k²) title similarity)
        4. Cluster resolution — select representative + merge metadata
        5. Intelligence enrichment — credibility boost + cluster scoring
        6. Metrics + observability report

    Parameters
    ──────────
    articles : list[dict]
        Normalized article dicts as produced by news_engine.normalize_entries().
        Malformed dicts, None entries, and empty lists are all handled safely.

    Returns
    ───────
    list[dict]
        De-duplicated articles, each enriched with cluster intelligence fields:
            duplicate_count             int
            source_cluster              list[str]
            credibility_boost           float
            cluster_size_score          float
            is_multi_source_confirmation bool
    """
    wall_start = time.perf_counter()

    _log(1, "DEDUP", f"Starting duplicate filter · {len(articles)} input articles")

    # ── Guard: empty or invalid input ────────────────────────────────────────
    if not articles or not isinstance(articles, list):
        _log(1, "DEDUP", "No articles to filter — returning empty list")
        return []

    # Sanitize: discard any non-dict entries without crashing.
    clean_articles = [a for a in articles if isinstance(a, dict)]
    discarded      = len(articles) - len(clean_articles)
    if discarded:
        _log(2, "WARN", f"Discarded {discarded} non-dict entries from input")

    if not clean_articles:
        return []

    # ── PASS 1: Exact duplicate clustering ───────────────────────────────────
    exact_clusters = _cluster_exact_duplicates(clean_articles)
    _log(2, "PASS1", f"Exact clustering: {len(clean_articles)} articles → "
                      f"{len(exact_clusters)} clusters")

    # Resolve each exact cluster to its best representative.
    pass1_survivors = [_select_cluster_representative(cluster)
                       for cluster in exact_clusters]

    # ── PASS 2: Fuzzy duplicate clustering ───────────────────────────────────
    # Operates ONLY on survivors from Pass 1 — k << n after exact dedup.
    fuzzy_clusters = _cluster_fuzzy_duplicates(
        pass1_survivors,
        threshold = FUZZY_SIMILARITY_THRESHOLD,
    )
    _log(2, "PASS2", f"Fuzzy clustering: {len(pass1_survivors)} survivors → "
                      f"{len(fuzzy_clusters)} final clusters")

    # ── Resolve + enrich each final cluster ──────────────────────────────────
    survivors = []
    for cluster in fuzzy_clusters:
        representative = _select_cluster_representative(cluster)
        enriched       = _merge_cluster_metadata(representative, cluster)
        scored         = _score_cluster_strength(enriched)
        survivors.append(scored)

    # ── PLUGIN HOOK: post_filter ──────────────────────────────────────────────
    # Future: sqlite_store.log_clusters(fuzzy_clusters, run_id)
    #         audit_trail.record(original=clean_articles, survivors=survivors)

    # ── Metrics report ────────────────────────────────────────────────────────
    wall_elapsed = round(time.perf_counter() - wall_start, 4)
    _log_filter_summary(
        input_count    = len(clean_articles),
        output_count   = len(survivors),
        cluster_count  = len(fuzzy_clusters),
        elapsed        = wall_elapsed,
        survivors      = survivors,
    )

    return survivors


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 2  —  EXACT DUPLICATE PASS
#
#  O(n) hash-bucket grouping.  Three-tier fingerprinting strategy:
#
#    TIER 1 — dedupe_hash (pre-computed MD5 of title+link by news_engine.py)
#             Most reliable.  Two articles sharing a hash are byte-identical.
#
#    TIER 2 — normalized link
#             Catches the same article fetched from two slightly different
#             URL variants (UTM params, trailing slashes, etc.)
#             e.g. "https://reuters.com/story?utm_source=a"
#              and "https://reuters.com/story?utm_source=b"
#
#    TIER 3 — normalized title
#             Last resort.  Catches identical headlines from feeds that
#             don't provide a canonical link.
#
#  The three tiers are evaluated in order.  Once a fingerprint is found,
#  the article is bucketed under that key.  All articles in the same bucket
#  are exact duplicates by that tier's definition.
#
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_exact_duplicates(articles: list[dict]) -> list[list[dict]]:
    """
    Groups articles into exact-duplicate clusters using three-tier fingerprinting.

    Each bucket in the hash map is one cluster.  Articles with no usable
    fingerprint at any tier are placed in their own singleton cluster —
    they are never silently discarded.

    Parameters
    ──────────
    articles : list[dict]
        Sanitized article dicts (guaranteed to be dicts by caller).

    Returns
    ───────
    list[list[dict]]
        Each inner list is one exact-duplicate cluster.
        Singleton clusters contain articles with no duplicates detected.
    """
    buckets: dict[str, list[dict]] = {}

    for article in articles:
        key = _exact_fingerprint(article)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(article)

    return list(buckets.values())


def _exact_fingerprint(article: dict) -> str:
    """
    Produces a three-tier deduplication key for one article.

    Tier 1: dedupe_hash  (trusted if non-empty)
    Tier 2: normalized link  (strips UTM params + trailing slashes)
    Tier 3: normalized title  (lowercase, punctuation removed)
    Fallback: unique sentinel built from object id — ensures no accidental
              merging of articles with no usable identity field at all.

    Always returns a non-empty string.
    """
    # Tier 1: pre-computed hash from news_engine.py
    dedupe_hash = _safe_str(article.get("dedupe_hash"))
    if dedupe_hash and dedupe_hash != _NO_HASH_SENTINEL:
        return f"hash::{dedupe_hash}"

    # Tier 2: normalized link
    link = _normalize_link(_safe_str(article.get("link")))
    if link:
        return f"link::{link}"

    # Tier 3: normalized title
    title = _normalize_title(_safe_str(article.get("title")))
    if title and title != "[no title]":
        return f"title::{title}"

    # Fallback: guaranteed-unique key so this article forms its own cluster.
    return f"uid::{id(article)}"


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 3  —  FUZZY DUPLICATE CLUSTERING PASS
#
#  Operates on the survivors of the exact pass (k articles where k << n).
#  Groups articles whose normalized titles are "close enough" by SequenceMatcher.
#
#  ALGORITHM: greedy single-pass union-find variant.
#
#  Why not a full O(k²) comparison matrix?
#    For k ≤ 200 survivors (typical after exact pass), O(k²) = 40,000 ops.
#    SequenceMatcher on ~100-char strings runs in ~0.01ms each.
#    Total: ~400ms worst-case.  Acceptable for a batch pipeline.
#    For k > 500, consider LSH (Locality-Sensitive Hashing) as a future upgrade.
#
#  PLUGIN HOOK: future lsh_cluster.py can replace this layer as a drop-in
#  replacement for the return value of _cluster_fuzzy_duplicates().
#
#  CLUSTERING APPROACH: greedy seed assignment.
#    - Iterate survivors in order.
#    - Each article is tested against the title of each existing cluster seed.
#    - If similarity >= threshold: joins that cluster.
#    - If no match: starts a new cluster (becomes the new seed).
#    - An article can only join ONE cluster (first match wins).
#
#  This is O(k × c) where c = number of clusters so far.
#  In practice c grows much slower than k, making this near-linear.
#
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_fuzzy_duplicates(
    articles  : list[dict],
    threshold : float = FUZZY_SIMILARITY_THRESHOLD,
) -> list[list[dict]]:
    """
    Groups articles into fuzzy-duplicate clusters by title similarity.

    Uses difflib.SequenceMatcher for standard-library-only compliance.
    Each article joins the first cluster whose seed title is >= threshold
    similar to its own normalized title.  Articles with titles too short
    to compare meaningfully are treated as singletons.

    Parameters
    ──────────
    articles  : list[dict]   Survivors from the exact duplicate pass.
    threshold : float        Similarity threshold (0.0–1.0).  Default 0.82.

    Returns
    ───────
    list[list[dict]]
        Each inner list is one fuzzy cluster.
        Singletons are clusters of length 1.
    """
    # clusters     → the growing list of clusters (each is a list of dicts)
    # seed_titles  → the normalized title of the FIRST article in each cluster
    #                (the seed is fixed; only the seed is compared against,
    #                 not all members — this keeps complexity bounded)
    clusters    : list[list[dict]] = []
    seed_titles : list[str]        = []

    for article in articles:
        raw_title = _safe_str(article.get("title"))
        norm      = _normalize_title(raw_title)

        # Short or empty titles cannot be reliably compared — treat as singleton.
        if len(norm) < MIN_TITLE_LENGTH_FOR_FUZZY:
            clusters.append([article])
            seed_titles.append(norm)
            continue

        # Find the first cluster whose seed is similar enough.
        matched_cluster_index = _find_fuzzy_match(norm, seed_titles, threshold)

        if matched_cluster_index is not None:
            clusters[matched_cluster_index].append(article)
            _log(3, "FUZZY", f"Clustered '{raw_title[:60]}...' "
                              f"into cluster {matched_cluster_index}")
        else:
            # No match — this article seeds a new cluster.
            clusters.append([article])
            seed_titles.append(norm)

    return clusters


def _find_fuzzy_match(
    norm_title  : str,
    seed_titles : list[str],
    threshold   : float,
) -> Optional[int]:
    """
    Scans existing cluster seeds to find the best fuzzy match.

    Returns the index of the best-matching cluster if similarity >= threshold,
    or None if no cluster is close enough.

    WHY best match and not first match:
        Greedy first-match can cause "chain drift" — article B clusters with A
        even though C is a much better match for B.  Taking the argmax over all
        seeds prevents this and produces tighter, semantically purer clusters.

    Parameters
    ──────────
    norm_title  : str        Normalized title of the article being tested.
    seed_titles : list[str]  Normalized titles of all current cluster seeds.
    threshold   : float      Minimum similarity to qualify as a match.

    Returns
    ───────
    int  → index of the best-matching cluster
    None → no cluster met the threshold
    """
    best_score = 0.0
    best_index = None

    for idx, seed in enumerate(seed_titles):
        if not seed:
            continue
        score = _safe_similarity(norm_title, seed)
        if score >= threshold and score > best_score:
            best_score = score
            best_index = idx

    return best_index


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 4  —  CLUSTER RESOLUTION
#
#  Given a cluster of equivalent articles, selects the single best
#  representative and merges metadata from all members into it.
#
#  RESOLUTION RULES  (applied in strict priority order)
#  ─────────────────
#  1. Lowest priority NUMBER wins (priority 1 = most institutional weight)
#  2. Tie-break: earliest published_time (first-mover advantage)
#  3. Summary: longest non-None summary from any cluster member
#  4. Tags: de-duplicated union of all tags across all members
#  5. source_cluster: de-duplicated list of all source names in the cluster
#
#  The representative is a COPY — original article dicts are never mutated.
#  This is critical for pipeline safety: upstream modules hold references
#  to the original dicts and must not see unexpected field changes.
#
# ══════════════════════════════════════════════════════════════════════════════

def _select_cluster_representative(cluster: list[dict]) -> dict:
    """
    Selects and returns the single best article from a cluster.

    Selection criteria (in order):
        1. Lowest priority value (1 = highest institutional weight)
        2. Earliest published_time as tie-break
        3. Longest summary as final tie-break

    Always returns a shallow copy of the winning article dict.
    Never mutates the original.

    Parameters
    ──────────
    cluster : list[dict]
        One group of duplicate articles (guaranteed non-empty by caller).

    Returns
    ───────
    dict  → shallow copy of the best representative article
    """
    if len(cluster) == 1:
        return dict(cluster[0])  # Shallow copy — do not mutate original.

    def sort_key(article: dict):
        priority  = _safe_int(article.get("priority"), fallback=3)
        pub_time  = _safe_parse_time(article.get("published_time"))
        # Negate time for ascending sort (earlier = better = lower epoch value)
        time_sort = pub_time.timestamp() if pub_time else float("inf")
        # Negate summary length (longer = better = we want descending)
        summary   = _safe_str(article.get("summary"))
        summ_sort = -len(summary)
        return (priority, time_sort, summ_sort)

    winner = sorted(cluster, key=sort_key)[0]
    return dict(winner)


def _merge_cluster_metadata(representative: dict, cluster: list[dict]) -> dict:
    """
    Merges the best metadata from all cluster members into the representative.

    Mutates the representative dict IN PLACE (it is already a copy from
    _select_cluster_representative — the original is safe).

    Merges:
        · summary      → longest non-empty summary from any member
        · tags         → de-duplicated union of all tags
        · source_cluster → de-duplicated list of all source names

    Parameters
    ──────────
    representative : dict        Shallow copy of the winning article.
    cluster        : list[dict]  All articles in this cluster (including rep).

    Returns
    ───────
    dict  → the enriched representative (same object, mutated)
    """
    best_summary  = _safe_str(representative.get("summary"))
    all_tags      : list[str] = []
    all_sources   : list[str] = []

    for article in cluster:
        # Summary: keep longest non-empty one.
        candidate_summary = _safe_str(article.get("summary"))
        if len(candidate_summary) > len(best_summary):
            best_summary = candidate_summary

        # Tags: accumulate all — deduplicate at the end.
        raw_tags = article.get("tags")
        if isinstance(raw_tags, (list, tuple, set)):
            for tag in raw_tags:
                tag_str = _safe_str(tag)
                if tag_str:
                    all_tags.append(tag_str)

        # Sources: accumulate all outlet names.
        source_name = _safe_str(article.get("source"))
        if source_name:
            all_sources.append(source_name)

    # Apply merged values.
    representative["summary"]        = best_summary or None
    representative["tags"]           = _deduplicate_ordered(all_tags)
    representative["source_cluster"] = _deduplicate_ordered(all_sources)

    return representative


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 5  —  INTELLIGENCE ENRICHMENT
#
#  This is where the module earns its "confidence amplifier" designation.
#
#  For each cluster survivor, four intelligence fields are computed and
#  injected.  These fields are consumed downstream by:
#
#    signal_engine.py     → uses credibility_boost as a scoring multiplier
#    alert_router.py      → uses is_multi_source_confirmation for dispatch priority
#    fake_news_validator  → uses source_cluster depth as credibility evidence
#    sqlite_store.py      → persists cluster audit trail for historical analysis
#
#  CREDIBILITY BOOST FORMULA
#  ─────────────────────────
#  base:     1.0  (single-source article, no confirmation)
#  per extra source: +CREDIBILITY_BOOST_PER_SOURCE (default 0.30)
#  cap:      MAX_CREDIBILITY_BOOST (default 2.5)
#
#  Example:
#    Reuters alone               → boost 1.0
#    Reuters + Bloomberg         → boost 1.3
#    Reuters + Bloomberg + CNBC  → boost 1.6
#    5-source cluster            → boost 2.2
#    8-source cluster            → boost 2.5 (capped)
#
#  CLUSTER SIZE SCORE FORMULA
#  ──────────────────────────
#  Normalized 0.0–1.0.  Saturates at CLUSTER_SIZE_SATURATION (default 8).
#  cluster_size_score = min(cluster_size, saturation) / saturation
#  Single article → 0.125.  8+ articles → 1.0.
#
# ══════════════════════════════════════════════════════════════════════════════

def _score_cluster_strength(article: dict) -> dict:
    """
    Computes and injects intelligence enrichment fields into a representative.

    Fields added:
        duplicate_count             → total articles in the cluster
        source_cluster              → already set by _merge_cluster_metadata
        credibility_boost           → multi-source confirmation multiplier
        cluster_size_score          → normalized 0.0–1.0 cluster size
        is_multi_source_confirmation → True if 2+ distinct sources reported

    Mutates the article dict IN PLACE (it is a copy — originals are safe).

    Parameters
    ──────────
    article : dict
        Representative article with source_cluster already merged in.

    Returns
    ───────
    dict  → the same article dict with intelligence fields added
    """
    source_cluster = article.get("source_cluster", [])
    if not isinstance(source_cluster, list):
        source_cluster = []

    cluster_size = max(len(source_cluster), 1)  # At minimum 1 (itself)
    unique_sources = len(set(source_cluster)) if source_cluster else 1

    # ── credibility_boost ────────────────────────────────────────────────────
    boost = 1.0 + (unique_sources - 1) * CREDIBILITY_BOOST_PER_SOURCE
    boost = round(min(boost, MAX_CREDIBILITY_BOOST), 4)

    # ── cluster_size_score ───────────────────────────────────────────────────
    size_score = round(
        min(cluster_size, CLUSTER_SIZE_SATURATION) / CLUSTER_SIZE_SATURATION,
        4,
    )

    # ── is_multi_source_confirmation ─────────────────────────────────────────
    is_multi = unique_sources >= 2

    # ── duplicate_count ──────────────────────────────────────────────────────
    # Represents total articles seen (incl. self) that were clustered together.
    duplicate_count = cluster_size

    article["duplicate_count"]              = duplicate_count
    article["credibility_boost"]            = boost
    article["cluster_size_score"]           = size_score
    article["is_multi_source_confirmation"] = is_multi

    _log(3, "ENRICH",
         f"[{article.get('source', '?')}] "
         f"cluster={cluster_size} sources={unique_sources} "
         f"boost={boost} multi={is_multi}")

    return article


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 6  —  NORMALIZATION HELPERS
#
#  Pure string-cleaning utilities.  No side effects.  No I/O.
#  All functions are independently testable.
#
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_title(raw: str) -> str:
    """
    Produces a canonical, comparison-ready version of an article title.

    Transformations applied (in order):
        1. Lowercase
        2. Strip leading/trailing whitespace
        3. Remove all punctuation (replaced with space)
        4. Collapse multiple spaces into one
        5. Strip again

    This ensures that:
        "Fed Raises Rates 0.25% — Markets Respond"
        "Fed raises rates 0.25   markets respond"
        "FED RAISES RATES 0.25% MARKETS RESPOND"
    all produce the same comparison key, and therefore score 1.0 similarity.

    Parameters
    ──────────
    raw : str   Raw title string (may be empty or "[No Title]").

    Returns
    ───────
    str  → normalized comparison string (may be empty)
    """
    if not raw:
        return ""
    lowered    = raw.lower().strip()
    no_punct   = re.sub(r"[^\w\s]", " ", lowered)
    collapsed  = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def _normalize_link(raw: str) -> str:
    """
    Strips UTM parameters, trailing slashes, and common tracking suffixes
    from a URL to produce a canonical link key for exact deduplication.

    Handles the common case where the same Reuters article appears as:
        https://reuters.com/story/abc123?utm_source=google
        https://reuters.com/story/abc123?utm_medium=email
        https://reuters.com/story/abc123/

    All three should deduplicate to the same key.

    Parameters
    ──────────
    raw : str   Raw link string (may be empty or malformed).

    Returns
    ───────
    str  → normalized link (may be empty)
    """
    if not raw:
        return ""
    # Strip query string (UTM params and other trackers).
    link = raw.split("?")[0]
    # Strip fragment identifiers.
    link = link.split("#")[0]
    # Remove trailing slashes.
    link = link.rstrip("/").lower().strip()
    return link


def _deduplicate_ordered(items: list[str]) -> list[str]:
    """
    Removes duplicates from a list while preserving insertion order.

    Uses a set for O(1) membership testing at O(n) total cost.
    Order is preserved — the first occurrence of each value is kept.

    Parameters
    ──────────
    items : list[str]   Input list (may contain duplicates and empty strings).

    Returns
    ───────
    list[str]  → deduplicated, order-preserved list (empty strings excluded)
    """
    seen   : set[str]  = set()
    result : list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 7  —  SAFETY HELPERS
#
#  Defensive wrappers used throughout the module.
#  Every function here has a guaranteed return type — never raises.
#
# ══════════════════════════════════════════════════════════════════════════════

def _safe_str(value, fallback: str = "") -> str:
    """
    Safely coerces any value to a string.

    Handles None, non-string types, and objects with broken __str__.
    Always returns a string — never raises.
    """
    if value is None:
        return fallback
    try:
        return str(value).strip()
    except Exception:
        return fallback


def _safe_int(value, fallback: int = 3) -> int:
    """
    Safely converts a value to int.

    Handles None, float strings ("3.0"), negatives, and garbage input.
    Routes through float() first to handle "3.0" → 3 correctly.
    Always returns an int — never raises.
    """
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def _safe_similarity(a: str, b: str) -> float:
    """
    Computes SequenceMatcher similarity ratio between two strings.

    Wraps in try/except — SequenceMatcher should never raise on valid strings
    but defensive wrapping ensures this helper never crashes the fuzzy pass.

    Parameters
    ──────────
    a, b : str   Normalized title strings to compare.

    Returns
    ───────
    float  → similarity ratio in [0.0, 1.0].  0.0 on any error.
    """
    if not a or not b:
        return 0.0
    try:
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _safe_parse_time(raw: Optional[str]) -> Optional[datetime]:
    """
    Parses an ISO-8601 timestamp string into a timezone-aware datetime.

    Used for chronological tie-breaking in cluster representative selection.
    The upstream news_engine.py produces timestamps in one of two formats:
        · ISO-8601 UTC:  "2025-04-09T14:32:00+00:00"
        · Raw RSS string: "Wed, 09 Apr 2025 14:32:00 GMT" (less reliable)

    Strategy:
        1. Try datetime.fromisoformat() (handles news_engine output correctly)
        2. Try a set of common RSS date format strings as fallback
        3. Return None on all failures — caller uses float("inf") as sentinel

    Always returns a UTC-aware datetime or None — never raises.

    Parameters
    ──────────
    raw : str | None   Published time string from the article dict.

    Returns
    ───────
    datetime  → timezone-aware UTC datetime
    None      → unparseable or missing timestamp
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip()

    # Attempt 1: ISO-8601 (primary format from news_engine.py)
    try:
        dt = datetime.fromisoformat(raw)
        # Ensure timezone-awareness — attach UTC if naive.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass

    # Attempt 2: Common RSS/HTTP date formats as fallback.
    RSS_FORMATS = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in RSS_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 8  —  METRICS & OBSERVABILITY
#
#  Mirrors the observability philosophy of news_engine.py:
#  · All output routed through _log()
#  · Single VERBOSITY constant controls detail level
#  · Drop-in compatible with Python logging module
#  · Prometheus-compatible field naming in summary dict
#
# ══════════════════════════════════════════════════════════════════════════════

def _log(level: int, tag: str, message: str) -> None:
    """
    Central log dispatcher.  All output in this module goes through here.

    Future swap:
        Replace print() with:
            logging.getLogger("duplicate_filter").info(f"[{tag}] {message}")
        That is the ONLY change needed for production-grade logging.
    """
    if VERBOSITY >= level:
        print(f"  [{tag}] {message}")


def _log_filter_summary(
    input_count  : int,
    output_count : int,
    cluster_count: int,
    elapsed      : float,
    survivors    : list[dict],
) -> None:
    """
    Prints the filter run summary report at VERBOSITY >= 1.

    Reports:
        · Articles in / out / removed
        · Reduction rate
        · Multi-source confirmation count
        · Top credibility boosts
        · Pipeline latency
    """
    if VERBOSITY < 1:
        return

    removed      = input_count - output_count
    reduction    = round((removed / input_count) * 100, 1) if input_count else 0.0
    multi_source = sum(1 for a in survivors if a.get("is_multi_source_confirmation"))
    confirmed    = [a for a in survivors if a.get("is_multi_source_confirmation")]

    print()
    print("═" * 72)
    print("  DUPLICATE FILTER  ·  RUN SUMMARY")
    print("═" * 72)
    print()
    print("  VOLUME")
    print(f"  {'Articles in':<30}: {input_count}")
    print(f"  {'Articles out (survivors)':<30}: {output_count}")
    print(f"  {'Articles removed':<30}: {removed}")
    print(f"  {'Reduction rate':<30}: {reduction}%")
    print(f"  {'Final clusters':<30}: {cluster_count}")
    print()
    print("  INTELLIGENCE")
    print(f"  {'Multi-source confirmations':<30}: {multi_source}")

    if confirmed:
        print()
        print("  TOP MULTI-SOURCE STORIES  (by credibility boost)")
        print("  " + "─" * 56)
        top = sorted(confirmed, key=lambda a: a.get("credibility_boost", 1.0), reverse=True)[:5]
        for art in top:
            boost   = art.get("credibility_boost", 1.0)
            sources = art.get("source_cluster", [])
            title   = _safe_str(art.get("title"))[:55]
            print(f"  ✦ boost={boost:.2f}  sources={len(sources)}  \"{title}...\"")
            if VERBOSITY >= 2:
                for src in sources:
                    print(f"       └─ {src}")

    print()
    print("  LATENCY")
    print(f"  {'Filter pipeline time':<30}: {elapsed}s")
    print()
    print("═" * 72)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#
#  DIRECT EXECUTION ENTRY POINT
#
#  python duplicate_filter.py
#
#  Runs a self-contained smoke test using synthetic articles.
#  Demonstrates exact dedup, fuzzy clustering, and intelligence enrichment.
#
#  In production this file is never run directly.
#  It is imported and called via:
#      articles = filter_duplicates(articles)
#  in the post-pipeline hook of news_engine.py.
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Synthetic article factory ─────────────────────────────────────────────
    def _make_article(
        title      : str,
        source     : str,
        link       : str       = "",
        priority   : int       = 3,
        category   : str       = "macro",
        region     : str       = "global",
        summary    : str | None = None,
        tags       : list[str] = None,
        dedupe_hash: str | None = None,
        pub_time   : str | None = None,
    ) -> dict:
        import hashlib
        h = hashlib.md5(f"{title}{link}".encode()).hexdigest()[:10]
        return {
            "source"                    : source,
            "category"                  : category,
            "priority"                  : priority,
            "region"                    : region,
            "title"                     : title,
            "link"                      : link,
            "summary"                   : summary,
            "published_time"            : pub_time,
            "tags"                      : tags or [],
            "dedupe_hash"               : dedupe_hash or h,
            "confidence_score"          : None,
            "impact_score"              : None,
            "regime_tag"                : None,
            "risk_flag"                 : None,
        }

    # ── Build test dataset ────────────────────────────────────────────────────
    test_articles = [

        # GROUP A — Exact duplicates (same hash): 3 articles, should → 1
        _make_article(
            "Fed raises rates by 25bps in surprise move",
            "Reuters", "https://reuters.com/fed-rates?utm_source=rss",
            priority=1, pub_time="2025-04-09T14:00:00+00:00",
            summary="The Federal Reserve raised its benchmark rate by 25 basis points.",
            tags=["fed", "rates", "macro"], dedupe_hash="EXACT001",
        ),
        _make_article(
            "Fed raises rates by 25bps in surprise move",
            "AP News", "https://apnews.com/fed-rates",
            priority=2, pub_time="2025-04-09T14:05:00+00:00",
            summary="AP: Fed hikes.",
            tags=["fed", "rates", "us"], dedupe_hash="EXACT001",
        ),
        _make_article(
            "Fed raises rates by 25bps in surprise move",
            "MarketWatch", "https://marketwatch.com/fed-rates",
            priority=3, pub_time="2025-04-09T14:10:00+00:00",
            summary=None,
            tags=["rates"], dedupe_hash="EXACT001",
        ),

        # GROUP B — Fuzzy duplicates (same story, different wording): 3 → 1
        _make_article(
            "CPI inflation data comes in hotter than expected at 3.8%",
            "Bloomberg", "https://bloomberg.com/cpi-march",
            priority=1, pub_time="2025-04-09T13:30:00+00:00",
            summary="Bloomberg: March CPI rose 3.8% year-over-year, beating consensus of 3.5%.",
            tags=["cpi", "inflation", "macro"],
        ),
        _make_article(
            "CPI inflation data hotter than expected, comes in at 3.8 percent",
            "CNBC", "https://cnbc.com/cpi-march-data",
            priority=2, pub_time="2025-04-09T13:32:00+00:00",
            summary="CNBC: CPI reading surprises to the upside.",
            tags=["cpi", "inflation", "us"],
        ),
        _make_article(
            "March CPI inflation comes in hotter than expected at 3.8%",
            "FT", "https://ft.com/cpi-march",
            priority=2, pub_time="2025-04-09T13:35:00+00:00",
            summary="FT: Hotter-than-expected inflation data rattles bond markets.",
            tags=["cpi", "bonds", "macro"],
        ),

        # GROUP C — Unique story (singleton): should survive unchanged
        _make_article(
            "Bitcoin ETF sees record $2.1bn inflow in single session",
            "CoinDesk", "https://coindesk.com/btc-etf-inflow",
            priority=2, category="crypto",
            summary="Bitcoin ETF recorded its largest single-day inflow since launch.",
            tags=["btc", "etf", "crypto"],
        ),

        # GROUP D — Exact duplicate via link (different hash, same normalized link)
        _make_article(
            "Oil prices drop on OPEC production news",
            "OilPrice.com", "https://oilprice.com/opec-output?utm_source=twitter",
            priority=3, category="commodities",
            tags=["oil", "opec"], dedupe_hash="LINK_TEST_A",
        ),
        _make_article(
            "Oil prices drop on OPEC production news",
            "Reuters Commodities", "https://oilprice.com/opec-output?utm_source=email",
            priority=2, category="commodities",
            tags=["oil", "commodities"], dedupe_hash="LINK_TEST_B",
        ),

        # GROUP E — Malformed articles (stress test resilience)
        {},                          # Empty dict
        {"title": None},             # Only a None title
        {"source": "BadFeed", "dedupe_hash": None, "priority": "banana"},
    ]

    print()
    print("═" * 72)
    print("  DUPLICATE FILTER  ·  SMOKE TEST")
    print(f"  Input: {len(test_articles)} articles (incl. malformed)")
    print("═" * 72)

    result = filter_duplicates(test_articles)

    print()
    print("  SURVIVORS  ·  Enriched Article Output")
    print("  " + "─" * 68)

    for i, art in enumerate(result, start=1):
        p       = art.get("priority", "?")
        cat     = _safe_str(art.get("category")).upper()
        boost   = art.get("credibility_boost", 1.0)
        multi   = art.get("is_multi_source_confirmation", False)
        cluster = art.get("source_cluster", [])
        title   = _safe_str(art.get("title"))[:60]

        print(f"  [{i:02}] [P{p}] [{cat}]")
        print(f"       Title       : {title}")
        print(f"       Sources     : {cluster}")
        print(f"       Boost       : {boost}")
        print(f"       Multi-Src   : {multi}")
        print(f"       Dup Count   : {art.get('duplicate_count', 1)}")
        print(f"       Size Score  : {art.get('cluster_size_score', 0.0)}")
        print(f"       Tags        : {art.get('tags', [])}")
        summary = _safe_str(art.get("summary"))
        print(f"       Summary     : {summary[:80] if summary else 'N/A'}")
        print()

    print("═" * 72)
    print(f"  INPUT   : {len(test_articles)} articles")
    print(f"  OUTPUT  : {len(result)} survivors")
    print(f"  REMOVED : {len(test_articles) - len(result)}")
    print("═" * 72)
    print()
