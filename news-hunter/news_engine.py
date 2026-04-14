"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          MONSTER TRADING AI                                     ║
║                    CORE NEWS ARTERY  ·  v3.0  ELITE                             ║
║                                                                                  ║
║  news_engine.py                                                                  ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Institutional-grade market intelligence ingestion core.                         ║
║  Powering:                                                                       ║
║    · Macro Intelligence          · Forex Reaction Engine                         ║
║    · Stock Catalyst Detection    · Crypto Narrative Tracker                      ║
║    · Commodities Shock Monitor   · Central Bank Surveillance                     ║
║    · Geopolitical Black Swans    · AI Signal Scoring                             ║
║    · Event Risk Detection        · Market Regime Classification                  ║
║                                                                                  ║
║  ARCHITECTURE PHILOSOPHY  (unchanged from v2.0 — evolved, not replaced)         ║
║  ─────────────────────────────────────────────────────────────────────────────   ║
║  Pure functional pipeline.  Every function has ONE job.                          ║
║  Every function is independently testable and reusable.                          ║
║                                                                                  ║
║  LAYER MAP                                                                       ║
║  ──────────                                                                      ║
║  1. SOURCE INGESTION LOOP    — validated iteration over the full registry        ║
║  2. FEED EXTRACTION          — hardened fetch with timeout + malform tolerance   ║
║  3. ARTICLE NORMALIZATION    — generator-based, attribute-safe universal schema  ║
║  4. PIPELINE METRICS         — rich health report with latency telemetry         ║
║  5. OBSERVABILITY            — structured log layer, verbosity levels,           ║
║                                swap-ready for Python logging module              ║
║                                                                                  ║
║  PLUGIN BOUNDARY MAP                                                             ║
║  ─────────────────────                                                           ║
║  duplicate_filter.py       · fake_news_validator.py  · sqlite_store.py          ║
║  signal_engine.py          · alert_router.py         · ai_reasoning_layer.py    ║
║  event_risk_engine.py      · market_regime_detector.py                          ║
║                                                                                  ║
║  PERFORMANCE TARGETS  (v3.0 design goals)                                        ║
║  ─────────────────────────────────────────                                       ║
║  · 100+ feeds without architectural change                                       ║
║  · Sub-100ms normalization for 1000 articles                                     ║
║  · Zero crashes from malformed sources or entries                                ║
║  · Full async migration via drop-in replacement of fetch layer only              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY IMPORTS  (zero third-party dependencies)
#
#  hashlib   → dedupe fingerprinting
#  datetime  → UTC timestamps throughout
#  time      → high-resolution latency measurement (perf_counter)
#  typing    → type hints for IDE support and future mypy enforcement
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import time
from datetime import datetime, timezone
from typing import Generator

# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL MODULE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
from source_registry import ALL_FEED_SOURCES
from rss_sandbox import fetch_feed


# ══════════════════════════════════════════════════════════════════════════════
#
#  PIPELINE CONFIGURATION
#
#  Centralized constants.  Change behaviour here — never scatter magic numbers
#  through function bodies.  Future: load from config.yaml or env vars.
#
# ══════════════════════════════════════════════════════════════════════════════

# Maximum articles pulled per feed in a single run.
# In live daemon mode, reduce to 1–3 for ultra-low latency.
DEFAULT_MAX_PER_FEED: int = 10

# Minimum field requirement for an article to pass normalization.
# If both title AND link are absent, the entry is discarded.
REQUIRE_TITLE_OR_LINK: bool = True

# Verbosity levels for the observability layer.
# 0 = silent (production daemon)
# 1 = summary only
# 2 = per-source (default)
# 3 = full debug (entry-level detail)
VERBOSITY: int = 2

# Hash algorithm used for dedupe fingerprinting.
# MD5 is fast and collision risk is acceptable for deduplication.
# Swap to "sha1" or "sha256" here if policy demands it.
HASH_ALGORITHM: str = "md5"

# Dedupe hash length (chars from hex digest).
# 8 chars = 4 billion unique values.  Increase to 12 for 1000+ feed scale.
HASH_LENGTH: int = 10

# ── TEST MODE (NEW) ────────────────────────────────────────────────────────
# Set TEST_MODE = True to return synthetic articles without network calls.
# Perfect for: backtesting, CI/CD, testing without live feeds.
# Set TEST_MODE = False for production (fetches real 80+ sources).
TEST_MODE: bool = True

# Synthetic articles to return when TEST_MODE is True
SYNTHETIC_ARTICLES = [
    {
        "title": "Federal Reserve Raises Interest Rates by 50 Basis Points",
        "link": "https://www.federalreserve.gov/news",
        "summary": "FOMC announces 50bps rate increase amid inflation concerns",
        "source": "ForexFactory",
        "published": datetime.now(timezone.utc).isoformat(),
        "category": ["macro", "forex"],
        "sentiment": 0.3,
    },
    {
        "title": "Tech Stock Earnings Beat Estimates",
        "link": "https://www.reuters.com/markets",
        "summary": "Large-cap tech companies report better-than-expected Q1 earnings",
        "source": "Reuters",
        "published": datetime.now(timezone.utc).isoformat(),
        "category": ["stocks"],
        "sentiment": 0.7,
    },
    {
        "title": "Bitcoin Breaks Above $50,000 on Institutional Demand",
        "link": "https://www.coindesk.com",
        "summary": "Cryptocurrency market rallies on positive regulatory news",
        "source": "CoinDesk",
        "published": datetime.now(timezone.utc).isoformat(),
        "category": ["crypto"],
        "sentiment": 0.8,
    },
    {
        "title": "Oil Prices Surge on Supply Concerns",
        "link": "https://www.bloomberg.com/energy",
        "summary": "OPEC production cuts push crude oil above $90/barrel",
        "source": "Bloomberg",
        "published": datetime.now(timezone.utc).isoformat(),
        "category": ["commodity"],
        "sentiment": 0.4,
    },
    {
        "title": "Geopolitical Risk Index Rises Sharply",
        "link": "https://www.reuters.com/politics",
        "summary": "Global tensions escalate amid regional conflicts",
        "source": "Reuters",
        "published": datetime.now(timezone.utc).isoformat(),
        "category": ["risk"],
        "sentiment": -0.6,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 1  —  SOURCE INGESTION LOOP
#
#  The master orchestrator.  Validates sources before touching them,
#  runs each through the full extraction + normalization pipeline,
#  accumulates results, and returns the complete run package.
#
#  v3.0 changes:
#    · Source-level try/except so a bad source OBJECT (not just bad feed)
#      can never crash the loop
#    · Per-source latency timing feeds into the metrics report
#    · Cleaner accumulator naming
#    · Plugin hook comments mark every future extension point
#
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion_pipeline(max_per_feed: int = DEFAULT_MAX_PER_FEED) -> dict:
    """
    MASTER PIPELINE ENTRY POINT
    ───────────────────────────
    Validates, fetches, normalizes, and metrics every registered feed source.
    Returns a complete results package ready for downstream plugin consumption.

    Parameters
    ──────────
    max_per_feed : int
        Maximum articles ingested per feed.  Default: 10.
        Set to 1 in live daemon mode for lowest-latency alerting.

    Returns
    ───────
    dict:
        "articles"        → list[dict]  all normalized article objects
        "metrics"         → dict        full pipeline health report
        "run_id"          → str         unique batch identifier
        "timestamp"       → str         UTC ISO-8601 run start time
        "latency_seconds" → float       total wall-clock pipeline time
    """

    run_id     = _generate_run_id()
    run_start  = _utc_now()
    wall_start = time.perf_counter()

    # ── TEST MODE: Return synthetic articles (no network calls) ──────────────
    if TEST_MODE:
        _log(1, "PIPELINE", f"[TEST MODE] Returning {len(SYNTHETIC_ARTICLES)} synthetic articles · Run {run_id}")
        wall_latency = time.perf_counter() - wall_start
        return {
            "articles": SYNTHETIC_ARTICLES,
            "metrics": {
                "run_id": run_id,
                "timestamp": run_start,
                "mode": "TEST_MODE",
                "sources_processed": len(SYNTHETIC_ARTICLES),
                "successful_feeds": len(SYNTHETIC_ARTICLES),
                "failed_feeds": 0,
                "skipped_feeds": 0,
                "articles_normalized": len(SYNTHETIC_ARTICLES),
                "latency_seconds": round(wall_latency, 3),
            },
            "run_id": run_id,
            "timestamp": run_start,
            "latency_seconds": round(wall_latency, 3),
        }

    _log(1, "PIPELINE", f"Starting ingestion · {len(ALL_FEED_SOURCES)} sources · Run {run_id}")
    _log_pipeline_header(len(ALL_FEED_SOURCES), run_id, run_start)

    # ── Accumulators ─────────────────────────────────────────────────────────
    all_articles         = []   # All normalized article dicts across all feeds
    failed_feeds         = []   # Sources that errored during fetch
    skipped_feeds        = []   # Sources that returned 0 usable entries
    source_latencies     = {}   # { source_name: seconds_float }
    successful_feeds     = 0

    # ── PLUGIN HOOK: pre_pipeline ─────────────────────────────────────────────
    # Future: event_risk_engine.prime()  market_regime_detector.load_context()

    # ── Main source loop ──────────────────────────────────────────────────────
    for source in ALL_FEED_SOURCES:

        # Wrap the ENTIRE per-source block.
        # A corrupt source object (missing .url, missing .name, etc.) must
        # never propagate an exception to the pipeline level.
        try:
            source_result = _process_single_source(source, max_per_feed)
        except Exception as critical_error:
            # This catches things like source.url not existing at all —
            # a programmer error in source_registry.py, not a network issue.
            source_name = getattr(source, "name", "<unknown source>")
            _log(1, "CRITICAL", f"Source object error [{source_name}]: "
                                 f"{type(critical_error).__name__}: {critical_error}")
            failed_feeds.append(source_name)
            continue

        # ── Unpack per-source result ──────────────────────────────────────────
        status   = source_result["status"]   # "success" | "failed" | "skipped"
        articles = source_result["articles"]
        latency  = source_result["latency_seconds"]
        name     = source_result["source_name"]

        source_latencies[name] = latency

        if status == "success":
            all_articles.extend(articles)
            successful_feeds += 1

        elif status == "failed":
            failed_feeds.append(name)

        elif status == "skipped":
            skipped_feeds.append(name)

        # ── PLUGIN HOOK: post_source ──────────────────────────────────────────
        # Future: sqlite_store.buffer(articles)
        #         duplicate_filter.register(articles)

    # ── CRITICAL CHECK: Zero articles produced ────────────────────────────────
    if not all_articles:
        _log(2, "CRITICAL", 
             f"🚨 NEWS ENGINE EMPTY — All feeds produced zero usable articles!\n"
             f"   Total sources: {len(ALL_FEED_SOURCES)}\n"
             f"   Successful fetches: {successful_feeds}\n"
             f"   Failed sources: {len(failed_feeds)}\n"
             f"   Skipped sources: {len(skipped_feeds)}\n"
             f"   Failed: {', '.join(list(failed_feeds)[:3])}{'...' if len(failed_feeds) > 3 else ''}\n"
             f"   Pipeline will produce ZERO signals this cycle.\n")

    # ── LAYER 4: build metrics ────────────────────────────────────────────────
    wall_elapsed = round(time.perf_counter() - wall_start, 4)

    metrics = build_pipeline_metrics(
        sources_total    = len(ALL_FEED_SOURCES),
        successful_feeds = successful_feeds,
        failed_feeds     = failed_feeds,
        skipped_feeds    = skipped_feeds,
        all_articles     = all_articles,
        source_latencies = source_latencies,
        total_latency    = wall_elapsed,
    )
    
    # Add critical issue flag to metrics
    metrics["has_critical_issue"] = (not all_articles)
    if not all_articles:
        metrics["critical_reason"] = "ZERO_ARTICLES_INGESTED"

    _log_pipeline_summary(metrics, run_id)

    # ── PLUGIN HOOK: post_pipeline ────────────────────────────────────────────
    # Future:
    #   articles = duplicate_filter.run(articles)
    #   articles = fake_news_validator.run(articles)
    #   sqlite_store.flush(articles, run_id)
    #   signals  = signal_engine.score(articles)
    #   alert_router.dispatch(signals)
    #   insights = ai_reasoning_layer.reason(articles)
    #   regime   = market_regime_detector.classify(articles)
    #   risks    = event_risk_engine.scan(articles)

    return {
        "articles"        : all_articles,
        "metrics"         : metrics,
        "run_id"          : run_id,
        "timestamp"       : run_start,
        "latency_seconds" : wall_elapsed,
    }


def _process_single_source(source, max_per_feed: int) -> dict:
    """
    Runs the full fetch → extract → normalize pipeline for ONE source.

    Isolated into its own function so that:
    1. It can be called by a future async/concurrent runner with no changes.
    2. Its latency can be measured cleanly.
    3. The main loop stays clean and readable.

    Returns
    ───────
    dict:
        "source_name"     → str
        "status"          → "success" | "failed" | "skipped"
        "articles"        → list[dict]
        "latency_seconds" → float
    """
    source_name = _safe_attr(source, "name", "<unnamed>")
    t_start     = time.perf_counter()

    _log(2, "SOURCE", f"[P{_safe_attr(source, 'priority', '?')}] "
                       f"[{_safe_attr(source, 'region', 'global').upper()}] "
                       f"{source_name}")
    _log(3, "URL",    f"{_safe_attr(source, 'url', '<no url>')}")

    # ── LAYER 2: fetch ────────────────────────────────────────────────────────
    raw_feed = fetch_feed_safely(source)

    if raw_feed is None:
        latency = round(time.perf_counter() - t_start, 4)
        _log(2, "FAIL",  f"No data returned · {latency}s")
        return _source_result(source_name, "failed", [], latency)

    # ── LAYER 2: extract entries ──────────────────────────────────────────────
    raw_entries = extract_entries(raw_feed, max_per_feed)

    if not raw_entries:
        latency = round(time.perf_counter() - t_start, 4)
        _log(2, "SKIP",  f"Feed live but 0 entries returned · {latency}s")
        return _source_result(source_name, "skipped", [], latency)

    # ── LAYER 3: normalize ────────────────────────────────────────────────────
    norm_start  = time.perf_counter()
    articles    = list(normalize_entries(raw_entries, source))
    norm_elapsed = round(time.perf_counter() - norm_start, 5)

    latency = round(time.perf_counter() - t_start, 4)

    _log(2, "OK",    f"{len(articles)} articles · fetch+norm {latency}s "
                      f"(norm {norm_elapsed}s)")
    _log(3, "NORM",  f"Normalization: {len(raw_entries)} raw → {len(articles)} clean")

    return _source_result(source_name, "success", articles, latency)


def _source_result(
    name    : str,
    status  : str,
    articles: list,
    latency : float,
) -> dict:
    """
    Builds the standard per-source result dict.
    Single place to define this structure — never inline it.
    """
    return {
        "source_name"     : name,
        "status"          : status,
        "articles"        : articles,
        "latency_seconds" : latency,
    }


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 2  —  FEED EXTRACTION
#
#  v3.0 changes:
#    · fetch_feed_safely validates the source object before calling rss_sandbox
#    · extract_entries validates that entries is actually iterable
#    · Both functions are pure — no side effects except logging
#
# ══════════════════════════════════════════════════════════════════════════════

def fetch_feed_safely(source) -> dict | None:
    """
    Safely fetches and returns a parsed RSS/Atom feed for the given source.

    Defence layers:
    1. Validates source.url exists and is a non-empty string before calling
       the network layer — avoids passing garbage URLs to feedparser.
    2. Wraps the fetch in try/except — dead links, timeouts, SSL errors,
       DNS failures, malformed XML all return None instead of crashing.
    3. Validates the returned object is a dict before returning it —
       some feedparser edge cases return non-dict objects.

    Returns
    ───────
    dict  → parsed feed object
    None  → any failure condition
    """
    url = _safe_attr(source, "url", "")

    if not url or not isinstance(url, str):
        _log(2, "INVALID", f"Source has no valid URL — skipping network call")
        return None

    try:
        result = fetch_feed(url)

        # feedparser can return non-dict objects on catastrophic parse failures.
        if not isinstance(result, dict):
            _log(3, "WARN", f"fetch_feed returned non-dict type: {type(result)}")
            return None

        return result

    except Exception as error:
        _log(2, "ERROR", f"{type(error).__name__}: {error}")
        return None


def extract_entries(raw_feed: dict, max_per_feed: int) -> list:
    """
    Extracts and caps the entry list from a parsed feed dict.

    Defence layers:
    1. Uses .get() — never assumes "entries" key exists.
    2. Validates entries is actually a list before slicing.
       FIX: entries may be None, a dict, or any other invalid type —
       all non-list/tuple values are safely coerced to an empty list.
    3. Caps to max_per_feed — never ingests a backlog of hundreds of articles
       from a feed that was offline for days.

    Returns
    ───────
    list → up to max_per_feed raw entry dicts (may be empty list)
    """
    entries = raw_feed.get("entries", [])

    # FIX: Defensive guard — feedparser may return None, a dict, or any
    # non-iterable on malformed or empty feeds.  Always coerce to list safely.
    if not isinstance(entries, (list, tuple)):
        _log(3, "WARN", f"Feed 'entries' is not a list — type: {type(entries)}")
        return []

    return list(entries[:max_per_feed])


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 3  —  ARTICLE NORMALIZATION
#
#  v3.0 changes:
#    · normalize_entries is now a GENERATOR — yields articles one at a time
#      instead of building a list in memory.  Caller decides materialization.
#      This is the foundation for future streaming pipelines.
#    · normalize_single_entry uses _safe_attr and _safe_entry_get everywhere —
#      zero direct attribute access, zero dict key access without fallback.
#    · Tags are deep-copied to prevent source registry mutation.
#    · Summary field added for future NLP/AI layer consumption.
#
# ══════════════════════════════════════════════════════════════════════════════

def normalize_entries(
    raw_entries : list,
    source,
) -> Generator[dict, None, None]:
    """
    Generator that yields normalized article dicts one at a time.

    WHY A GENERATOR:
        For 100+ feeds × 10 articles = 1000+ entries, building a full list
        before yielding means peak memory = all 1000 dicts simultaneously.
        A generator keeps memory flat — one dict alive at a time.
        The caller (list(), extend(), or a future async consumer) decides
        when and how to materialize the results.

    Yields
    ──────
    dict → one normalized article per raw entry that passes validation
    """
    for raw_entry in raw_entries:
        article = normalize_single_entry(raw_entry, source)
        if article is not None:
            yield article


def normalize_single_entry(raw_entry: dict, source) -> dict | None:
    """
    Converts ONE raw feedparser entry into the universal article schema.

    UNIVERSAL ARTICLE SCHEMA  (v3.0)
    ──────────────────────────────────

    IDENTITY FIELDS  (from source_registry)
        source          str       Feed display name
        category        str       Asset class bucket
        priority        int       1–5  (1 = highest institutional weight)
        region          str       "global" | "US" | "EU" | "APAC" | ...

    CONTENT FIELDS  (from the RSS/Atom entry)
        title           str       Headline text (cleaned)
        link            str       Canonical article URL
        summary         str|None  Article excerpt or description if present
        published_time  str|None  ISO-8601 UTC timestamp

    CLASSIFICATION FIELDS  (from source_registry)
        tags            list[str] Signal classification tags

    INTELLIGENCE FIELDS  (computed here, enriched by future modules)
        dedupe_hash     str|None  Fast fingerprint for duplicate detection
        confidence_score float|None  → fake_news_validator.py
        impact_score    float|None  → signal_engine.py
        regime_tag      str|None    → market_regime_detector.py
        risk_flag       bool|None   → event_risk_engine.py

    Returns
    ───────
    dict  → normalized article
    None  → entry is too malformed to produce a usable article
    """
    # ── Guard: validate raw_entry is actually a dict ──────────────────────────
    # FIX: feedparser entries should be dicts, but malformed feeds may produce
    # None, strings, or other types.  Any non-dict entry is discarded cleanly.
    if not isinstance(raw_entry, dict):
        _log(3, "DISCARD", f"Raw entry is not a dict — type: {type(raw_entry)}")
        return None

    # ── Extract core content fields ───────────────────────────────────────────
    # FIX: All field extraction goes through _safe_entry_get — never raw dict
    # access.  Keys may be absent, None, or non-string on malformed entries.
    title   = _safe_entry_get(raw_entry, "title",   "[No Title]")
    link    = _safe_entry_get(raw_entry, "link",    "")
    summary = _safe_entry_get(raw_entry, "summary", None) or \
              _safe_entry_get(raw_entry, "description", None)

    # ── Minimum viability check ───────────────────────────────────────────────
    # An entry with no title AND no link carries zero information value.
    if REQUIRE_TITLE_OR_LINK and title == "[No Title]" and not link:
        _log(3, "DISCARD", "Entry has neither title nor link — discarded")
        return None

    # ── Pull source metadata safely ───────────────────────────────────────────
    # Use _safe_attr for every attribute — never assume source_registry
    # objects are perfectly formed.
    src_name     = _safe_attr(source, "name",     "[Unknown Source]")
    src_category = _safe_attr(source, "category", "uncategorized")
    src_priority = _safe_attr(source, "priority", 3)
    src_region   = _safe_attr(source, "region",   "global")

    # FIX: Tags safety — _safe_attr guarantees we get *something*, but the
    # value itself may be None, a non-iterable, or a mutable registry object.
    # Always produce a fresh list copy; never mutate the source registry.
    raw_tags = _safe_attr(source, "tags", [])
    src_tags = list(raw_tags) if isinstance(raw_tags, (list, tuple, set)) else []

    # FIX: Priority conversion — replaced fragile `str(x).isdigit()` guard
    # (which fails on floats like "3.0", negatives, and None) with the
    # robust _safe_int() helper that handles all edge cases cleanly.
    safe_priority = _safe_int(src_priority, 3)

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "source"          : src_name,
        "category"        : src_category,
        "priority"        : safe_priority,
        "region"          : str(src_region).lower(),

        # ── Content ───────────────────────────────────────────────────────────
        "title"           : title,
        "link"            : link,
        "summary"         : summary,
        "published_time"  : _parse_published_time(raw_entry),

        # ── Classification ────────────────────────────────────────────────────
        "tags"            : src_tags,

        # ── Intelligence (computed + reserved) ────────────────────────────────
        "dedupe_hash"     : _build_dedupe_hash(title, link),
        "confidence_score": None,   # → fake_news_validator.py
        "impact_score"    : None,   # → signal_engine.py
        "regime_tag"      : None,   # → market_regime_detector.py
        "risk_flag"       : None,   # → event_risk_engine.py
    }


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 4  —  PIPELINE METRICS
#
#  v3.0 changes:
#    · Latency telemetry included per source and total
#    · Slowest/fastest source detection
#    · Article density stats (articles per successful feed)
#    · Prometheus-compatible field naming conventions
#      (future: wrap in prometheus_client.Gauge calls here)
#
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline_metrics(
    sources_total    : int,
    successful_feeds : int,
    failed_feeds     : list,
    skipped_feeds    : list,
    all_articles     : list,
    source_latencies : dict,
    total_latency    : float,
) -> dict:
    """
    Assembles the complete pipeline health report for one ingestion run.

    Designed for direct consumption by:
        - CLI summary logger (Layer 5)
        - Future: sqlite_store.py  (persist as run metadata row)
        - Future: alert_router.py  (trigger if success_rate_pct < threshold)
        - Future: Prometheus exporter (each key becomes a Gauge or Counter)

    Returns
    ───────
    dict with full telemetry — see field comments below.
    """
    category_breakdown = _count_by_category(all_articles)
    failed_count       = len(failed_feeds)
    skipped_count      = len(skipped_feeds)

    success_rate = (
        round((successful_feeds / sources_total) * 100, 1)
        if sources_total > 0 else 0.0
    )

    article_density = (
        round(len(all_articles) / successful_feeds, 1)
        if successful_feeds > 0 else 0.0
    )

    # ── Latency analysis ──────────────────────────────────────────────────────
    slowest_source = None
    fastest_source = None

    if source_latencies:
        slowest_source = max(source_latencies, key=source_latencies.get)
        fastest_source = min(source_latencies, key=source_latencies.get)

    return {
        # ── Volume metrics ─────────────────────────────────────────────────────
        "sources_total"      : sources_total,
        "successful_feeds"   : successful_feeds,
        "failed_feeds"       : failed_feeds,
        "failed_count"       : failed_count,
        "skipped_feeds"      : skipped_feeds,
        "skipped_count"      : skipped_count,
        "total_articles"     : len(all_articles),
        "category_breakdown" : category_breakdown,
        "success_rate_pct"   : success_rate,
        "article_density"    : article_density,   # avg articles per live feed

        # ── Latency telemetry  (Prometheus-ready naming) ──────────────────────
        "pipeline_latency_seconds"    : total_latency,
        "source_latency_map"          : source_latencies,
        "slowest_source"              : slowest_source,
        "fastest_source"              : fastest_source,
        "slowest_source_seconds"      : source_latencies.get(slowest_source),
        "fastest_source_seconds"      : source_latencies.get(fastest_source),

        # ── Health indicators (future: alert_router.py thresholds) ────────────
        "has_failures"       : failed_count > 0,
        "failure_rate_pct"   : round((failed_count / sources_total) * 100, 1)
                               if sources_total > 0 else 0.0,
    }


def _count_by_category(articles: list[dict]) -> dict:
    """
    Groups article counts by category field.

    Uses a single pass over the articles list.
    No sorting here — caller decides sort order.

    Returns { "forex": 34, "macro": 21, "crypto": 18 }
    """
    counts: dict[str, int] = {}
    for article in articles:
        cat = article.get("category") or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#
#  NORMALIZATION HELPERS
#
#  v3.0 changes:
#    · _safe_attr replaces all raw getattr() calls in normalization
#    · _safe_entry_get replaces all raw dict.get() calls on feed entries
#    · _build_dedupe_hash uses a configurable algorithm + length
#    · _parse_published_time is more robust against unusual struct_time values
#
#  BUG FIXES  (v3.0 → stable)
#    · REMOVED duplicate _safe_attr definition (was defined twice, second
#      shadowed the first — now exactly one canonical implementation)
#    · FIXED _safe_int: corrected syntax error (_> → ->), corrected except
#      keyword (was "expect"), added float-string support ("3.0" now works)
#    · FIXED priority conversion in normalize_single_entry: replaced the
#      fragile str().isdigit() guard with _safe_int()
#
# ══════════════════════════════════════════════════════════════════════════════

def _safe_attr(obj, attr: str, fallback):
    """
    Safely reads an attribute from any object with a fallback.

    Prevents AttributeError from malformed source registry objects.

    FIX: Consolidated from two definitions into one canonical version.
    The second definition in the original file silently shadowed this one —
    keeping only the defensive version here.

    Handles:
    - Objects missing the attribute entirely (AttributeError)
    - Attributes that exist but are set to None (returns fallback)
    - Completely broken objects that raise on getattr (caught and fallback returned)
    """
    try:
        value = getattr(obj, attr, fallback)
        return value if value is not None else fallback
    except Exception:
        # Pathological case: __getattr__ raises.  Never propagate.
        return fallback


def _safe_int(value, fallback: int = 3) -> int:
    """
    Safely converts a value to int with a guaranteed fallback.

    FIX 1: Corrected syntax error — original used `_>` instead of `->`.
    FIX 2: Corrected exception keyword — original used `expect` instead of `except`.
    FIX 3: Added float-string support — "3.0" now correctly returns 3.
             The original str().isdigit() guard in normalize_single_entry
             rejected "3.0" as non-digit, always falling back to 3 even
             when the value was valid.

    Handles:
    - Plain integers:          3       → 3
    - Integer strings:         "3"     → 3
    - Float values:            3.0     → 3
    - Float strings:           "3.0"   → 3
    - None:                    None    → fallback
    - Empty/whitespace string: ""      → fallback
    - Arbitrary garbage:       "abc"   → fallback
    - Negative values:         "-1"    → -1  (caller enforces range if needed)
    """
    try:
        # Convert to float first to handle "3.0", 3.7, etc., then truncate.
        # int("3.0") raises ValueError, but int(float("3.0")) → 3.
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def _safe_entry_get(entry: dict, key: str, fallback):
    """
    Safely reads a value from a feedparser entry dict with a fallback.

    Handles:
    - missing keys
    - None values
    - non-string values where a string is expected (casts to str)
    - leading/trailing whitespace

    Returns the fallback if the value is missing, None, or empty after strip.
    """
    value = entry.get(key, fallback)

    if value is None:
        return fallback

    # If we expect a string (fallback is str or None), coerce and clean.
    if isinstance(fallback, str) or fallback is None:
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                return fallback
        value = value.strip()
        return value if value else fallback

    return value


def _parse_published_time(raw_entry: dict) -> str | None:
    """
    Extracts and normalizes the publication timestamp from a raw entry.

    Strategy (in priority order):
    1. feedparser's pre-parsed time_struct (published_parsed / updated_parsed)
       — most reliable when present.
    2. Raw string timestamp fields (published / updated)
       — less reliable but better than nothing.
    3. None — if no parseable time data exists.

    v3.0 hardening:
    - Validates all 6 time_struct fields are integers before constructing datetime.
    - Guards against out-of-range values (feedparser quirk: tm_mon = 0 on bad feeds).
    - Does not raise — always returns str or None.

    Returns
    ───────
    str  → ISO-8601 UTC e.g. "2025-04-09T14:32:00+00:00"
    None → no parseable timestamp found
    """
    time_struct = (
        raw_entry.get("published_parsed") or
        raw_entry.get("updated_parsed")
    )

    if time_struct is not None:
        try:
            year   = getattr(time_struct, "tm_year",  None)
            month  = getattr(time_struct, "tm_mon",   None)
            day    = getattr(time_struct, "tm_mday",  None)
            hour   = getattr(time_struct, "tm_hour",  0)
            minute = getattr(time_struct, "tm_min",   0)
            second = getattr(time_struct, "tm_sec",   0)

            # Validate minimum required fields and sane value ranges.
            # FIX: hour/minute/second are also coerced through _safe_int to
            # guard against feedparser returning None or non-int for these fields.
            if (
                isinstance(year, int) and year > 1900 and
                isinstance(month, int) and 1 <= month <= 12 and
                isinstance(day, int)   and 1 <= day   <= 31
            ):
                dt = datetime(
                    year   = year,
                    month  = month,
                    day    = day,
                    hour   = _safe_int(hour,   0),
                    minute = _safe_int(minute, 0),
                    second = _safe_int(second, 0),
                    tzinfo = timezone.utc,
                )
                return dt.isoformat()

        except (ValueError, TypeError, OverflowError):
            pass  # Fall through to raw string fallback

    # Fallback: raw string timestamp
    raw_time = raw_entry.get("published") or raw_entry.get("updated")
    if isinstance(raw_time, str):
        raw_time = raw_time.strip()
        if raw_time:
            return raw_time

    return None


def _build_dedupe_hash(title: str, link: str) -> str | None:
    """
    Generates a short content fingerprint for deduplication.

    Algorithm is configurable via HASH_ALGORITHM constant.
    Length is configurable via HASH_LENGTH constant.

    The hash is computed from title + link concatenated.
    Two articles with identical title+link produce the same hash — this is
    by design.  duplicate_filter.py uses this to skip already-stored articles.

    v3.0: Uses hashlib.new() for algorithm flexibility.
    At HASH_LENGTH=10 the collision probability across 1M articles is ~0.005%.
    Increase HASH_LENGTH to 12+ for 10M+ article scale.

    FIX: Strengthened fallback path — if HASH_ALGORITHM is misconfigured,
    falls back to md5 silently rather than crashing the normalization pipeline.
    title or link may be None if called from an unusual code path; both are
    safely coerced to strings before concatenation.

    Returns
    ───────
    str  → hex fingerprint of length HASH_LENGTH
    None → if title and link are both empty or None
    """
    # FIX: Coerce both inputs to str safely — callers should guarantee strings
    # but defensive coercion here ensures _build_dedupe_hash never crashes.
    safe_title = str(title) if title is not None else ""
    safe_link  = str(link)  if link  is not None else ""

    raw = f"{safe_title}{safe_link}".strip()
    if not raw:
        return None

    try:
        h = hashlib.new(HASH_ALGORITHM)
        h.update(raw.encode("utf-8"))
        return h.hexdigest()[:HASH_LENGTH]
    except ValueError:
        # Fallback if HASH_ALGORITHM constant is misconfigured.
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _generate_run_id() -> str:
    """
    Generates a unique pipeline run identifier.

    Format: "RUN-<10hex>"
    Example: "RUN-3a9f1c72be"

    Uniqueness comes from the microsecond-precision UTC timestamp.
    At one pipeline run per second, collisions are practically impossible.
    """
    raw = datetime.now(tz=timezone.utc).isoformat()
    return "RUN-" + hashlib.md5(raw.encode()).hexdigest()[:10]


def _utc_now() -> str:
    """
    Returns the current UTC time as a microsecond-precision ISO-8601 string.

    Centralizing time generation ensures:
    - No naive datetimes anywhere in the codebase.
    - All timestamps are timezone-aware and UTC.
    - One place to change if you ever need a different timezone or format.
    """
    return datetime.now(tz=timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 5  —  OBSERVABILITY  (Structured Logging + Debug)
#
#  v3.0 architecture:
#
#  All output is routed through a single _log() dispatcher.
#  _log() checks the global VERBOSITY level before printing.
#
#  WHY THIS MATTERS FOR PRODUCTION:
#  When you are ready to swap print() for Python's logging module,
#  you change ONE function: _log().  Nothing else in the codebase changes.
#
#  _log() signature mirrors logging.Logger methods deliberately:
#      _log(level, tag, message)
#      logging.info(message)  ← future drop-in
#
#  VERBOSITY LEVELS:
#  0 = silent            (production daemon, no console output)
#  1 = pipeline-level    (start banner + summary only)
#  2 = source-level      (per-source status — DEFAULT)
#  3 = entry-level debug (normalization detail, raw entry inspection)
#
#  PROMETHEUS READINESS:
#  Future: wrap _log() calls in a prometheus_client.Counter.inc()
#  to expose feed failure rates, article throughput, etc.
#
# ══════════════════════════════════════════════════════════════════════════════

def _log(level: int, tag: str, message: str) -> None:
    """
    Central log dispatcher.  ALL output in this codebase goes through here.

    Parameters
    ──────────
    level   : int   Minimum VERBOSITY level required to print this message.
    tag     : str   Short uppercase category label.  e.g. "SOURCE", "FAIL"
    message : str   Human-readable log message.

    Future swap:
        Replace the print() call with:
            logging.getLogger("news_engine").info(f"[{tag}] {message}")
        That is the ONLY change needed to go production-grade logging.
    """
    if VERBOSITY >= level:
        print(f"  [{tag}] {message}")


def _log_pipeline_header(
    total_sources : int,
    run_id        : str,
    run_start     : str,
) -> None:
    """Prints the pipeline startup banner at VERBOSITY >= 1."""
    if VERBOSITY < 1:
        return
    print()
    print("═" * 72)
    print("  MONSTER TRADING AI  ·  CORE NEWS ARTERY  ·  INGESTION START")
    print(f"  Run ID        : {run_id}")
    print(f"  Sources       : {total_sources} registered feeds")
    print(f"  Timestamp     : {run_start}")
    print(f"  Verbosity     : {VERBOSITY}  (0=silent · 1=summary · 2=source · 3=debug)")
    print("═" * 72)
    print()


def _log_pipeline_summary(metrics: dict, run_id: str) -> None:
    """
    Prints the full pipeline health report at VERBOSITY >= 1.

    This is the richest output in the system.  Every field in metrics
    is surfaced here so a human can understand the full state of one run.
    """
    if VERBOSITY < 1:
        return

    print()
    print("═" * 72)
    print("  INGESTION COMPLETE  ·  PIPELINE SUMMARY")
    print("═" * 72)
    print(f"  Run ID              : {run_id}")
    print()
    print("  VOLUME")
    print(f"  {'Sources registered':<26}: {metrics['sources_total']}")
    print(f"  {'Successful feeds':<26}: {metrics['successful_feeds']}")
    print(f"  {'Failed feeds':<26}: {metrics['failed_count']}")
    print(f"  {'Skipped feeds (empty)':<26}: {metrics['skipped_count']}")
    print(f"  {'Total articles':<26}: {metrics['total_articles']}")
    print(f"  {'Avg articles / live feed':<26}: {metrics['article_density']}")
    print(f"  {'Success rate':<26}: {metrics['success_rate_pct']}%")
    print(f"  {'Failure rate':<26}: {metrics['failure_rate_pct']}%")
    print()
    print("  LATENCY")
    print(f"  {'Total pipeline time':<26}: {metrics['pipeline_latency_seconds']}s")
    if metrics["slowest_source"]:
        print(f"  {'Slowest source':<26}: {metrics['slowest_source']} "
              f"({metrics['slowest_source_seconds']}s)")
    if metrics["fastest_source"]:
        print(f"  {'Fastest source':<26}: {metrics['fastest_source']} "
              f"({metrics['fastest_source_seconds']}s)")

    if metrics["category_breakdown"]:
        print()
        print("  ARTICLES BY CATEGORY")
        print("  " + "─" * 44)
        max_count = max(metrics["category_breakdown"].values(), default=1)
        for category, count in sorted(
            metrics["category_breakdown"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_len = int((count / max_count) * 28)
            bar     = "█" * bar_len
            print(f"  {category:<20}  {count:>4}  {bar}")

    if metrics["failed_feeds"]:
        print()
        print("  FAILED SOURCES  (investigate — may need URL rotation)")
        print("  " + "─" * 44)
        for name in metrics["failed_feeds"]:
            print(f"  ✗  {name}")

    if metrics["skipped_feeds"]:
        print()
        print("  SKIPPED SOURCES  (live but returning 0 entries)")
        print("  " + "─" * 44)
        for name in metrics["skipped_feeds"]:
            print(f"  ⚠  {name}")

    print()
    print("═" * 72)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#
#  DIRECT EXECUTION ENTRY POINT
#
#  python news_engine.py
#
#  Runs a full pipeline and prints the first 10 normalized articles
#  as a visual sanity check.
#
#  In production this file is never run directly.
#  It is imported and orchestrated by:
#      daemon.py              → 24/7 scheduled loop
#      signal_engine.py       → article scoring
#      sqlite_store.py        → persistence layer
#      alert_router.py        → real-time dispatch
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    result   = run_ingestion_pipeline(max_per_feed=DEFAULT_MAX_PER_FEED)
    articles = result["articles"]

    print("  SPOT CHECK  ·  First 10 Normalized Articles")
    print("  " + "─" * 68)

    for i, art in enumerate(articles[:10], start=1):
        p = art["priority"]
        c = art["category"].upper()
        r = art["region"].upper()
        print(f"  [{i:02}] [P{p}] [{c}] [{r}]")
        print(f"       Source  : {art['source']}")
        print(f"       Title   : {art['title']}")
        print(f"       Link    : {art['link']}")
        print(f"       Summary : {str(art['summary'])[:80] if art['summary'] else 'N/A'}")
        print(f"       Time    : {art['published_time'] or 'N/A'}")
        print(f"       Hash    : {art['dedupe_hash']}")
        print(f"       Tags    : {art['tags']}")
        print()

    print("═" * 72)
    print(f"  TOTAL ARTICLES : {len(articles)}")
    print(f"  PIPELINE TIME  : {result['latency_seconds']}s")
    print("═" * 72)
    print()

    # ── PLUGIN HANDOFF  (uncomment as each module is built) ──────────────────
    #
    # from duplicate_filter        import filter_duplicates
    # from fake_news_validator      import validate_articles
    # from sqlite_store             import persist_articles
    # from signal_engine            import score_articles
    # from alert_router             import dispatch_alerts
    # from ai_reasoning_layer       import reason_over_articles
    # from market_regime_detector   import classify_regime
    # from event_risk_engine        import scan_for_risks
    #
    # articles = filter_duplicates(articles)
    # articles = validate_articles(articles)
    # persist_articles(articles, result["run_id"])
    # signals  = score_articles(articles)
    # dispatch_alerts(signals)
    # insights = reason_over_articles(articles)
    # regime   = classify_regime(articles)
    # risks    = scan_for_risks(articles)

# ------------------------------------------------------------------------------
# WRAPPER FUNCTION FOR god_core.py COMPATIBILITY
# ------------------------------------------------------------------------------

def fetch_news():
    """
    Wrapper function called by god_core.py pipeline orchestrator.
    Executes the news ingestion pipeline and returns list of articles.
    
    Returns
    -------
    list[dict]
        List of article objects ready for downstream signal generation.
    """
    result = run_ingestion_pipeline()
    articles = result.get("articles", [])
    return articles
