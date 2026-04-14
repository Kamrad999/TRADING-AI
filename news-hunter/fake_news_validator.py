"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          MONSTER TRADING AI                                     ║
║             CREDIBILITY VALIDATION ENGINE  ·  v1.0  INSTITUTIONAL               ║
║                                                                                  ║
║  fake_news_validator.py                                                          ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Hedge-fund-grade credibility scoring and misinformation risk detection.         ║
║                                                                                  ║
║  Pipeline position:                                                              ║
║      news_engine.py                                                              ║
║        → duplicate_filter.py                                                     ║
║          → fake_news_validator.py     ← YOU ARE HERE                            ║
║            → signal_engine.py                                                    ║
║              → alert_router.py                                                   ║
║                                                                                  ║
║  MISSION                                                                         ║
║  ───────                                                                         ║
║  This is NOT a simple fake news checker.                                         ║
║                                                                                  ║
║  Every article that reaches this module has already survived:                    ║
║    · news_engine.py  — schema normalization + structural validation              ║
║    · duplicate_filter.py — exact/fuzzy dedup + cluster enrichment               ║
║                                                                                  ║
║  This module answers a harder institutional question:                            ║
║    "Should a quantitative trading system ACT on this article?"                   ║
║                                                                                  ║
║  The answer is a composite of:                                                   ║
║    · Source pedigree (Reuters/Bloomberg/AP vs. anonymous blog)                   ║
║    · Multi-source confirmation depth (cluster intelligence from dedup layer)     ║
║    · Linguistic credibility signals (neutral vs. sensational wording)            ║
║    · Structural integrity signals (timestamp, summary, link quality)             ║
║    · Misinformation risk markers (rumor phrases, emotional language)             ║
║    · Contradiction heuristics (language that signals retraction or dispute)      ║
║                                                                                  ║
║  SCORING ARCHITECTURE                                                            ║
║  ─────────────────────                                                           ║
║  All scores are in [0.0, 1.0].  Higher = more trustworthy.                      ║
║                                                                                  ║
║  validation_score   Composite weighted score.  Final institutional verdict.      ║
║  trust_score        Source pedigree + structural integrity + cluster depth.      ║
║  misinformation_risk Inverse: 0.0 = clean, 1.0 = high misinformation signal.   ║
║                                                                                  ║
║  validation_score = (trust_score × TRUST_WEIGHT)                                ║
║                   − (misinformation_risk × MISINFO_WEIGHT)                      ║
║                   clamped to [0.0, 1.0]                                          ║
║                                                                                  ║
║  VERIFICATION STATUS THRESHOLDS                                                  ║
║  ────────────────────────────────                                                ║
║  VERIFIED          validation_score >= 0.75                                      ║
║  PROBABLE          validation_score >= 0.55                                      ║
║  UNVERIFIED        validation_score >= 0.35                                      ║
║  SUSPECT           validation_score <  0.35                                      ║
║                                                                                  ║
║  LAYER MAP                                                                       ║
║  ──────────                                                                      ║
║  1. PUBLIC API          — validate_articles() entry point                        ║
║  2. COMPOSITE SCORING   — weighted trust + risk combination                      ║
║  3. TRUST SCORING       — source pedigree, cluster depth, structural signals     ║
║  4. MISINFORMATION RISK — rumor phrases, emotional wording, structural red flags ║
║  5. CONTRADICTION       — language patterns signaling disputed/retracted claims  ║
║  6. WORDING ANALYSIS    — caps detection, exclamation abuse, phrase scanning     ║
║  7. SOURCE REGISTRY     — institutional tier classification                      ║
║  8. SAFETY HELPERS      — defensive wrappers, pure string utilities              ║
║  9. OBSERVABILITY       — structured logging + run summary reporting             ║
║                                                                                  ║
║  PLUGIN BOUNDARY MAP                                                             ║
║  ─────────────────────                                                           ║
║  Upstream   : duplicate_filter.py  → credibility_boost, source_cluster          ║
║  Downstream : signal_engine.py     → validation_score, is_high_confidence_news  ║
║               alert_router.py      → needs_manual_review, verification_status   ║
║               sqlite_store.py      → full validation audit trail                 ║
║               ai_reasoning_layer   → risk_reasons for explainability            ║
║                                                                                  ║
║  PERFORMANCE TARGETS                                                             ║
║  ─────────────────────                                                           ║
║  · 1000 articles validated in < 150ms                                           ║
║  · All scoring is O(1) per article (regex pre-compiled at module load)           ║
║  · Zero state between articles — fully stateless, parallelizable                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY IMPORTS  (zero third-party dependencies)
#
#  re       → pre-compiled regex for O(1) wording analysis per article
#  time     → perf_counter for pipeline latency measurement
#  math     → clamp utility (math.inf sentinel usage)
#  typing   → type hints for IDE support and future mypy enforcement
# ─────────────────────────────────────────────────────────────────────────────
import re
import time
import math
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#
#  MODULE CONFIGURATION
#
#  Every tunable constant lives here.  No magic numbers in function bodies.
#  Future: load from config.yaml or environment variables.
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Composite score weights ───────────────────────────────────────────────────
# validation_score = (trust × TRUST_WEIGHT) − (risk × MISINFO_WEIGHT), clamped.
# Must sum to 1.0 for the score to remain in [0.0, 1.0] when risk = 0.
TRUST_WEIGHT  : float = 0.65
MISINFO_WEIGHT: float = 0.35

# ── Verification status thresholds ───────────────────────────────────────────
THRESHOLD_VERIFIED  : float = 0.75   # Institutional-grade confidence
THRESHOLD_PROBABLE  : float = 0.55   # Tradeable with normal risk management
THRESHOLD_UNVERIFIED: float = 0.35   # Requires corroboration before action
# Below THRESHOLD_UNVERIFIED → "SUSPECT"

# ── Manual review trigger ─────────────────────────────────────────────────────
# Articles scoring below this threshold are flagged for human inspection.
MANUAL_REVIEW_THRESHOLD: float = 0.45

# ── High-confidence publication gate ─────────────────────────────────────────
# is_high_confidence_news = True only above this threshold.
HIGH_CONFIDENCE_THRESHOLD: float = 0.70

# ── Trust score component weights ─────────────────────────────────────────────
# Must sum to 1.0.  Adjust to shift emphasis between source vs. structure.
TRUST_W_SOURCE_TIER    : float = 0.35   # Outlet pedigree (tier classification)
TRUST_W_CLUSTER_DEPTH  : float = 0.25   # Multi-source confirmation depth
TRUST_W_CREDIBILITY    : float = 0.15   # credibility_boost from dedup layer
TRUST_W_TIMESTAMP      : float = 0.10   # Timestamp presence and recency
TRUST_W_SUMMARY        : float = 0.08   # Summary presence and substance
TRUST_W_WORDING        : float = 0.07   # Neutral / institutional language quality

# ── Misinformation risk component weights ────────────────────────────────────
# Must sum to 1.0.  Each component contributes its share to the risk score.
RISK_W_RUMOR_PHRASES   : float = 0.28   # "unconfirmed", "sources say", etc.
RISK_W_EMOTIONAL       : float = 0.18   # Sensational/emotional language
RISK_W_CAPS_ABUSE      : float = 0.12   # ALL CAPS title or excessive caps
RISK_W_EXCLAMATION     : float = 0.10   # Exclamation mark abuse
RISK_W_WEAK_SOURCE     : float = 0.14   # Low-priority outlet, no confirmation
RISK_W_SOCIAL_ORIGIN   : float = 0.10   # Social-media-first URL patterns
RISK_W_CONTRADICTION   : float = 0.08   # Contradiction / retraction language

# ── Source tier score map ─────────────────────────────────────────────────────
# Tier score is a direct input to trust_score.
# 1.0 = central bank or institutional primary source
# 0.0 = anonymous blog / unclassified outlet
SOURCE_TIER_SCORES: dict[str, float] = {
    "tier_1": 1.00,   # Wire services: Reuters, Bloomberg, AP, AFP
    "tier_2": 0.85,   # Premium financial press: FT, WSJ, NYT, Economist
    "tier_3": 0.70,   # Major broadcast / digital: CNBC, BBC, Forbes, Barron's
    "tier_4": 0.50,   # Specialized / regional: Seeking Alpha, Business Insider
    "tier_5": 0.25,   # Low-pedigree blogs, aggregators, unknown outlets
}

# ── Cluster depth scoring ─────────────────────────────────────────────────────
# Maps source_cluster length → trust contribution for that component.
# Single source = 0.30 base.  Each additional unique outlet adds CLUSTER_STEP.
# Capped at CLUSTER_CAP_SIZE outlets.
CLUSTER_BASE_SCORE : float = 0.30
CLUSTER_STEP       : float = 0.175
CLUSTER_CAP_SIZE   : int   = 5      # 5+ unique outlets → max cluster trust

# ── Caps abuse threshold ──────────────────────────────────────────────────────
# Fraction of alphabetic characters that are uppercase before flagging.
CAPS_RATIO_THRESHOLD: float = 0.65   # e.g. "FED SHOCK CRASH PANIC NOW" → flag

# ── Exclamation abuse threshold ───────────────────────────────────────────────
EXCLAMATION_THRESHOLD: int = 2   # > 2 exclamation marks in title → flag

# ── Summary minimum substance length ─────────────────────────────────────────
# Summaries shorter than this are treated as "absent" for trust scoring.
SUMMARY_MIN_CHARS: int = 40

# ── Social media domain patterns ─────────────────────────────────────────────
SOCIAL_MEDIA_DOMAINS: tuple[str, ...] = (
    "twitter.com", "x.com", "t.co",
    "facebook.com", "fb.com",
    "reddit.com",
    "telegram.org", "t.me",
    "tiktok.com",
    "instagram.com",
    "linkedin.com",       # Lower weight — LinkedIn can host financial news
)

# ── Verbosity ─────────────────────────────────────────────────────────────────
# 0 = silent  |  1 = summary  |  2 = per-article  |  3 = full debug
VERBOSITY: int = 2


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 7  —  SOURCE REGISTRY
#
#  Defined before the scoring layers so that compiled sets and dicts are
#  available at module load time.  All scoring functions reference these
#  constants — never hardcode outlet names in scoring logic.
#
#  HOW TO MAINTAIN THIS:
#  Add new outlets to the appropriate tier set.  The scoring engine picks
#  up the change automatically — no function body changes required.
#
#  TIER DEFINITIONS:
#  Tier 1 — Primary wire services.  First-mover, edit-checked, licensed.
#  Tier 2 — Premium financial press.  Deep editorial standards.
#  Tier 3 — Major digital / broadcast.  Reliable but occasionally sensational.
#  Tier 4 — Specialized/regional press.  Variable quality.
#  Tier 5 — Blogs, aggregators, social-media-adjacent.  High noise floor.
#
# ══════════════════════════════════════════════════════════════════════════════

# Each set contains lowercase outlet name fragments.
# Matching is substring-based: "bloomberg" matches "Bloomberg Markets" etc.

TIER_1_SOURCES: frozenset[str] = frozenset({
    "reuters", "bloomberg", "associated press", " ap ", "ap news",
    "afp", "dow jones", "federal reserve", "ecb", "bank of england",
    "imf", "world bank", "bis", "sec.gov", "treasury",
})

TIER_2_SOURCES: frozenset[str] = frozenset({
    "financial times", "wall street journal", "wsj", "new york times",
    "economist", "ft.com", "barron", "nikkei", "south china morning post",
    "handelsblatt", "les echos", "frankfurter allgemeine",
})

TIER_3_SOURCES: frozenset[str] = frozenset({
    "cnbc", "bbc", "forbes", "marketwatch", "yahoo finance",
    "business insider", "the guardian", "washington post",
    "fox business", "sky news", "deutsche welle",
    "al jazeera", "channel news asia",
})

TIER_4_SOURCES: frozenset[str] = frozenset({
    "seeking alpha", "investing.com", "benzinga", "thestreet",
    "motley fool", "zerohedge", "mish talk", "calculated risk",
    "market insider", "stockanalysis", "simply wall st",
})

# Tier 5 is the implicit fallback — anything not in Tiers 1–4.


# ══════════════════════════════════════════════════════════════════════════════
#
#  PRE-COMPILED REGEX PATTERNS
#
#  All regex patterns are compiled ONCE at module load time.
#  O(1) per-article match cost.  Never compile inside a loop.
#
#  Pattern categories:
#    · Rumor / unverified language detection
#    · Emotional / sensational language detection
#    · Contradiction / retraction language detection
#    · Neutral institutional language detection (positive trust signal)
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Rumor & unverified claim phrases ─────────────────────────────────────────
# These patterns are the strongest misinformation risk signal in financial news.
# Legitimate institutional reporting avoids these constructions.
_RUMOR_PATTERNS: re.Pattern = re.compile(
    r"""
    \b(
        unconfirmed                     |
        rumou?r[sd]?                    |   # "rumor", "rumours", "rumors"
        sources?\s+say                  |
        sources?\s+told                 |
        people\s+familiar               |
        according\s+to\s+sources        |
        allegedly                       |
        reportedly                      |   # weak — but a signal when combined
        whispers?                       |
        speculation[s]?                 |
        speculated                      |
        could\s+not\s+be\s+(independently\s+)?verified |
        has\s+not\s+been\s+(independently\s+)?verified |
        unverified                      |
        not\s+yet\s+confirmed           |
        yet\s+to\s+be\s+confirmed
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Emotional & sensational language ─────────────────────────────────────────
# Financial misinformation almost always uses urgency and fear as hooks.
# Institutional reporting uses neutral, measured language.
_EMOTIONAL_PATTERNS: re.Pattern = re.compile(
    r"""
    \b(
        BREAKING                        |
        URGENT                          |
        SHOCK(ED|ING)?                  |
        BOMBSHELL                       |
        EXPLOSIVE                       |
        CATASTROPH(E|IC|ICALLY)         |
        CRASH(ED|ING)?                  |   # context-dependent; scored lightly
        COLLAPS(E|ING|ED)               |
        PANIC(KING|KED)?                |
        MELTDOWN                        |
        TERRIF(Y|YING|IED)              |
        DISAST(ER|ROUS)                 |
        WIPE[D]?\s*OUT                  |
        END\s+OF\s+(THE\s+)?WORLD       |
        EVERYTHING\s+(IS\s+)?OVER       |
        MUST\s+(READ|SEE|KNOW)          |
        YOU\s+WON.T\s+BELIEVE           |
        SECRET(S|LY)?                   |
        HIDDEN\s+(TRUTH|AGENDA|PLAN)    |
        THEY\s+DON.T\s+WANT\s+YOU      |
        EXPOSE[DS]?                     |
        REVEALED?                       |   # overused in clickbait
        UNPRECEDENTED                   |   # overused in financial panic pieces
        HISTORIC(AL)?\s+(CRASH|COLLAPSE|PANIC)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Contradiction & retraction language ──────────────────────────────────────
# These phrases signal that a story is disputed or contradicts prior reporting.
# A contradicted article should not trigger a trade before the dust settles.
_CONTRADICTION_PATTERNS: re.Pattern = re.compile(
    r"""
    \b(
        contradicts                     |
        contradicting                   |
        contradicted                    |
        disputes?                       |   # "disputes", "dispute"
        disputed\s+by                   |
        refutes?                        |
        refuted\s+by                    |
        denies?\s+(the\s+)?report       |
        denied\s+by                     |
        walks?\s+back                   |
        walked\s+back                   |
        walks?\s+it\s+back              |
        reverses?\s+(course|position|stance)    |
        reversed\s+course               |
        U-turn                          |
        rowing\s+back                   |
        previous\s+reports?\s+(were\s+)?(wrong|incorrect|inaccurate) |
        despite\s+previous\s+reports?   |
        contrary\s+to\s+(earlier|previous)\s+reports? |
        correction\s*:                  |
        retraction\s*:                  |
        clarification\s*:
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Institutional / neutral financial wording (POSITIVE trust signal) ────────
# These patterns are characteristic of wire-service financial reporting.
# Presence of these phrases is a mild positive trust indicator.
_INSTITUTIONAL_PATTERNS: re.Pattern = re.compile(
    r"""
    \b(
        basis\s+points?                 |
        year[-\s]over[-\s]year          |
        quarter[-\s]over[-\s]quarter    |
        according\s+to\s+(the\s+)?(fed|ecb|imf|bis|treasury|sec|company) |
        in\s+a\s+statement              |
        said\s+in\s+a\s+(press\s+)?release |
        confirmed\s+by                  |
        official\s+(data|figures?|statement|report) |
        (government|central\s+bank)\s+(data|report|statement|figures?) |
        (q[1-4]|fiscal\s+(year|quarter))\s+20\d\d |
        (earnings|revenue|net\s+income)\s+(rose|fell|grew|declined) |
        reported\s+a\s+(profit|loss|gain|decline) |
        (above|below)\s+(analyst|consensus|street)\s+estimates?
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Social media URL patterns ─────────────────────────────────────────────────
# Pre-compiled for O(1) matching per article.
_SOCIAL_URL_PATTERN: re.Pattern = re.compile(
    r"(twitter\.com|x\.com|t\.co|facebook\.com|fb\.com|"
    r"reddit\.com|telegram\.org|t\.me|tiktok\.com|instagram\.com)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 1  —  PUBLIC API
#
# ══════════════════════════════════════════════════════════════════════════════

def validate_articles(articles: list[dict]) -> list[dict]:
    """
    PUBLIC ENTRY POINT
    ──────────────────
    Runs credibility validation and misinformation risk scoring over the full
    article list received from duplicate_filter.py.

    Designed for direct drop-in at the post-filter plugin hook:

        articles = validate_articles(articles)

    Each article is enriched in-place (on a copy) with seven new fields:

        validation_score            float   [0.0–1.0] composite verdict
        trust_score                 float   [0.0–1.0] source + structural trust
        misinformation_risk         float   [0.0–1.0] risk signal (0=clean)
        verification_status         str     VERIFIED | PROBABLE | UNVERIFIED | SUSPECT
        is_high_confidence_news     bool    True if score >= HIGH_CONFIDENCE_THRESHOLD
        needs_manual_review         bool    True if score < MANUAL_REVIEW_THRESHOLD
        risk_reasons                list    Human-readable risk factor descriptions

    Parameters
    ──────────
    articles : list[dict]
        Enriched article dicts from duplicate_filter.py.
        Malformed dicts, None entries, and empty lists are handled safely.

    Returns
    ───────
    list[dict]
        Same articles with validation fields injected.
        Article order is preserved.
        Malformed inputs are passed through with safe default scores.
    """
    wall_start = time.perf_counter()

    _log(1, "VALIDATOR", f"Starting credibility validation · {len(articles)} articles")

    if not articles or not isinstance(articles, list):
        _log(1, "VALIDATOR", "No articles to validate — returning empty list")
        return []

    validated = []
    for article in articles:
        if not isinstance(article, dict):
            _log(3, "SKIP", f"Non-dict entry skipped: {type(article)}")
            continue
        enriched = _validate_single_article(article)
        validated.append(enriched)

    wall_elapsed = round(time.perf_counter() - wall_start, 4)

    # ── PLUGIN HOOK: post_validation ──────────────────────────────────────────
    # Future: sqlite_store.persist_validation(validated, run_id)
    #         ai_reasoning_layer.explain(validated)
    #         alert_router.flag_manual_review(validated)

    _log_validation_summary(validated, wall_elapsed)

    return validated


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 2  —  COMPOSITE SCORING ORCHESTRATOR
#
#  Single function coordinates all sub-scorers and assembles the final
#  validation verdict.  Each sub-scorer returns a float in [0.0, 1.0]
#  and an optional list of risk_reason strings.
#
#  COMPOSITE FORMULA:
#      raw = (trust_score × TRUST_WEIGHT) − (misinformation_risk × MISINFO_WEIGHT)
#      validation_score = clamp(raw, 0.0, 1.0)
#
#  The subtraction model means misinformation risk actively SUPPRESSES
#  even high-trust articles — a Reuters article with rumor language still
#  gets penalized, because the language choice itself is a red flag.
#
# ══════════════════════════════════════════════════════════════════════════════

def _validate_single_article(article: dict) -> dict:
    """
    Runs the full validation pipeline for ONE article.

    Coordinates all sub-scorers, computes the composite score,
    assigns verification status, and injects all output fields.

    Always returns a dict — never raises.

    Parameters
    ──────────
    article : dict   One article dict from duplicate_filter.py.

    Returns
    ───────
    dict  → the same article with seven validation fields injected.
    """
    risk_reasons: list[str] = []

    # ── LAYER 3: trust scoring ────────────────────────────────────────────────
    trust_score, trust_reasons = _score_trust(article)

    # ── LAYER 4: misinformation risk scoring ─────────────────────────────────
    misinfo_risk, risk_factors = _score_misinformation_risk(article)
    risk_reasons.extend(risk_factors)

    # ── Composite validation score ────────────────────────────────────────────
    raw_score       = (trust_score * TRUST_WEIGHT) - (misinfo_risk * MISINFO_WEIGHT)
    validation_score = _clamp(raw_score, 0.0, 1.0)
    validation_score = round(validation_score, 4)
    trust_score      = round(trust_score, 4)
    misinfo_risk     = round(misinfo_risk, 4)

    # ── Verification status classification ───────────────────────────────────
    verification_status   = _classify_verification_status(validation_score)
    is_high_confidence    = validation_score >= HIGH_CONFIDENCE_THRESHOLD
    needs_manual_review   = validation_score < MANUAL_REVIEW_THRESHOLD

    _log(2, "SCORE",
         f"[{_safe_str(article.get('source'))[:22]:<22}] "
         f"val={validation_score:.3f}  "
         f"trust={trust_score:.3f}  "
         f"risk={misinfo_risk:.3f}  "
         f"→ {verification_status}")

    if risk_reasons:
        for reason in risk_reasons:
            _log(3, "RISK", reason)

    # ── Inject output fields ──────────────────────────────────────────────────
    # We mutate a shallow copy so the upstream reference is never touched.
    result = dict(article)
    result["validation_score"]         = validation_score
    result["trust_score"]              = trust_score
    result["misinformation_risk"]      = misinfo_risk
    result["verification_status"]      = verification_status
    result["is_high_confidence_news"]  = is_high_confidence
    result["needs_manual_review"]      = needs_manual_review
    result["risk_reasons"]             = risk_reasons

    return result


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 3  —  TRUST SCORING
#
#  Builds the trust_score from six independent sub-components.
#  Each sub-component returns a float in [0.0, 1.0] and is weighted
#  by the TRUST_W_* constants defined in the configuration section.
#
#  COMPONENTS:
#  ─────────────────────────────────────────────────────────────────
#  source_tier_score     Outlet pedigree classification (Tier 1–5)
#  cluster_depth_score   Multi-source confirmation depth from dedup layer
#  credibility_score     credibility_boost field from duplicate_filter.py
#  timestamp_score       Timestamp presence and (future) recency
#  summary_score         Summary presence and minimum content substance
#  wording_score         Institutional language quality signal
#
# ══════════════════════════════════════════════════════════════════════════════

def _score_trust(article: dict) -> tuple[float, list[str]]:
    """
    Computes the composite trust_score for one article.

    Each sub-component contributes its weighted share to the total.
    Sub-components are independently calculated — a failure in one
    does not affect the others.

    Returns
    ───────
    tuple[float, list[str]]
        float     → trust_score in [0.0, 1.0]
        list[str] → trust enhancement notes (for future explainability layer)
    """
    notes: list[str] = []

    source_name = _safe_str(article.get("source"))
    title       = _safe_str(article.get("title"))
    summary     = _safe_str(article.get("summary"))

    # ── Sub-component 1: Source tier ─────────────────────────────────────────
    tier, tier_score = _classify_source_tier(source_name)
    weighted_tier    = tier_score * TRUST_W_SOURCE_TIER
    if tier in ("tier_1", "tier_2"):
        notes.append(f"High-pedigree source [{tier.upper()}]: {source_name}")

    # ── Sub-component 2: Cluster depth ───────────────────────────────────────
    cluster_score  = _score_cluster_depth(article)
    weighted_cluster = cluster_score * TRUST_W_CLUSTER_DEPTH

    # ── Sub-component 3: credibility_boost from dedup layer ──────────────────
    raw_boost        = _safe_float(article.get("credibility_boost"), fallback=1.0)
    # Normalize: boost 1.0 = 0.0 additional trust; boost 2.5 = full contribution.
    # Maps [1.0, MAX_BOOST] → [0.0, 1.0] linearly.
    max_boost        = 2.5
    cred_component   = _clamp((raw_boost - 1.0) / (max_boost - 1.0), 0.0, 1.0)
    weighted_cred    = cred_component * TRUST_W_CREDIBILITY

    # ── Sub-component 4: Timestamp validity ──────────────────────────────────
    ts_score      = _score_timestamp(article)
    weighted_ts   = ts_score * TRUST_W_TIMESTAMP

    # ── Sub-component 5: Summary substance ───────────────────────────────────
    summ_score    = _score_summary_substance(summary)
    weighted_summ = summ_score * TRUST_W_SUMMARY

    # ── Sub-component 6: Institutional wording quality ───────────────────────
    wording_score    = _score_institutional_wording(title, summary)
    weighted_wording = wording_score * TRUST_W_WORDING

    # ── Composite trust score ─────────────────────────────────────────────────
    trust = (
        weighted_tier    +
        weighted_cluster +
        weighted_cred    +
        weighted_ts      +
        weighted_summ    +
        weighted_wording
    )

    _log(3, "TRUST",
         f"tier={tier_score:.2f} cluster={cluster_score:.2f} "
         f"cred={cred_component:.2f} ts={ts_score:.2f} "
         f"summ={summ_score:.2f} wording={wording_score:.2f} "
         f"→ trust={trust:.3f}")

    return _clamp(trust, 0.0, 1.0), notes


def _classify_source_tier(source_name: str) -> tuple[str, float]:
    """
    Assigns a tier classification and trust score to an outlet name.

    Uses substring matching against the tier sets defined in Layer 7.
    Case-insensitive.  First match in tier order wins.

    Parameters
    ──────────
    source_name : str   The "source" field from the article dict.

    Returns
    ───────
    tuple[str, float]
        str   → "tier_1" | "tier_2" | "tier_3" | "tier_4" | "tier_5"
        float → corresponding SOURCE_TIER_SCORES value
    """
    if not source_name:
        return "tier_5", SOURCE_TIER_SCORES["tier_5"]

    lower = source_name.lower()

    for tier_name, tier_set in [
        ("tier_1", TIER_1_SOURCES),
        ("tier_2", TIER_2_SOURCES),
        ("tier_3", TIER_3_SOURCES),
        ("tier_4", TIER_4_SOURCES),
    ]:
        for fragment in tier_set:
            if fragment in lower:
                return tier_name, SOURCE_TIER_SCORES[tier_name]

    return "tier_5", SOURCE_TIER_SCORES["tier_5"]


def _score_cluster_depth(article: dict) -> float:
    """
    Scores the multi-source confirmation depth using source_cluster from
    the duplicate_filter.py enrichment fields.

    Scoring formula:
        base = CLUSTER_BASE_SCORE (0.30)
        score = base + (unique_sources − 1) × CLUSTER_STEP
        capped at 1.0 at CLUSTER_CAP_SIZE unique sources

    A single-source article scores 0.30 (not zero — existence itself is signal).
    A 5-source cluster scores 1.0 (full institutional confidence from depth).

    Parameters
    ──────────
    article : dict   Article dict (may or may not have source_cluster field).

    Returns
    ───────
    float  → cluster depth trust component in [0.0, 1.0]
    """
    source_cluster = article.get("source_cluster")

    if not isinstance(source_cluster, list) or not source_cluster:
        # No cluster data — treat as single-source.
        return CLUSTER_BASE_SCORE

    unique_count = len(set(source_cluster))
    score = CLUSTER_BASE_SCORE + (unique_count - 1) * CLUSTER_STEP
    return _clamp(score, 0.0, 1.0)


def _score_timestamp(article: dict) -> float:
    """
    Scores timestamp quality as a trust signal.

    Rationale: Credible financial news always carries a timestamp.
    Missing timestamps are a mild but real red flag — they're common
    in scraped, aggregated, or poorly-maintained RSS feeds.

    Scoring:
        Timestamp present and non-empty → 1.0
        Timestamp absent or empty       → 0.0

    Future extension point:
        · Recency scoring: articles older than N hours lose trust in a
          live trading context.
        · Timezone validation: naive timestamps get a penalty.
        · Publish vs. ingest lag detection.

    Returns float in {0.0, 1.0}
    """
    ts = _safe_str(article.get("published_time"))
    return 1.0 if ts else 0.0


def _score_summary_substance(summary: str) -> float:
    """
    Scores summary presence and minimum content substance.

    Scoring:
        Summary >= SUMMARY_MIN_CHARS → 1.0  (substantial content present)
        Summary present but short    → 0.50 (token presence, low substance)
        Summary absent or empty      → 0.0

    Rationale: Credible institutional articles always include substantive
    lead paragraphs.  Very short summaries suggest scraper truncation or
    low-quality feed output.

    Parameters
    ──────────
    summary : str   Already safe-coerced string from the caller.

    Returns float in [0.0, 1.0]
    """
    if not summary:
        return 0.0
    if len(summary) >= SUMMARY_MIN_CHARS:
        return 1.0
    return 0.50


def _score_institutional_wording(title: str, summary: str) -> float:
    """
    Scores the linguistic quality of an article's wording as an
    institutional credibility signal.

    Detects the presence of formal financial reporting language
    (basis points, official statements, confirmed figures, etc.)
    which is characteristic of wire service and premium press output.

    Scoring:
        2+ institutional pattern matches → 1.0
        1 institutional pattern match   → 0.60
        0 matches                       → 0.25  (neutral, not zero)

    The 0.25 floor prevents this component from excessively penalizing
    articles that are simply headline-only or from markets not covered
    by the institutional pattern set (e.g. crypto-native sources).

    Parameters
    ──────────
    title   : str   Article title (safe-coerced by caller).
    summary : str   Article summary (safe-coerced by caller).

    Returns float in [0.0, 1.0]
    """
    combined = f"{title} {summary}"
    matches  = len(_INSTITUTIONAL_PATTERNS.findall(combined))

    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.60
    return 0.25


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 4  —  MISINFORMATION RISK SCORING
#
#  Builds the misinformation_risk score from seven independent risk
#  sub-components.  Risk score is INVERSE: 0.0 = clean, 1.0 = high risk.
#
#  COMPONENTS:
#  ─────────────────────────────────────────────────────────────────
#  rumor_risk          Unconfirmed/sources-say/allegedly phrases
#  emotional_risk      Sensational/panic language detection
#  caps_risk           ALL CAPS title or excessive caps ratio
#  exclamation_risk    Exclamation mark abuse
#  weak_source_risk    Low-pedigree outlet with no cluster confirmation
#  social_origin_risk  Social-media-first URL patterns
#  contradiction_risk  Language signaling disputed/retracted claims
#
#  Each component returns a float in [0.0, 1.0] where:
#    0.0 = no risk signal detected for this component
#    1.0 = maximum risk signal for this component
#
# ══════════════════════════════════════════════════════════════════════════════

def _score_misinformation_risk(article: dict) -> tuple[float, list[str]]:
    """
    Computes the composite misinformation_risk score for one article.

    Each sub-component runs independently.  A clean article returns 0.0.
    A maximally suspicious article returns 1.0.

    Returns
    ───────
    tuple[float, list[str]]
        float     → misinformation_risk in [0.0, 1.0]
        list[str] → human-readable risk factor descriptions for risk_reasons
    """
    risk_reasons: list[str] = []

    title   = _safe_str(article.get("title"))
    summary = _safe_str(article.get("summary"))
    link    = _safe_str(article.get("link"))
    source  = _safe_str(article.get("source"))
    combined = f"{title} {summary}"

    # ── Risk 1: Rumor / unverified claim language ─────────────────────────────
    rumor_risk, rumor_matches = _score_rumor_risk(combined)
    if rumor_risk > 0:
        risk_reasons.append(
            f"Rumor/unverified language detected: {rumor_matches[:3]}"
        )

    # ── Risk 2: Emotional / sensational language ──────────────────────────────
    emotional_risk, emotional_matches = _score_emotional_risk(combined)
    if emotional_risk > 0:
        risk_reasons.append(
            f"Sensational/emotional language: {emotional_matches[:3]}"
        )

    # ── Risk 3: ALL CAPS title abuse ──────────────────────────────────────────
    caps_risk = _score_caps_risk(title)
    if caps_risk > 0:
        risk_reasons.append("Title uses excessive uppercase — possible clickbait")

    # ── Risk 4: Exclamation mark abuse ───────────────────────────────────────
    exclamation_risk = _score_exclamation_risk(title)
    if exclamation_risk > 0:
        risk_reasons.append(
            f"Excessive exclamation marks in title ({title.count('!')} found)"
        )

    # ── Risk 5: Weak source with no cluster confirmation ─────────────────────
    weak_source_risk = _score_weak_source_risk(article, source)
    if weak_source_risk > 0:
        risk_reasons.append(
            f"Low-pedigree source with no multi-outlet confirmation: {source}"
        )

    # ── Risk 6: Social-media-first origin ────────────────────────────────────
    social_risk = _score_social_origin_risk(link)
    if social_risk > 0:
        risk_reasons.append(f"Article originates from social media platform: {link[:60]}")

    # ── Risk 7: Contradiction / retraction language ───────────────────────────
    contradiction_risk, contradiction_matches = _score_contradiction_risk(combined)
    if contradiction_risk > 0:
        risk_reasons.append(
            f"Contradiction/retraction language detected: {contradiction_matches[:3]}"
        )

    # ── Composite risk score ──────────────────────────────────────────────────
    risk = (
        rumor_risk         * RISK_W_RUMOR_PHRASES  +
        emotional_risk     * RISK_W_EMOTIONAL       +
        caps_risk          * RISK_W_CAPS_ABUSE      +
        exclamation_risk   * RISK_W_EXCLAMATION     +
        weak_source_risk   * RISK_W_WEAK_SOURCE     +
        social_risk        * RISK_W_SOCIAL_ORIGIN   +
        contradiction_risk * RISK_W_CONTRADICTION
    )

    _log(3, "RISK ",
         f"rumor={rumor_risk:.2f} emo={emotional_risk:.2f} "
         f"caps={caps_risk:.2f} excl={exclamation_risk:.2f} "
         f"weak={weak_source_risk:.2f} social={social_risk:.2f} "
         f"contra={contradiction_risk:.2f} → risk={risk:.3f}")

    return _clamp(risk, 0.0, 1.0), risk_reasons


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 5  —  CONTRADICTION HEURISTICS
#  LAYER 6  —  WORDING ANALYSIS SUB-SCORERS
#
#  All individual risk scoring functions.
#  Each has ONE job.  Each returns a float in [0.0, 1.0].
#
# ══════════════════════════════════════════════════════════════════════════════

def _score_rumor_risk(text: str) -> tuple[float, list[str]]:
    """
    Detects rumor, unverified claim, and anonymous-source language.

    Scoring:
        3+ distinct rumor phrases → 1.0
        2  distinct phrases       → 0.75
        1  distinct phrase        → 0.50
        0  matches                → 0.0

    Returns (risk_score, list_of_matched_phrases)
    """
    if not text:
        return 0.0, []

    matches = list(set(m.lower() for m in _RUMOR_PATTERNS.findall(text)))
    count   = len(matches)

    if count == 0:
        return 0.0, []
    if count == 1:
        return 0.50, matches
    if count == 2:
        return 0.75, matches
    return 1.0, matches


def _score_emotional_risk(text: str) -> tuple[float, list[str]]:
    """
    Detects sensational, fear-driven, and clickbait language.

    Scoring:
        3+ distinct emotional phrases → 1.0
        2  distinct phrases           → 0.65
        1  distinct phrase            → 0.35
        0  matches                    → 0.0

    A single "BREAKING" is a mild signal; three panic words compound the risk.

    Returns (risk_score, list_of_matched_phrases)
    """
    if not text:
        return 0.0, []

    matches = list(set(m.upper() for m in _EMOTIONAL_PATTERNS.findall(text)))
    count   = len(matches)

    if count == 0:
        return 0.0, []
    if count == 1:
        return 0.35, matches
    if count == 2:
        return 0.65, matches
    return 1.0, matches


def _score_caps_risk(title: str) -> float:
    """
    Detects ALL CAPS titles or excessive capitalization ratios.

    Two-test strategy:
        Test A: If the ENTIRE title is uppercase → score 1.0 immediately.
        Test B: If the ratio of uppercase to alphabetic chars > threshold → 0.70.

    Why both tests?  Test A catches "BREAKING: FED HIKES RATES".
    Test B catches mixed-caps abuse like "The FED Has SECRETLY Planned a CRASH".

    Returns float in [0.0, 1.0]
    """
    if not title:
        return 0.0

    alpha_chars = [c for c in title if c.isalpha()]
    if not alpha_chars:
        return 0.0

    # Test A: fully uppercase title
    if title == title.upper() and len(alpha_chars) > 5:
        return 1.0

    # Test B: caps ratio
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    caps_ratio  = upper_count / len(alpha_chars)

    if caps_ratio >= CAPS_RATIO_THRESHOLD:
        return 0.70

    return 0.0


def _score_exclamation_risk(title: str) -> float:
    """
    Detects exclamation mark abuse in article titles.

    Legitimate financial headlines almost never use exclamation marks.
    Even one is unusual.  Multiple is a strong clickbait signal.

    Scoring:
        > 2 exclamation marks → 1.0
        2 exclamation marks   → 0.60
        1 exclamation mark    → 0.25
        0 exclamation marks   → 0.0

    Returns float in [0.0, 1.0]
    """
    if not title:
        return 0.0

    count = title.count("!")

    if count == 0:
        return 0.0
    if count == 1:
        return 0.25
    if count == 2:
        return 0.60
    return 1.0


def _score_weak_source_risk(article: dict, source_name: str) -> float:
    """
    Scores the risk of a low-pedigree article that lacks multi-source backing.

    A Tier 4/5 article that is NOT confirmed by any other outlet is a
    meaningful misinformation risk vector.  The same article confirmed by
    Bloomberg and Reuters would have a large source_cluster — in that case
    this risk component scores 0.0 because cluster depth de-risks it.

    Logic:
        Tier 1/2 source           → 0.0  (inherently credible)
        Tier 3 + multi-source     → 0.0  (confirmed)
        Tier 3 + singleton        → 0.20 (mild signal)
        Tier 4 + multi-source     → 0.30 (confirmed but weak source)
        Tier 4 + singleton        → 0.75 (unconfirmed low-tier source)
        Tier 5 + any              → 0.90 (always high risk)

    Returns float in [0.0, 1.0]
    """
    tier, _ = _classify_source_tier(source_name)

    # Check if this article has multi-source backing from the dedup layer.
    is_confirmed = article.get("is_multi_source_confirmation", False)
    cluster_size = len(article.get("source_cluster") or [])

    if tier == "tier_1":
        return 0.0
    if tier == "tier_2":
        return 0.0
    if tier == "tier_3":
        return 0.0 if is_confirmed else 0.20
    if tier == "tier_4":
        return 0.30 if is_confirmed else 0.75
    # Tier 5
    return 0.90 if not is_confirmed else 0.60


def _score_social_origin_risk(link: str) -> float:
    """
    Detects articles whose canonical link points to a social media platform.

    Social-media-first "news" bypasses editorial review.  In financial markets,
    this pattern has historically preceded pump-and-dump schemes and coordinated
    misinformation campaigns.

    Scoring:
        Known social domain in link → 1.0
        No social signal            → 0.0

    Note: LinkedIn is intentionally excluded from _SOCIAL_URL_PATTERN
    because financial institutions (central banks, corporations) do
    legitimately publish official statements on LinkedIn.

    Returns float in {0.0, 1.0}
    """
    if not link:
        return 0.0

    return 1.0 if _SOCIAL_URL_PATTERN.search(link) else 0.0


def _score_contradiction_risk(text: str) -> tuple[float, list[str]]:
    """
    Detects language that signals disputed, retracted, or contradicted claims.

    Rationale: An article reporting that previous reports were wrong, or that
    a company denied a story, is NOT the same as breaking news.  Acting on
    an unresolved contradiction before the dust settles is a trading risk.
    This module flags it — the human or signal_engine.py decides what to do.

    Scoring:
        2+ contradiction phrases → 0.80
        1  contradiction phrase  → 0.50
        0  matches               → 0.0

    Returns (risk_score, list_of_matched_phrases)
    """
    if not text:
        return 0.0, []

    matches = list(set(m.lower() for m in _CONTRADICTION_PATTERNS.findall(text)))
    count   = len(matches)

    if count == 0:
        return 0.0, []
    if count == 1:
        return 0.50, matches
    return 0.80, matches


# ══════════════════════════════════════════════════════════════════════════════
#
#  VERIFICATION STATUS CLASSIFIER
#
# ══════════════════════════════════════════════════════════════════════════════

def _classify_verification_status(validation_score: float) -> str:
    """
    Maps a composite validation_score to a human-readable status string.

    VERIFIED    → Institutional-grade confidence.  Signal engine can act.
    PROBABLE    → Tradeable with normal risk management and position sizing.
    UNVERIFIED  → Requires corroboration from another source before action.
    SUSPECT     → Do not trade.  Flag for manual review immediately.

    Thresholds are defined in the configuration constants at the top of the file.

    Parameters
    ──────────
    validation_score : float   Composite score in [0.0, 1.0].

    Returns
    ───────
    str  → one of "VERIFIED" | "PROBABLE" | "UNVERIFIED" | "SUSPECT"
    """
    if validation_score >= THRESHOLD_VERIFIED:
        return "VERIFIED"
    if validation_score >= THRESHOLD_PROBABLE:
        return "PROBABLE"
    if validation_score >= THRESHOLD_UNVERIFIED:
        return "UNVERIFIED"
    return "SUSPECT"


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 8  —  SAFETY HELPERS
#
# ══════════════════════════════════════════════════════════════════════════════

def _safe_str(value, fallback: str = "") -> str:
    """Safely coerces any value to a stripped string.  Never raises."""
    if value is None:
        return fallback
    try:
        return str(value).strip()
    except Exception:
        return fallback


def _safe_float(value, fallback: float = 0.0) -> float:
    """Safely coerces any value to a float.  Never raises."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamps a float to [lo, hi].  Pure.  No side effects."""
    return max(lo, min(hi, value))


# ══════════════════════════════════════════════════════════════════════════════
#
#  LAYER 9  —  OBSERVABILITY
#
# ══════════════════════════════════════════════════════════════════════════════

def _log(level: int, tag: str, message: str) -> None:
    """
    Central log dispatcher.  All output in this module goes through here.

    Future swap:
        logging.getLogger("fake_news_validator").info(f"[{tag}] {message}")
    """
    if VERBOSITY >= level:
        print(f"  [{tag}] {message}")


def _log_validation_summary(articles: list[dict], elapsed: float) -> None:
    """
    Prints the validation run summary at VERBOSITY >= 1.

    Surfaces:
        · Status distribution (VERIFIED / PROBABLE / UNVERIFIED / SUSPECT)
        · High-confidence count
        · Manual review queue size
        · Top risk articles for human inspection
        · Pipeline latency
    """
    if VERBOSITY < 1:
        return

    status_counts: dict[str, int] = {
        "VERIFIED": 0, "PROBABLE": 0, "UNVERIFIED": 0, "SUSPECT": 0
    }
    high_conf_count     = 0
    manual_review_count = 0
    suspect_articles    = []

    for art in articles:
        status = art.get("verification_status", "UNVERIFIED")
        status_counts[status] = status_counts.get(status, 0) + 1

        if art.get("is_high_confidence_news"):
            high_conf_count += 1

        if art.get("needs_manual_review"):
            manual_review_count += 1

        if status == "SUSPECT":
            suspect_articles.append(art)

    total = len(articles)

    print()
    print("═" * 72)
    print("  CREDIBILITY VALIDATION  ·  RUN SUMMARY")
    print("═" * 72)
    print()
    print("  VOLUME")
    print(f"  {'Articles validated':<34}: {total}")
    print(f"  {'High-confidence articles':<34}: {high_conf_count}")
    print(f"  {'Manual review queue':<34}: {manual_review_count}")
    print()
    print("  VERIFICATION STATUS DISTRIBUTION")

    bar_max = max(status_counts.values(), default=1)
    status_order = ["VERIFIED", "PROBABLE", "UNVERIFIED", "SUSPECT"]
    status_icons = {"VERIFIED": "✔", "PROBABLE": "◈", "UNVERIFIED": "⚠", "SUSPECT": "✗"}

    for status in status_order:
        count   = status_counts.get(status, 0)
        pct     = round((count / total) * 100, 1) if total else 0.0
        bar_len = int((count / bar_max) * 28) if bar_max else 0
        bar     = "█" * bar_len
        icon    = status_icons[status]
        print(f"  {icon} {status:<12}  {count:>4}  ({pct:>5.1f}%)  {bar}")

    if suspect_articles:
        print()
        print("  SUSPECT ARTICLES  (do not trade — manual review required)")
        print("  " + "─" * 56)
        for art in suspect_articles[:5]:
            score   = art.get("validation_score", 0.0)
            source  = _safe_str(art.get("source"))[:25]
            title   = _safe_str(art.get("title"))[:50]
            reasons = art.get("risk_reasons", [])
            print(f"  ✗ [{score:.3f}] [{source}] {title}")
            for reason in reasons[:2]:
                print(f"      └─ {reason}")

    print()
    print("  LATENCY")
    print(f"  {'Validation pipeline time':<34}: {elapsed}s")
    print()
    print("═" * 72)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#
#  DIRECT EXECUTION — SMOKE TEST
#
#  python fake_news_validator.py
#
#  Runs a synthetic battery of articles spanning every risk category.
#  Designed to exercise every scoring path and produce human-readable output.
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _make(
        title          : str,
        source         : str,
        summary        : str | None   = None,
        link           : str          = "https://example.com/article",
        priority       : int          = 3,
        published_time : str | None   = "2025-04-09T14:00:00+00:00",
        credibility_boost              : float      = 1.0,
        source_cluster : list[str]    = None,
        is_multi_source_confirmation   : bool       = False,
        tags           : list[str]    = None,
    ) -> dict:
        return {
            "source"                    : source,
            "category"                  : "macro",
            "priority"                  : priority,
            "region"                    : "global",
            "title"                     : title,
            "link"                      : link,
            "summary"                   : summary,
            "published_time"            : published_time,
            "tags"                      : tags or ["macro"],
            "dedupe_hash"               : "abc123",
            "credibility_boost"         : credibility_boost,
            "source_cluster"            : source_cluster or [source],
            "is_multi_source_confirmation": is_multi_source_confirmation,
            "duplicate_count"           : len(source_cluster) if source_cluster else 1,
            "cluster_size_score"        : 1.0 if source_cluster else 0.125,
            "confidence_score"          : None,
            "impact_score"              : None,
            "regime_tag"                : None,
            "risk_flag"                 : None,
        }

    test_articles = [

        # ── Case 1: Ideal institutional article ──────────────────────────────
        _make(
            title    = "Fed raises rates by 25 basis points; official statement released",
            source   = "Reuters",
            summary  = ("The Federal Reserve raised its benchmark rate by 25 basis points "
                        "to 5.50%, according to an official statement. The decision was "
                        "unanimous among voting members."),
            priority = 1,
            credibility_boost = 1.6,
            source_cluster    = ["Reuters", "Bloomberg", "AP News"],
            is_multi_source_confirmation = True,
        ),

        # ── Case 2: Bloomberg CPI with multi-source confirmation ──────────────
        _make(
            title    = "CPI inflation data comes in at 3.8% year-over-year, above consensus",
            source   = "Bloomberg",
            summary  = ("March CPI rose 3.8% year-over-year, beating the street estimate "
                        "of 3.5%, according to official Bureau of Labor Statistics data."),
            priority = 1,
            credibility_boost = 1.9,
            source_cluster    = ["Bloomberg", "Reuters", "FT", "WSJ"],
            is_multi_source_confirmation = True,
        ),

        # ── Case 3: Rumor-heavy article from a mid-tier outlet ────────────────
        _make(
            title   = "Sources say Fed may be considering emergency rate cut, allegedly",
            source  = "MarketWatch",
            summary = "People familiar with the matter say the Fed is reportedly in "
                      "unconfirmed discussions about an unscheduled rate decision.",
            priority = 3,
        ),

        # ── Case 4: Emotional/clickbait title from a weak outlet ──────────────
        _make(
            title   = "BREAKING: MARKET CRASH INCOMING! PANIC NOW! CATASTROPHIC COLLAPSE!!!",
            source  = "ZeroHedge",
            summary = None,
            priority = 5,
            published_time = None,
        ),

        # ── Case 5: Social media origin ───────────────────────────────────────
        _make(
            title   = "Massive layoffs coming at major bank — insider",
            source  = "Twitter/X",
            summary = "A post circulating on social media claims a major US bank will "
                      "announce thousands of layoffs next week.",
            link    = "https://twitter.com/financialgossip/status/1234567890",
            priority = 5,
        ),

        # ── Case 6: Contradiction / retraction language ───────────────────────
        _make(
            title   = "Company contradicts earlier report, walks back merger announcement",
            source  = "CNBC",
            summary = ("The company has denied the report, issuing a clarification: "
                       "the story was inaccurate and previous reports were incorrect."),
            priority = 2,
        ),

        # ── Case 7: FT article, solid but no cluster ─────────────────────────
        _make(
            title   = "ECB signals possible rate cuts in Q3 2025 as inflation eases",
            source  = "Financial Times",
            summary = ("ECB officials said in a press conference that the central bank "
                       "is considering rate cuts in Q3 2025, per official data."),
            priority = 2,
        ),

        # ── Case 8: Structurally empty / malformed ────────────────────────────
        {"title": None, "source": None},     # Minimal malformed
        {},                                   # Fully empty dict
        _make(
            title   = "[No Title]",
            source  = "Unknown Blog",
            summary = None,
            link    = "",
            priority = 5,
            published_time = None,
        ),
    ]

    print()
    print("═" * 72)
    print("  FAKE NEWS VALIDATOR  ·  SMOKE TEST")
    print(f"  Input: {len(test_articles)} articles")
    print("═" * 72)

    results = validate_articles(test_articles)

    print()
    print("  DETAILED RESULTS")
    print("  " + "─" * 68)

    for i, art in enumerate(results, 1):
        status   = art.get("verification_status", "N/A")
        val      = art.get("validation_score",    0.0)
        trust    = art.get("trust_score",         0.0)
        risk     = art.get("misinformation_risk", 0.0)
        hi_conf  = art.get("is_high_confidence_news", False)
        review   = art.get("needs_manual_review",     False)
        title    = _safe_str(art.get("title"))[:55]
        source   = _safe_str(art.get("source"))[:20]
        reasons  = art.get("risk_reasons", [])

        status_icon = {"VERIFIED": "✔", "PROBABLE": "◈",
                       "UNVERIFIED": "⚠", "SUSPECT": "✗"}.get(status, "?")

        print(f"  [{i:02}] {status_icon} {status:<12} val={val:.3f}  "
              f"trust={trust:.3f}  risk={risk:.3f}")
        print(f"       Source   : {source}")
        print(f"       Title    : {title}")
        print(f"       HiConf   : {hi_conf}   NeedsReview: {review}")
        if reasons:
            for r in reasons:
                print(f"       ⚑ {r}")
        print()

    print("═" * 72)
    print(f"  TOTAL VALIDATED : {len(results)}")
    print("═" * 72)
    print()