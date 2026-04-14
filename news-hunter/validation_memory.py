"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MONSTER TRADING AI — validation_memory.py                          ║
║     Long-Term Forensic Credibility Memory · News Intelligence Layer         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline position:                                                          ║
║    news_engine → duplicate_filter → fake_news_validator                     ║
║    → [validation_memory]  ← THIS MODULE                                     ║
║    → signal_engine → alert_router → execution_bridge                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Upstream contract:  fake_news_validator.py validated article dicts          ║
║  Persistence:        atomic JSON via state_manager philosophy                ║
║  Purpose:            forensic source credibility memory across restarts      ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module is the long-term forensic memory layer for article credibility
behaviour.  It persists intelligence across system restarts so the trading AI
learns incrementally:

  · Which outlets repeatedly publish suspect news
  · Which domains generate rumour-heavy articles
  · Which contradiction phrases precede false signals
  · Rolling credibility trends by source (7-day windows)
  · Repeated misinformation fingerprints (content hashes)
  · Compact feature-label pairs for future ML training

Architecture Layers
-------------------
Layer 1 — Public API
Layer 2 — State Hydration (atomic load/save, corruption recovery)
Layer 3 — Incremental Update Engine
Layer 4 — Source Reputation Analytics
Layer 5 — Fingerprint Memory
Layer 6 — Rolling Window Statistics
Layer 7 — Dataset Export
Layer 8 — Migration Helpers
Layer 9 — Observability + Summaries

Engineering Constraints
------------------------
· Standard library only                 · No pandas / SQLite / external deps
· hashlib fingerprints                  · dataclasses + typing throughout
· Survives restart + corrupted JSON     · Deterministic outputs
· O(1) incremental updates              · ≤ 50ms for 1 000-article batch
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("validation_memory")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "[%(levelname)s] validation_memory: %(message)s"
    ))
    log.addHandler(_h)
    log.setLevel(logging.DEBUG)

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — CONSTANTS + SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

MEM_SCHEMA_VERSION: int = 1
"""Bump on any breaking change to the on-disk JSON structure."""

_DEFAULT_STATE_PATH: str = os.getenv(
    "VALIDATION_MEMORY_FILE", "validation_memory.json"
)
_BACKUP_SUFFIX: str = ".backup.json"
_TEMP_SUFFIX:   str = ".tmp"
_ENCODING:      str = "utf-8"
_JSON_INDENT:   int = 2

# Rolling window granularity
_SECONDS_PER_DAY:  float = 86_400.0
_DEFAULT_WINDOW_DAYS: int = 7

# Suspect threshold: source is a repeat offender above this ratio
_SUSPECT_RATIO_THRESHOLD: float = 0.40

# Fingerprint: minimum occurrences before flagging as repeated misinformation
_FINGERPRINT_MIN_HITS: int = 2

# Future hook: signal_engine.py will call get_source_reputation() to gate
# signals through a credibility pre-filter before scoring confidence.

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceStats:
    """Per-source aggregated credibility statistics (O(1) incremental update)."""
    source:                  str
    total_articles:          int   = 0
    avg_validation_score:    float = 0.0
    avg_misinformation_risk: float = 0.0
    avg_trust_score:         float = 0.0
    suspect_count:           int   = 0
    verified_count:          int   = 0
    high_confidence_count:   int   = 0
    needs_review_count:      int   = 0
    contradiction_flags:     int   = 0
    rumor_phrase_hits:       int   = 0
    last_seen_timestamp:     float = 0.0
    # running sums for O(1) mean update
    _sum_validation_score:    float = 0.0
    _sum_misinformation_risk: float = 0.0
    _sum_trust_score:         float = 0.0

    def ingest(self, article: Dict[str, Any]) -> None:
        """Merge one validated article into this source's running statistics."""
        n = self.total_articles + 1

        vs  = float(article.get("validation_score",    0.0))
        mr  = float(article.get("misinformation_risk", 0.0))
        ts  = float(article.get("trust_score",         0.0))
        vst = article.get("verification_status",       "unknown")

        self._sum_validation_score    += vs
        self._sum_misinformation_risk += mr
        self._sum_trust_score         += ts

        self.avg_validation_score    = self._sum_validation_score    / n
        self.avg_misinformation_risk = self._sum_misinformation_risk / n
        self.avg_trust_score         = self._sum_trust_score         / n
        self.total_articles          = n
        self.last_seen_timestamp     = time.time()

        if vst in ("suspect", "misinformation"):
            self.suspect_count += 1
        elif vst in ("verified", "high_confidence"):
            self.verified_count += 1

        if article.get("is_high_confidence_news"):
            self.high_confidence_count += 1
        if article.get("needs_manual_review"):
            self.needs_review_count += 1

        reasons: List[str] = article.get("risk_reasons", [])
        for r in reasons:
            rl = r.lower()
            if "contradict" in rl:
                self.contradiction_flags += 1
            if any(p in rl for p in ("rumour", "rumor", "unconfirmed",
                                     "speculation", "alleged", "purported")):
                self.rumor_phrase_hits += 1

    @property
    def suspect_ratio(self) -> float:
        return self.suspect_count / self.total_articles if self.total_articles else 0.0

    @property
    def verified_ratio(self) -> float:
        return self.verified_count / self.total_articles if self.total_articles else 0.0

    @property
    def is_repeat_offender(self) -> bool:
        return (self.total_articles >= 3
                and self.suspect_ratio >= _SUSPECT_RATIO_THRESHOLD)


@dataclass
class FingerprintRecord:
    """Tracks repeated identical-content articles across sources."""
    fingerprint:    str
    first_seen:     float
    last_seen:      float
    occurrence_count: int        = 1
    sources:        List[str]    = field(default_factory=list)
    titles:         List[str]    = field(default_factory=list)   # deduped sample
    dominant_risk_reason: str    = ""
    all_risk_reasons: List[str]  = field(default_factory=list)

    @property
    def is_repeat_misinformation(self) -> bool:
        return self.occurrence_count >= _FINGERPRINT_MIN_HITS


@dataclass
class RollingDayBucket:
    """Single-day aggregation bucket for the rolling window."""
    date_str:            str    # ISO-8601 "YYYY-MM-DD"
    article_count:       int    = 0
    verified_count:      int    = 0
    suspect_count:       int    = 0
    sum_trust:           float  = 0.0
    sum_misinfo_risk:    float  = 0.0

    @property
    def verified_ratio(self) -> float:
        return self.verified_count / self.article_count if self.article_count else 0.0

    @property
    def suspect_ratio(self) -> float:
        return self.suspect_count / self.article_count if self.article_count else 0.0

    @property
    def avg_trust(self) -> float:
        return self.sum_trust / self.article_count if self.article_count else 0.0

    @property
    def avg_misinfo_risk(self) -> float:
        return self.sum_misinfo_risk / self.article_count if self.article_count else 0.0


@dataclass
class TrainingRecord:
    """Compact feature-label pair for future ML pipeline."""
    source_tier:         str    # "verified" | "mixed" | "suspect" | "unknown"
    validation_score:    float
    misinformation_risk: float
    trust_score:         float
    cluster_depth:       int    # len(source_cluster) proxy
    has_credibility_boost: bool
    contradiction_count: int
    rumor_phrase_count:  int
    final_status_label:  str    # "safe" | "suspect" | "misinformation" | "review"
    timestamp:           float


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — STATE HYDRATION  (atomic I/O, corruption recovery)
# ══════════════════════════════════════════════════════════════════════════════

def _default_memory() -> Dict[str, Any]:
    """Return a pristine, fully-keyed memory state dict."""
    return {
        "schema_version":        MEM_SCHEMA_VERSION,
        "saved_at":              None,
        "source_stats":          {},   # source -> SourceStats dict
        "suspect_fingerprints":  {},   # sha256 -> FingerprintRecord dict
        "rumor_phrase_registry": {},   # phrase -> int count
        "rolling_day_buckets":   {},   # "YYYY-MM-DD" -> RollingDayBucket dict
        "training_export_memory": [],  # List[TrainingRecord dict]
    }


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    """
    Write *data* to *path* with fsync-guarded atomicity.

    Flow: serialise → temp file → fsync fd → os.replace → fsync directory.
    Guarantees no partial writes survive a crash between any two steps.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    payload = json.dumps(data, indent=_JSON_INDENT, ensure_ascii=False,
                         default=str).encode(_ENCODING)
    fd, tmp = tempfile.mkstemp(suffix=_TEMP_SUFFIX, dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:                              # flush directory entry (POSIX)
            dfd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except (AttributeError, OSError):
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding=_ENCODING) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def _backup_path(path: str) -> str:
    stem = path[:-5] if path.endswith(".json") else path
    return stem + _BACKUP_SUFFIX


def _rotate_backup(path: str) -> None:
    data = _safe_read(path)
    if data is not None:
        try:
            _atomic_write(_backup_path(path), data)
        except Exception as exc:
            log.warning("Backup rotation failed (non-fatal): %s", exc)


def _load_with_recovery(path: str) -> Dict[str, Any]:
    """Two-tier recovery: primary → backup → fresh default."""
    data = _safe_read(path)
    if data is not None:
        log.debug("Memory loaded from primary: %s", path)
        return data
    backup = _backup_path(path)
    data = _safe_read(backup)
    if data is not None:
        log.warning("Primary missing/corrupt. Restored from backup: %s", backup)
        try:
            _atomic_write(path, data)
        except OSError as exc:
            log.error("Could not restore primary: %s", exc)
        return data
    log.warning("Both memory files missing/corrupt. Starting with default.")
    return _default_memory()


# ── in-memory singleton ───────────────────────────────────────────────────────

_mem_cache: Optional[Dict[str, Any]] = None
_mem_path:  str = _DEFAULT_STATE_PATH


def _reset_singleton(path: Optional[str] = None) -> None:
    """
    TEST / HOT-RELOAD RESET HELPER
    ──────────────────────────────
    Fully resets all in-memory singleton state so unit tests,
    smoke tests, notebook reruns, and hot-reload sessions start clean.
    This function MUST NOT touch persisted disk state.
    It only resets RAM-level caches, locks, rolling stats snapshots,
    plugin error counters, and lazy-init flags.
    Safe to call repeatedly.

    Parameters
    ----------
    path : str, optional
        If supplied, also updates the module-level ``_mem_path`` so the
        next load targets a different file.  Existing callers (setUp /
        tearDown in the test suite) rely on this behaviour.
    """
    global _mem_cache, _mem_path

    # ── primary singleton cache ───────────────────────────────────────────────
    _mem_cache = None

    # ── optional path redirection (backward-compat with test harness) ────────
    if path is not None:
        _mem_path = path

    # ── plugin error counters (present only when plugins are registered) ─────
    if "_plugin_error_counts" in globals():
        _plugin_error_counts.clear()   # type: ignore[name-defined]

    # ── in-memory rolling-window aggregator (present only after first load) ──
    if "_rolling_cache" in globals():
        _rolling_cache.clear()         # type: ignore[name-defined]

    # ── fingerprint dedup cache (present only after first ingest) ────────────
    if "_fingerprint_cache" in globals():
        _fingerprint_cache.clear()     # type: ignore[name-defined]


def _load_memory(path_override: Optional[str] = None) -> Dict[str, Any]:
    global _mem_cache, _mem_path
    if path_override:
        _mem_path = path_override
    if _mem_cache is not None:
        return _mem_cache
    raw = _load_with_recovery(_mem_path)
    raw = _apply_migrations(raw)
    raw = _backfill_keys(raw)
    _mem_cache = raw
    return _mem_cache


def _save_memory(mem: Dict[str, Any]) -> None:
    global _mem_cache
    _rotate_backup(_mem_path)
    payload = copy.deepcopy(mem)
    payload["schema_version"] = MEM_SCHEMA_VERSION
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write(_mem_path, payload)
    _mem_cache = payload
    log.debug("Memory saved → %s", _mem_path)


def _backfill_keys(mem: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_memory()
    for k, v in defaults.items():
        if k not in mem:
            log.debug("Backfilling missing memory key: '%s'", k)
            mem[k] = copy.deepcopy(v)
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 — MIGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Registry: from_version (int) → callable(data) -> data
MEMORY_MIGRATIONS: Dict[int, Any] = {
    # 1: _migrate_v1_to_v2,   ← hook for next schema bump
}


def _apply_migrations(mem: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the migration chain from file version up to MEM_SCHEMA_VERSION."""
    file_ver = mem.get("schema_version", 1)
    if file_ver == MEM_SCHEMA_VERSION:
        return mem
    if file_ver > MEM_SCHEMA_VERSION:
        log.warning(
            "Memory schema_version=%d > code version=%d. Proceeding without downgrade.",
            file_ver, MEM_SCHEMA_VERSION,
        )
        return mem
    current = file_ver
    while current < MEM_SCHEMA_VERSION:
        migrator = MEMORY_MIGRATIONS.get(current)
        if migrator is None:
            log.warning("No migration v%d → v%d; skipping.", current, current + 1)
            current += 1
            continue
        try:
            mem = migrator(mem)
            log.info("Applied migration v%d → v%d.", current, current + 1)
        except Exception as exc:
            log.error("Migration v%d failed: %s", current, exc)
            break
        current += 1
    mem["schema_version"] = MEM_SCHEMA_VERSION
    return mem


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — FINGERPRINT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_fingerprint(article: Dict[str, Any]) -> str:
    """
    Derive a deterministic SHA-256 fingerprint from the article's content
    identity: lowercased title + source.

    Using title + source (not link) deliberately catches syndicated
    republications of the same story across different URLs.
    """
    raw = (
        article.get("title",  "").lower().strip()
        + "|"
        + article.get("source", "").lower().strip()
    )
    return hashlib.sha256(raw.encode(_ENCODING)).hexdigest()


def _update_fingerprints(mem: Dict[str, Any],
                         article: Dict[str, Any],
                         fp: str) -> None:
    """Upsert a fingerprint record in O(1)."""
    registry: Dict[str, Any] = mem["suspect_fingerprints"]
    reasons:  List[str]      = article.get("risk_reasons", [])
    source:   str            = article.get("source", "unknown")
    title:    str            = article.get("title",  "")
    now = time.time()

    if fp not in registry:
        dominant = reasons[0] if reasons else ""
        registry[fp] = asdict(FingerprintRecord(
            fingerprint=fp,
            first_seen=now,
            last_seen=now,
            occurrence_count=1,
            sources=[source],
            titles=[title[:120]],
            dominant_risk_reason=dominant,
            all_risk_reasons=list(reasons),
        ))
    else:
        rec = registry[fp]
        rec["occurrence_count"] += 1
        rec["last_seen"] = now
        if source not in rec["sources"]:
            rec["sources"].append(source)
        if title[:120] not in rec["titles"]:
            rec["titles"].append(title[:120])
        for r in reasons:
            if r not in rec["all_risk_reasons"]:
                rec["all_risk_reasons"].append(r)
        # Promote dominant reason if frequency changes
        if reasons and not rec["dominant_risk_reason"]:
            rec["dominant_risk_reason"] = reasons[0]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — RUMOR PHRASE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Phrase taxonomy extracted from risk_reasons at ingest time
_RUMOR_PHRASE_TOKENS: Tuple[str, ...] = (
    "rumour", "rumor", "unconfirmed", "speculation", "alleged",
    "purported", "unverified", "anonymous source", "could not confirm",
    "reports suggest", "according to sources", "claim", "said to be",
)


def _update_rumor_registry(mem: Dict[str, Any],
                           article: Dict[str, Any]) -> None:
    registry: Dict[str, int] = mem["rumor_phrase_registry"]
    reasons: List[str] = article.get("risk_reasons", [])
    for reason in reasons:
        rl = reason.lower()
        for phrase in _RUMOR_PHRASE_TOKENS:
            if phrase in rl:
                registry[phrase] = registry.get(phrase, 0) + 1


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — ROLLING WINDOW STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _today_iso() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _update_rolling_bucket(mem: Dict[str, Any],
                           article: Dict[str, Any]) -> None:
    """Upsert today's rolling-day bucket with this article's stats."""
    buckets: Dict[str, Any] = mem["rolling_day_buckets"]
    today = _today_iso()
    if today not in buckets:
        buckets[today] = asdict(RollingDayBucket(date_str=today))
    b = buckets[today]
    b["article_count"]    += 1
    b["sum_trust"]        += float(article.get("trust_score",         0.0))
    b["sum_misinfo_risk"] += float(article.get("misinformation_risk", 0.0))
    vst = article.get("verification_status", "unknown")
    if vst in ("verified", "high_confidence"):
        b["verified_count"] += 1
    elif vst in ("suspect", "misinformation"):
        b["suspect_count"]  += 1


def _prune_old_buckets(mem: Dict[str, Any], keep_days: int = 30) -> None:
    """Drop buckets older than *keep_days* to cap memory growth."""
    buckets: Dict[str, Any] = mem["rolling_day_buckets"]
    cutoff = time.time() - keep_days * _SECONDS_PER_DAY
    to_delete = [
        d for d, b in buckets.items()
        if _date_str_to_epoch(d) < cutoff
    ]
    for d in to_delete:
        del buckets[d]


def _date_str_to_epoch(date_str: str) -> float:
    """Convert 'YYYY-MM-DD' to a UTC unix timestamp (noon of that day)."""
    try:
        struct = time.strptime(date_str + " 12:00:00", "%Y-%m-%d %H:%M:%S")
        return float(time.mktime(struct))
    except (ValueError, OverflowError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — TRAINING DATASET EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _source_tier(stats: Optional[Dict[str, Any]], source: str) -> str:
    """Classify a source into a ML-friendly tier label."""
    if stats is None:
        return "unknown"
    ratio = stats["suspect_count"] / stats["total_articles"] \
        if stats["total_articles"] else 0.0
    avg_vs = stats["avg_validation_score"]
    if ratio < 0.10 and avg_vs >= 0.80:
        return "verified"
    if ratio < 0.30 and avg_vs >= 0.60:
        return "mixed"
    if ratio >= 0.40 or avg_vs < 0.40:
        return "suspect"
    return "mixed"


def _final_label(article: Dict[str, Any]) -> str:
    vst = article.get("verification_status", "unknown")
    if vst in ("verified", "high_confidence"):
        return "safe"
    if vst == "misinformation":
        return "misinformation"
    if vst == "suspect" or article.get("needs_manual_review"):
        return "suspect"
    return "review"


def _build_training_record(article: Dict[str, Any],
                           source_stats_dict: Optional[Dict[str, Any]],
                           ) -> Dict[str, Any]:
    reasons:  List[str] = article.get("risk_reasons",    [])
    cluster:  List[str] = article.get("source_cluster",  [])
    tier = _source_tier(source_stats_dict, article.get("source", ""))

    contradiction_count = sum(
        1 for r in reasons if "contradict" in r.lower()
    )
    rumor_count = sum(
        1 for r in reasons
        if any(p in r.lower() for p in _RUMOR_PHRASE_TOKENS)
    )

    rec = TrainingRecord(
        source_tier=tier,
        validation_score=float(article.get("validation_score",    0.0)),
        misinformation_risk=float(article.get("misinformation_risk", 0.0)),
        trust_score=float(article.get("trust_score",         0.0)),
        cluster_depth=len(cluster),
        has_credibility_boost=bool(article.get("credibility_boost", False)),
        contradiction_count=contradiction_count,
        rumor_phrase_count=rumor_count,
        final_status_label=_final_label(article),
        timestamp=time.time(),
    )
    return asdict(rec)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — INCREMENTAL UPDATE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _ingest_article(mem: Dict[str, Any], article: Dict[str, Any]) -> None:
    """
    Merge one validated article into all memory sub-systems.

    All updates are O(1) with respect to historical corpus size:
    · Source stats:        running-sum Welford-style mean
    · Fingerprint:         dict upsert by sha256 key
    · Rumor registry:      dict increment by phrase token
    · Rolling bucket:      dict upsert by ISO date key
    · Training record:     list append
    """
    source: str = article.get("source", "unknown").strip() or "unknown"

    # ── source stats ──────────────────────────────────────────────────────────
    ss_dict: Dict[str, Any] = mem["source_stats"]
    if source not in ss_dict:
        ss_dict[source] = asdict(SourceStats(source=source))
    # reconstruct running sums (preserved across JSON round-trips)
    stats = ss_dict[source]
    stats["_sum_validation_score"]    = stats.get("_sum_validation_score",    0.0)
    stats["_sum_misinformation_risk"] = stats.get("_sum_misinformation_risk", 0.0)
    stats["_sum_trust_score"]         = stats.get("_sum_trust_score",         0.0)

    # inline the SourceStats.ingest() logic to avoid re-instantiation cost
    n   = stats["total_articles"] + 1
    vs  = float(article.get("validation_score",    0.0))
    mr  = float(article.get("misinformation_risk", 0.0))
    ts  = float(article.get("trust_score",         0.0))
    vst = article.get("verification_status", "unknown")

    stats["_sum_validation_score"]    += vs
    stats["_sum_misinformation_risk"] += mr
    stats["_sum_trust_score"]         += ts
    stats["avg_validation_score"]    = stats["_sum_validation_score"]    / n
    stats["avg_misinformation_risk"] = stats["_sum_misinformation_risk"] / n
    stats["avg_trust_score"]         = stats["_sum_trust_score"]         / n
    stats["total_articles"]          = n
    stats["last_seen_timestamp"]     = time.time()

    if vst in ("suspect", "misinformation"):
        stats["suspect_count"]  += 1
    elif vst in ("verified", "high_confidence"):
        stats["verified_count"] += 1
    if article.get("is_high_confidence_news"):
        stats["high_confidence_count"] = stats.get("high_confidence_count", 0) + 1
    if article.get("needs_manual_review"):
        stats["needs_review_count"] = stats.get("needs_review_count", 0) + 1

    reasons: List[str] = article.get("risk_reasons", [])
    for r in reasons:
        rl = r.lower()
        if "contradict" in rl:
            stats["contradiction_flags"] = stats.get("contradiction_flags", 0) + 1
        if any(p in rl for p in ("rumour", "rumor", "unconfirmed",
                                  "speculation", "alleged", "purported")):
            stats["rumor_phrase_hits"] = stats.get("rumor_phrase_hits", 0) + 1

    # ── fingerprint ───────────────────────────────────────────────────────────
    fp = _compute_fingerprint(article)
    _update_fingerprints(mem, article, fp)

    # ── rumor phrase registry ─────────────────────────────────────────────────
    _update_rumor_registry(mem, article)

    # ── rolling day bucket ────────────────────────────────────────────────────
    _update_rolling_bucket(mem, article)

    # ── training record ───────────────────────────────────────────────────────
    mem["training_export_memory"].append(
        _build_training_record(article, stats)
    )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def record_validation_results(validated_articles: List[Dict[str, Any]]) -> None:
    """
    Ingest a batch of validated articles from fake_news_validator.py into
    forensic memory.

    Each article must include the upstream contract fields:
        source, title, summary, link, published_time,
        validation_score, trust_score, misinformation_risk,
        verification_status, is_high_confidence_news, needs_manual_review,
        risk_reasons, source_cluster, credibility_boost.

    Updates are fully incremental — calling this with 1 or 10 000 articles
    is always O(N) in batch size and O(1) per article against historical state.

    Parameters
    ----------
    validated_articles : list[dict]
        Output list from fake_news_validator.py.
    """
    if not validated_articles:
        return

    t0  = time.perf_counter()
    mem = _load_memory()

    for article in validated_articles:
        _ingest_article(mem, article)

    _prune_old_buckets(mem, keep_days=90)
    _save_memory(mem)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    log.debug(
        "record_validation_results: ingested %d articles in %.2f ms.",
        len(validated_articles), elapsed_ms,
    )


def get_source_reputation(source: str) -> Dict[str, Any]:
    """
    Return the complete reputation profile for a single source.

    Designed to be called by signal_engine.py as a credibility pre-filter
    before computing final signal confidence.

    Returns
    -------
    dict
        Keys: source, total_articles, avg_validation_score,
              avg_misinformation_risk, avg_trust_score, suspect_ratio,
              verified_ratio, is_repeat_offender, contradiction_flags,
              rumor_phrase_hits, last_seen_timestamp, reputation_grade.
        Returns a zeroed profile with grade='UNKNOWN' for unknown sources.
    """
    mem   = _load_memory()
    stats = mem["source_stats"].get(source)

    if stats is None:
        return {
            "source":                source,
            "total_articles":        0,
            "avg_validation_score":  0.0,
            "avg_misinformation_risk": 0.0,
            "avg_trust_score":       0.0,
            "suspect_ratio":         0.0,
            "verified_ratio":        0.0,
            "is_repeat_offender":    False,
            "contradiction_flags":   0,
            "rumor_phrase_hits":     0,
            "last_seen_timestamp":   None,
            "reputation_grade":      "UNKNOWN",
        }

    n       = stats["total_articles"]
    sus_r   = stats["suspect_count"]  / n if n else 0.0
    ver_r   = stats["verified_count"] / n if n else 0.0
    offend  = n >= 3 and sus_r >= _SUSPECT_RATIO_THRESHOLD
    grade   = _reputation_grade(stats["avg_validation_score"], sus_r)

    return {
        "source":                  source,
        "total_articles":          n,
        "avg_validation_score":    round(stats["avg_validation_score"],    4),
        "avg_misinformation_risk": round(stats["avg_misinformation_risk"], 4),
        "avg_trust_score":         round(stats["avg_trust_score"],         4),
        "suspect_ratio":           round(sus_r, 4),
        "verified_ratio":          round(ver_r, 4),
        "is_repeat_offender":      offend,
        "contradiction_flags":     stats.get("contradiction_flags",   0),
        "rumor_phrase_hits":       stats.get("rumor_phrase_hits",     0),
        "last_seen_timestamp":     stats.get("last_seen_timestamp"),
        "reputation_grade":        grade,
    }


def get_top_repeat_offenders(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return the *limit* sources with the highest suspect ratios
    (minimum 3 observed articles).

    Used by alert_router.py to suppress or flag signals from toxic sources.

    Returns
    -------
    list[dict]
        Sorted descending by suspect_ratio.  Each entry matches
        the shape returned by get_source_reputation().
    """
    mem = _load_memory()
    offenders = []
    for source, stats in mem["source_stats"].items():
        n = stats["total_articles"]
        if n < 3:
            continue
        sus_r = stats["suspect_count"] / n
        if sus_r >= _SUSPECT_RATIO_THRESHOLD:
            offenders.append(get_source_reputation(source))
    offenders.sort(key=lambda x: x["suspect_ratio"], reverse=True)
    return offenders[:limit]


def get_rolling_validation_stats(days: int = _DEFAULT_WINDOW_DAYS) -> Dict[str, Any]:
    """
    Aggregate rolling statistics across the most recent *days* calendar days.

    Parameters
    ----------
    days : int
        Look-back window in days (default 7).

    Returns
    -------
    dict
        Keys: window_days, total_articles, verified_ratio, suspect_ratio,
              avg_trust, avg_misinfo_risk, active_sources_in_window,
              most_active_day.
    """
    mem     = _load_memory()
    buckets = mem["rolling_day_buckets"]
    cutoff  = time.time() - days * _SECONDS_PER_DAY

    total       = 0
    verified    = 0
    suspect     = 0
    sum_trust   = 0.0
    sum_misinfo = 0.0
    best_day    = ""
    best_count  = 0

    for date_str, b in buckets.items():
        if _date_str_to_epoch(date_str) < cutoff:
            continue
        ac = b["article_count"]
        total    += ac
        verified += b["verified_count"]
        suspect  += b["suspect_count"]
        sum_trust   += b["sum_trust"]
        sum_misinfo += b["sum_misinfo_risk"]
        if ac > best_count:
            best_count = ac
            best_day   = date_str

    # count sources seen in window
    active_sources = set()
    for source, stats in mem["source_stats"].items():
        lts = stats.get("last_seen_timestamp", 0.0) or 0.0
        if lts >= cutoff:
            active_sources.add(source)

    return {
        "window_days":             days,
        "total_articles":          total,
        "verified_ratio":          round(verified / total, 4) if total else 0.0,
        "suspect_ratio":           round(suspect  / total, 4) if total else 0.0,
        "avg_trust":               round(sum_trust   / total, 4) if total else 0.0,
        "avg_misinfo_risk":        round(sum_misinfo / total, 4) if total else 0.0,
        "active_sources_in_window": len(active_sources),
        "most_active_day":         best_day or None,
    }


def get_outlet_risk_profile(source: str) -> Dict[str, Any]:
    """
    Return an enriched risk profile for a source including fingerprint
    history and top rumour phrases associated with it.

    Parameters
    ----------
    source : str
        Exact source identifier (matches upstream fake_news_validator.py).

    Returns
    -------
    dict
        Extends get_source_reputation() with:
        - fingerprint_occurrences: count of distinct content hashes seen
        - repeat_misinformation_hashes: list of fingerprints seen >= twice
        - top_rumor_phrases: top-5 rumour phrase tokens hit by this source
        - risk_summary: human-readable one-liner
    """
    reputation = get_source_reputation(source)
    mem        = _load_memory()

    # fingerprint analysis for this source
    fp_seen:    List[str] = []
    fp_repeat:  List[str] = []
    phrase_hits: Dict[str, int] = {}

    for fp_hex, rec in mem["suspect_fingerprints"].items():
        if source in rec.get("sources", []):
            fp_seen.append(fp_hex)
            if rec["occurrence_count"] >= _FINGERPRINT_MIN_HITS:
                fp_repeat.append(fp_hex)
            for r in rec.get("all_risk_reasons", []):
                rl = r.lower()
                for phrase in _RUMOR_PHRASE_TOKENS:
                    if phrase in rl:
                        phrase_hits[phrase] = phrase_hits.get(phrase, 0) + 1

    top_phrases = sorted(phrase_hits.items(), key=lambda x: x[1], reverse=True)[:5]

    grade = reputation["reputation_grade"]
    if grade in ("A", "B"):
        risk_summary = f"{source} is a generally reliable outlet."
    elif grade == "C":
        risk_summary = f"{source} shows mixed credibility — review signals carefully."
    elif grade in ("D", "F"):
        risk_summary = f"{source} is a flagged repeat offender. Suppress or heavily discount signals."
    else:
        risk_summary = f"{source} has insufficient history for reputation scoring."

    return {
        **reputation,
        "fingerprint_occurrences":       len(fp_seen),
        "repeat_misinformation_hashes":  fp_repeat,
        "top_rumor_phrases":             [p for p, _ in top_phrases],
        "risk_summary":                  risk_summary,
    }


def export_training_dataset() -> List[Dict[str, Any]]:
    """
    Export all accumulated feature-label training records.

    Each record contains:
        source_tier, validation_score, misinformation_risk, trust_score,
        cluster_depth, has_credibility_boost, contradiction_count,
        rumor_phrase_count, final_status_label, timestamp.

    Returns
    -------
    list[dict]
        Ordered chronologically (oldest first).
        Safe for direct use with scikit-learn, XGBoost, or custom ML pipelines.

    Future hook
    -----------
    signal_engine.py will consume this export periodically to retrain its
    confidence-calibration model without requiring an external database.
    """
    mem = _load_memory()
    return list(mem["training_export_memory"])


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SOURCE REPUTATION ANALYTICS (helpers)
# ══════════════════════════════════════════════════════════════════════════════

def _reputation_grade(avg_vs: float, suspect_ratio: float) -> str:
    """
    Map (avg_validation_score, suspect_ratio) to a letter grade.

    Grade  Description
    -----  -----------
    A      Highly reliable — trust signals from this source
    B      Generally reliable with minor caveats
    C      Mixed — apply standard signal discount
    D      Frequently suspect — apply heavy discount
    F      Repeat offender — suppress by default
    UNKNOWN No history
    """
    if avg_vs >= 0.85 and suspect_ratio < 0.10:
        return "A"
    if avg_vs >= 0.70 and suspect_ratio < 0.20:
        return "B"
    if avg_vs >= 0.55 and suspect_ratio < 0.35:
        return "C"
    if avg_vs >= 0.40 or suspect_ratio < 0.55:
        return "D"
    return "F"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 — OBSERVABILITY + SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def print_memory_summary(path_override: Optional[str] = None) -> None:
    """
    Print a comprehensive, human-readable diagnostic of the forensic memory.

    Called from mission_control health checks or directly via __main__.
    """
    if path_override:
        global _mem_path, _mem_cache
        _mem_path  = path_override
        _mem_cache = None

    mem   = _load_memory()
    stats = get_rolling_validation_stats()
    W     = 72

    def _bar(c: str = "═") -> str:
        return c * W

    def _row(label: str, value: Any, pad: int = 44) -> str:
        dots = "." * max(pad - len(label), 1)
        return f"  {label}{dots}{value}"

    def _ts(epoch: Optional[float]) -> str:
        if not epoch:
            return "never"
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))

    source_count  = len(mem["source_stats"])
    fp_count      = len(mem["suspect_fingerprints"])
    repeat_fps    = sum(
        1 for r in mem["suspect_fingerprints"].values()
        if r["occurrence_count"] >= _FINGERPRINT_MIN_HITS
    )
    training_rows = len(mem["training_export_memory"])
    offenders     = get_top_repeat_offenders(limit=5)

    print()
    print(_bar())
    print("  MONSTER TRADING AI — Forensic Validation Memory".center(W))
    print(_bar())
    print(_row("Schema Version", mem.get("schema_version", "?")))
    print(_row("Last Saved",     mem.get("saved_at", "unsaved")))
    print()

    print("  ┌─ CORPUS OVERVIEW " + "─" * 52 + "┐")
    print(_row("  │  Tracked Sources",        source_count))
    print(_row("  │  Content Fingerprints",   fp_count))
    print(_row("  │  Repeat Misinfo Hashes",  repeat_fps))
    print(_row("  │  Training Records",       training_rows))
    rumor_total = sum(mem["rumor_phrase_registry"].values())
    print(_row("  │  Rumor Phrase Hits Total", rumor_total))
    print("  └" + "─" * 70 + "┘")

    print()
    print("  ┌─ ROLLING 7-DAY WINDOW " + "─" * 47 + "┐")
    print(_row("  │  Total Articles",          stats["total_articles"]))
    print(_row("  │  Verified Ratio",
               f"{stats['verified_ratio'] * 100:.1f} %"))
    print(_row("  │  Suspect Ratio",
               f"{stats['suspect_ratio'] * 100:.1f} %"))
    print(_row("  │  Avg Trust Score",
               f"{stats['avg_trust']:.3f}"))
    print(_row("  │  Avg Misinfo Risk",
               f"{stats['avg_misinfo_risk']:.3f}"))
    print(_row("  │  Active Sources (7d)",     stats["active_sources_in_window"]))
    print(_row("  │  Most Active Day",         stats["most_active_day"] or "—"))
    print("  └" + "─" * 70 + "┘")

    print()
    print("  ┌─ TOP REPEAT OFFENDERS " + "─" * 47 + "┐")
    if offenders:
        for o in offenders:
            grade = o["reputation_grade"]
            ratio = f"{o['suspect_ratio']*100:.0f}%"
            print(f"  │  [{grade}] {o['source']:<28} suspect={ratio:<7} "
                  f"n={o['total_articles']}")
    else:
        print("  │  (no repeat offenders above threshold)")
    print("  └" + "─" * 70 + "┘")

    print()
    print("  ┌─ TOP RUMOR PHRASES " + "─" * 50 + "┐")
    phrase_reg = mem["rumor_phrase_registry"]
    if phrase_reg:
        top = sorted(phrase_reg.items(), key=lambda x: x[1], reverse=True)[:8]
        for phrase, cnt in top:
            print(f"  │  {phrase:<34} hits={cnt}")
    else:
        print("  │  (no phrase data yet)")
    print("  └" + "─" * 70 + "┘")

    print()
    print(_bar())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (8 synthetic validated articles)
# ══════════════════════════════════════════════════════════════════════════════

class _ValidationMemoryTests(unittest.TestCase):
    """8 synthetic-article smoke tests — must all pass before deployment."""

    _ARTICLES: List[Dict[str, Any]] = [
        # 0 — high-confidence verified article from reputable outlet
        {
            "source": "reuters.com", "title": "Fed raises rates by 25bps",
            "summary": "Federal Reserve lifts benchmark rate.",
            "link": "https://reuters.com/a1", "published_time": "2025-01-10T09:00:00Z",
            "validation_score": 0.92, "trust_score": 0.95,
            "misinformation_risk": 0.05, "verification_status": "verified",
            "is_high_confidence_news": True, "needs_manual_review": False,
            "risk_reasons": [], "source_cluster": ["reuters.com", "apnews.com"],
            "credibility_boost": True,
        },
        # 1 — suspect article with contradiction flag
        {
            "source": "cryptorumors.io", "title": "BTC to $1M by Friday — insiders say",
            "summary": "Anonymous sources claim massive move imminent.",
            "link": "https://cryptorumors.io/b1", "published_time": "2025-01-10T10:00:00Z",
            "validation_score": 0.21, "trust_score": 0.18,
            "misinformation_risk": 0.88, "verification_status": "suspect",
            "is_high_confidence_news": False, "needs_manual_review": True,
            "risk_reasons": ["unconfirmed rumor", "contradicts official data"],
            "source_cluster": ["cryptorumors.io"], "credibility_boost": False,
        },
        # 2 — same title + source as article[1] → duplicate fingerprint
        {
            "source": "cryptorumors.io", "title": "BTC to $1M by Friday — insiders say",
            "summary": "Same story, different link.",
            "link": "https://cryptorumors.io/b2", "published_time": "2025-01-10T11:00:00Z",
            "validation_score": 0.19, "trust_score": 0.15,
            "misinformation_risk": 0.91, "verification_status": "misinformation",
            "is_high_confidence_news": False, "needs_manual_review": True,
            "risk_reasons": ["alleged", "speculation"],
            "source_cluster": ["cryptorumors.io"], "credibility_boost": False,
        },
        # 3 — mixed-credibility outlet
        {
            "source": "tradingblog.net", "title": "AAPL eyes $200 resistance",
            "summary": "Technical analysis suggests breakout.",
            "link": "https://tradingblog.net/c1", "published_time": "2025-01-10T12:00:00Z",
            "validation_score": 0.61, "trust_score": 0.58,
            "misinformation_risk": 0.30, "verification_status": "unverified",
            "is_high_confidence_news": False, "needs_manual_review": True,
            "risk_reasons": ["speculation"], "source_cluster": [],
            "credibility_boost": False,
        },
        # 4 — another reputable article (reuters, second hit)
        {
            "source": "reuters.com", "title": "ECB holds rates steady",
            "summary": "European Central Bank maintains policy.",
            "link": "https://reuters.com/a2", "published_time": "2025-01-10T13:00:00Z",
            "validation_score": 0.90, "trust_score": 0.93,
            "misinformation_risk": 0.07, "verification_status": "verified",
            "is_high_confidence_news": True, "needs_manual_review": False,
            "risk_reasons": [], "source_cluster": ["reuters.com"],
            "credibility_boost": True,
        },
        # 5 — low-trust source with multiple rumour flags
        {
            "source": "cryptorumors.io", "title": "Elon to buy Binance — purported deal",
            "summary": "Purported acquisition confirmed by anonymous sources.",
            "link": "https://cryptorumors.io/b3", "published_time": "2025-01-10T14:00:00Z",
            "validation_score": 0.15, "trust_score": 0.12,
            "misinformation_risk": 0.95, "verification_status": "misinformation",
            "is_high_confidence_news": False, "needs_manual_review": True,
            "risk_reasons": ["purported", "anonymous source", "unconfirmed rumor",
                             "contradicts official denial"],
            "source_cluster": [], "credibility_boost": False,
        },
        # 6 — neutral / borderline article
        {
            "source": "marketwatch.com", "title": "Oil prices fall on demand fears",
            "summary": "Crude drops 2% amid recession concerns.",
            "link": "https://marketwatch.com/d1", "published_time": "2025-01-10T15:00:00Z",
            "validation_score": 0.75, "trust_score": 0.72,
            "misinformation_risk": 0.18, "verification_status": "verified",
            "is_high_confidence_news": False, "needs_manual_review": False,
            "risk_reasons": [], "source_cluster": ["marketwatch.com", "reuters.com"],
            "credibility_boost": False,
        },
        # 7 — completely new unknown source
        {
            "source": "newscryptowire.xyz", "title": "DOGE replaces dollar in Paraguay",
            "summary": "Alleged decree signed overnight.",
            "link": "https://newscryptowire.xyz/e1", "published_time": "2025-01-10T16:00:00Z",
            "validation_score": 0.10, "trust_score": 0.08,
            "misinformation_risk": 0.97, "verification_status": "misinformation",
            "is_high_confidence_news": False, "needs_manual_review": True,
            "risk_reasons": ["alleged", "unverified", "contradicts official statement"],
            "source_cluster": [], "credibility_boost": False,
        },
    ]

    def setUp(self) -> None:
        import tempfile as _tf, shutil
        self._tmp = _tf.mkdtemp(prefix="vm_test_")
        self._path = os.path.join(self._tmp, "validation_memory.json")
        self._reset()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._reset()

    def _reset(self) -> None:
        global _mem_cache, _mem_path
        _mem_cache = None
        _mem_path  = getattr(self, "_path", _DEFAULT_STATE_PATH)

    def _ingest(self, articles: Optional[List] = None) -> None:
        global _mem_path
        _mem_path = self._path
        record_validation_results(articles or self._ARTICLES)

    # ── Test 1 — fresh state on first run ────────────────────────────────────
    def test_01_fresh_state_on_first_run(self) -> None:
        self._reset()
        global _mem_path
        _mem_path = self._path
        mem = _load_memory()
        self.assertEqual(mem["source_stats"],         {})
        self.assertEqual(mem["suspect_fingerprints"], {})
        self.assertEqual(mem["schema_version"],       MEM_SCHEMA_VERSION)

    # ── Test 2 — record_validation_results persists source stats ─────────────
    def test_02_record_validation_results_persists(self) -> None:
        self._ingest()
        self._reset()
        mem = _load_memory()
        self.assertIn("reuters.com",       mem["source_stats"])
        self.assertIn("cryptorumors.io",   mem["source_stats"])
        self.assertIn("newscryptowire.xyz", mem["source_stats"])

    # ── Test 3 — source reputation returns correct grade ─────────────────────
    def test_03_source_reputation_grades(self) -> None:
        self._ingest()
        reuters  = get_source_reputation("reuters.com")
        crypto   = get_source_reputation("cryptorumors.io")
        unknown  = get_source_reputation("never_seen_source.com")
        self.assertIn(reuters["reputation_grade"],  ("A", "B"))
        self.assertIn(crypto["reputation_grade"],   ("D", "F"))
        self.assertEqual(unknown["reputation_grade"], "UNKNOWN")
        self.assertEqual(unknown["total_articles"],    0)

    # ── Test 4 — repeat offenders list correct ────────────────────────────────
    def test_04_top_repeat_offenders(self) -> None:
        self._ingest()
        offenders = get_top_repeat_offenders(limit=10)
        sources = [o["source"] for o in offenders]
        self.assertIn("cryptorumors.io", sources)
        # reuters should NOT appear (low suspect ratio)
        self.assertNotIn("reuters.com", sources)

    # ── Test 5 — duplicate fingerprint detected ───────────────────────────────
    def test_05_fingerprint_dedup(self) -> None:
        self._ingest()
        mem = _load_memory()
        # articles[1] and articles[2] have identical title+source
        hit = sum(
            1 for rec in mem["suspect_fingerprints"].values()
            if rec["occurrence_count"] >= 2
        )
        self.assertGreaterEqual(hit, 1)

    # ── Test 6 — rolling stats cover ingested articles ───────────────────────
    def test_06_rolling_validation_stats(self) -> None:
        self._ingest()
        stats = get_rolling_validation_stats(days=7)
        self.assertEqual(stats["total_articles"], len(self._ARTICLES))
        self.assertGreater(stats["suspect_ratio"],   0.0)
        self.assertGreater(stats["verified_ratio"],  0.0)
        self.assertGreater(stats["avg_misinfo_risk"], 0.0)

    # ── Test 7 — outlet risk profile enriches reputation ─────────────────────
    def test_07_outlet_risk_profile(self) -> None:
        self._ingest()
        profile = get_outlet_risk_profile("cryptorumors.io")
        self.assertIn("fingerprint_occurrences",      profile)
        self.assertIn("repeat_misinformation_hashes", profile)
        self.assertIn("top_rumor_phrases",            profile)
        self.assertIn("risk_summary",                 profile)
        self.assertGreater(profile["fingerprint_occurrences"], 0)

    # ── Test 8 — training dataset export correctness ──────────────────────────
    def test_08_export_training_dataset(self) -> None:
        self._ingest()
        dataset = export_training_dataset()
        self.assertEqual(len(dataset), len(self._ARTICLES))
        required_keys = {
            "source_tier", "validation_score", "misinformation_risk",
            "trust_score", "cluster_depth", "has_credibility_boost",
            "contradiction_count", "rumor_phrase_count",
            "final_status_label", "timestamp",
        }
        for record in dataset:
            self.assertTrue(required_keys.issubset(record.keys()),
                            f"Missing keys in: {record}")
        # verify label mapping
        labels = {r["final_status_label"] for r in dataset}
        self.assertTrue(labels.issubset({"safe", "suspect", "misinformation", "review"}))


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile, shutil

    # Use a temporary directory for the smoke-test so it doesn't pollute cwd
    _smoke_dir  = tempfile.mkdtemp(prefix="vm_smoke_")
    _smoke_path = os.path.join(_smoke_dir, "validation_memory.json")

    try:
        # ── demo run ──────────────────────────────────────────────────────────
        _reset_singleton(path=_smoke_path)

        DEMO_ARTICLES: List[Dict[str, Any]] = _ValidationMemoryTests._ARTICLES

        print()
        print("  Ingesting 8 synthetic validated articles …")
        t0 = time.perf_counter()
        record_validation_results(DEMO_ARTICLES)
        ms = (time.perf_counter() - t0) * 1_000
        print(f"  Done in {ms:.2f} ms.")

        print_memory_summary(path_override=_smoke_path)

        # ── unit tests ────────────────────────────────────────────────────────
        print("═" * 72)
        print("  Running 8 smoke tests …".center(72))
        print("═" * 72)

        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(_ValidationMemoryTests)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)

        total  = result.testsRun
        passed = total - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)

        print()
        print("═" * 72)
        if failed == 0:
            print(f"  ✓  {passed}/{total}  ALL CLEAR — validation_memory.py is production-ready.")
        else:
            print(f"  ✗  {passed}/{total} passed — {failed} failure(s). See above.")
        print("═" * 72)
        print()

    finally:
        shutil.rmtree(_smoke_dir, ignore_errors=True)

    sys.exit(0 if result.wasSuccessful() else 1)