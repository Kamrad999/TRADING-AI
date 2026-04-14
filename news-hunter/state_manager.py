"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MONSTER TRADING AI — state_manager.py                          ║
║         Fault-Tolerant Persistence & Recovery Backbone                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Consumers: execution_bridge · risk_guardian · broker_sender                ║
║             performance_analytics · mission_control · config                ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module is the shared persistence backbone for AI Trading HQ.
It is NOT standalone business logic — it is the fault-tolerant
snapshot/restore layer that every other module depends on.

Architecture layers
-------------------
1.  Constants + schema version
2.  Dataclasses / TypedDict state models
3.  Atomic write engine
4.  Backup snapshot rotation
5.  Corruption recovery
6.  Version migration framework
7.  Public save/load APIs
8.  Plugin hooks + observability
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── optional config import (graceful fallback if run standalone) ──────────────
try:
    import config as _cfg
    _STATE_FILE_DEFAULT: str = _cfg.STATE_FILE
    _REPORTS_DIR_DEFAULT: str = _cfg.REPORTS_DIR
    _DEBUG_DEFAULT: bool = _cfg.DEBUG
except ModuleNotFoundError:
    _STATE_FILE_DEFAULT = "state.json"
    _REPORTS_DIR_DEFAULT = "reports"
    _DEBUG_DEFAULT = True

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS + SCHEMA VERSION
# ══════════════════════════════════════════════════════════════════════════════

STATE_SCHEMA_VERSION: int = 1
"""
Monotonically increasing integer. Bump whenever the on-disk schema changes.
All snapshots embed this value so the migration framework can detect stale files.
"""

_BACKUP_SUFFIX: str = ".backup.json"
_TEMP_SUFFIX: str = ".tmp"
_ENCODING: str = "utf-8"
_JSON_INDENT: int = 2
_MAX_PLUGIN_ERRORS: int = 5   # silence a plugin after this many consecutive failures

log = logging.getLogger("state_manager")
if _DEBUG_DEFAULT and not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] state_manager: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.DEBUG)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  STATE MODELS  (dataclasses)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionRecord:
    """Single open position as persisted to disk."""
    ticker: str
    quantity: float
    avg_price: float
    broker: str
    asset_class: str                   # CRYPTO | EQUITY | FOREX | MACRO
    open_timestamp: float              # unix epoch
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    conviction_score: Optional[float] = None
    conviction_bonus_applied: bool = False
    notes: str = ""


@dataclass
class DailyLossLedger:
    """Intraday P&L tracking consumed by risk_guardian."""
    utc_date: str                      # ISO-8601 date string  e.g. "2025-01-15"
    realized_loss_usd: float = 0.0
    unrealized_loss_usd: float = 0.0
    warning_triggered: bool = False
    lockout_triggered: bool = False
    trades_today: int = 0


@dataclass
class KillSwitchState:
    """Emergency circuit-breaker state."""
    active: bool = False
    activated_at: Optional[float] = None   # unix epoch, None if never triggered
    reason: str = ""
    triggered_by: str = ""                  # module name that flipped the switch


@dataclass
class AnalyticsCheckpoint:
    """Rolling analytics cache for performance_analytics."""
    rolling_pnl: List[float] = field(default_factory=list)
    win_streak: int = 0
    loss_streak: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    peak_portfolio_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    latest_report_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionControlHeartbeat:
    """Liveness telemetry for mission_control watchdog."""
    last_successful_loop: Optional[float] = None
    last_news_pull: Optional[float] = None
    last_broker_send: Optional[float] = None
    last_health_check: Optional[float] = None
    loop_count: int = 0
    error_count: int = 0


# ── canonical empty / default state ──────────────────────────────────────────

def _default_state() -> Dict[str, Any]:
    """Return a fresh, fully-populated default state dict."""
    now_iso = time.strftime("%Y-%m-%d", time.gmtime())
    return {
        "schema_version":         STATE_SCHEMA_VERSION,
        "saved_at":               None,
        "active_positions":       {},          # ticker -> PositionRecord dict
        "cooldown_registry":      {},          # ticker -> float (unix expiry)
        "daily_loss_ledger":      asdict(DailyLossLedger(utc_date=now_iso)),
        "kill_switch_state":      asdict(KillSwitchState()),
        "analytics_checkpoint":   asdict(AnalyticsCheckpoint()),
        "mission_control_heartbeat": asdict(MissionControlHeartbeat()),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLETON CACHE + THREAD SAFETY
# ══════════════════════════════════════════════════════════════════════════════

_state_lock = threading.RLock()     # re-entrant so internal calls compose safely
_memory_cache: Optional[Dict[str, Any]] = None   # hydrated on first load
_cache_dirty: bool = False

# runtime-overridable paths (used by tests and Docker deployments)
_state_file_path: str = _STATE_FILE_DEFAULT

def _backup_path(state_file: str) -> str:
    base = state_file[: -len(".json")] if state_file.endswith(".json") else state_file
    return base + _BACKUP_SUFFIX

# ══════════════════════════════════════════════════════════════════════════════
# 4.  ATOMIC WRITE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    """
    Write *data* to *path* atomically.

    Flow
    ----
    1. Serialise to JSON (pretty-printed, UTF-8).
    2. Write to a sibling temp file in the same directory.
    3. fsync the temp file descriptor.
    4. os.replace(temp -> target)  — atomic on POSIX; best-effort on Windows.
    5. fsync the parent directory to flush the directory entry (POSIX only).

    Raises
    ------
    OSError
        Propagated from the underlying filesystem calls so callers can handle
        partial-write scenarios.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)

    payload = json.dumps(data, indent=_JSON_INDENT, ensure_ascii=False,
                         default=str)
    encoded = payload.encode(_ENCODING)

    fd, tmp_path = tempfile.mkstemp(suffix=_TEMP_SUFFIX, dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        # Flush directory entry on POSIX
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (AttributeError, OSError):
            pass  # non-POSIX or unsupported — best effort
    except Exception:
        # clean up temp on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_read(path: str) -> Optional[Dict[str, Any]]:
    """
    Read and parse a JSON file.  Returns None on any error (missing, corrupt).
    """
    try:
        with open(path, "r", encoding=_ENCODING) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None

# ══════════════════════════════════════════════════════════════════════════════
# 5.  BACKUP ROTATION + CORRUPTION RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_backup(state_file: str) -> None:
    """
    Copy *state_file* -> *backup_file* atomically before overwriting the main
    file.  Called at the start of every save so the backup always holds the
    last-known-good snapshot.
    """
    backup = _backup_path(state_file)
    try:
        data = _safe_read(state_file)
        if data is not None:
            _atomic_write(backup, data)
    except Exception as exc:
        log.warning("Backup rotation failed (non-fatal): %s", exc)


def _load_with_recovery(state_file: str) -> Dict[str, Any]:
    """
    Load state from disk with two-tier recovery:

    Tier 1 — primary file healthy           → return parsed data
    Tier 2 — primary corrupt, backup good   → restore from backup, warn
    Tier 3 — both missing / corrupt         → return fresh default state
    """
    data = _safe_read(state_file)
    if data is not None:
        log.debug("State loaded from primary: %s", state_file)
        return data

    backup = _backup_path(state_file)
    data = _safe_read(backup)
    if data is not None:
        log.warning(
            "Primary state file missing/corrupt. Restored from backup: %s", backup
        )
        # Immediately restore the primary so next crash has a good primary
        try:
            _atomic_write(state_file, data)
        except OSError as exc:
            log.error("Could not restore primary from backup: %s", exc)
        return data

    log.warning(
        "Both state files missing/corrupt (%s, %s). Starting with default state.",
        state_file, backup,
    )
    return _default_state()

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VERSION MIGRATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

# Map  from_version -> migration_callable(data: dict) -> dict
MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    # Example (uncomment when v2 is introduced):
    # 1: _migrate_v1_to_v2,
}


def _apply_migrations(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk the migration chain from the file's schema_version up to
    STATE_SCHEMA_VERSION, applying each registered migration in order.

    If the file is already current this is a no-op.
    If the file is *newer* than the running code a warning is emitted but
    the data is returned unchanged (forward-compatibility best-effort).
    """
    file_version = data.get("schema_version", 1)

    if file_version == STATE_SCHEMA_VERSION:
        return data                          # fast path — already current

    if file_version > STATE_SCHEMA_VERSION:
        log.warning(
            "State file schema_version=%d is newer than code version=%d. "
            "Proceeding without downgrade migration.",
            file_version, STATE_SCHEMA_VERSION,
        )
        return data

    log.info(
        "Migrating state schema from v%d to v%d.", file_version, STATE_SCHEMA_VERSION
    )
    current = file_version
    while current < STATE_SCHEMA_VERSION:
        migrator = MIGRATIONS.get(current)
        if migrator is None:
            log.warning(
                "No migration registered for v%d -> v%d. Skipping.", current, current + 1
            )
            current += 1
            continue
        try:
            data = migrator(data)
            log.info("Applied migration v%d -> v%d.", current, current + 1)
        except Exception as exc:
            log.error(
                "Migration v%d -> v%d failed: %s. State left at v%d.",
                current, current + 1, exc, current,
            )
            break
        current += 1

    data["schema_version"] = STATE_SCHEMA_VERSION
    return data


def _ensure_schema_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge any missing top-level keys from the default state into *data*.
    This provides forward-compatibility when new keys are added between releases
    without requiring a full migration.
    """
    defaults = _default_state()
    for key, default_value in defaults.items():
        if key not in data:
            log.debug("Backfilling missing state key: '%s'", key)
            data[key] = copy.deepcopy(default_value)
    return data

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLUGIN HOOKS + OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PluginEntry:
    fn: Callable[[Dict[str, Any]], None]
    error_count: int = 0
    silenced: bool = False


_plugins: List[_PluginEntry] = []


def register_plugin(fn: Callable[[Dict[str, Any]], None]) -> None:
    """
    Register a post-save observer callback.

    The callback receives a *deep copy* of the full state dict immediately
    after every successful save.  It must not mutate the state or perform
    blocking I/O on the hot path.

    A plugin that raises more than _MAX_PLUGIN_ERRORS consecutive times is
    automatically silenced to protect the core save loop.

    Parameters
    ----------
    fn : callable
        Signature: fn(state: dict) -> None
    """
    _plugins.append(_PluginEntry(fn=fn))
    log.debug("Plugin registered: %s", getattr(fn, "__name__", repr(fn)))


def _fire_plugins(state: Dict[str, Any]) -> None:
    """Fire all non-silenced plugins with a copy of the saved state."""
    snapshot = copy.deepcopy(state)
    for entry in _plugins:
        if entry.silenced:
            continue
        try:
            entry.fn(snapshot)
            entry.error_count = 0          # reset on success
        except Exception as exc:
            entry.error_count += 1
            log.error(
                "Plugin '%s' raised (error %d/%d): %s",
                getattr(entry.fn, "__name__", "?"),
                entry.error_count, _MAX_PLUGIN_ERRORS, exc,
            )
            if entry.error_count >= _MAX_PLUGIN_ERRORS:
                entry.silenced = True
                log.error(
                    "Plugin '%s' silenced after %d consecutive failures.",
                    getattr(entry.fn, "__name__", "?"), _MAX_PLUGIN_ERRORS,
                )

# ══════════════════════════════════════════════════════════════════════════════
# 8.  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

# ── core load / save ──────────────────────────────────────────────────────────

def load_state(path_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the full state from disk (or return the in-memory singleton cache).

    Parameters
    ----------
    path_override : str, optional
        Override the default STATE_FILE path.  Useful for tests and Docker.

    Returns
    -------
    dict
        Fully-hydrated, migrated, and key-backfilled state dictionary.
    """
    global _memory_cache, _state_file_path
    with _state_lock:
        if path_override:
            _state_file_path = path_override
        if _memory_cache is not None:
            return _memory_cache          # O(1) singleton cache hit
        raw = _load_with_recovery(_state_file_path)
        raw = _apply_migrations(raw)
        raw = _ensure_schema_keys(raw)
        _memory_cache = raw
        log.debug("State hydrated into memory cache.")
        return _memory_cache


def save_state(state: Dict[str, Any], path_override: Optional[str] = None) -> None:
    """
    Persist the full state dict to disk atomically.

    Steps
    -----
    1. Rotate existing file to backup.
    2. Stamp saved_at + schema_version.
    3. Atomic write to primary path.
    4. Update in-memory cache.
    5. Fire post-save plugins.

    Parameters
    ----------
    state : dict
        Must be JSON-serialisable.
    path_override : str, optional
        Override the default STATE_FILE path.
    """
    global _memory_cache, _state_file_path
    with _state_lock:
        if path_override:
            _state_file_path = path_override
        target = _state_file_path
        _rotate_backup(target)
        payload = copy.deepcopy(state)
        payload["schema_version"] = STATE_SCHEMA_VERSION
        payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_write(target, payload)
        _memory_cache = payload
        log.debug("State saved → %s", target)
    _fire_plugins(payload)


# ── segment helpers ───────────────────────────────────────────────────────────

def _get_segment(key: str) -> Any:
    """Load state and return one top-level segment by key."""
    return load_state().get(key)


def _set_segment(key: str, value: Any) -> None:
    """Mutate one segment in the cache and persist the full state."""
    with _state_lock:
        state = load_state()
        state[key] = value
        save_state(state)


# ── positions ─────────────────────────────────────────────────────────────────

def save_positions(positions: Dict[str, Any]) -> None:
    """
    Persist the active positions map.

    Parameters
    ----------
    positions : dict
        Mapping of  ticker (str) → PositionRecord dict or raw dict.
        Pass an empty dict to clear all positions.
    """
    # Normalise: dataclasses → dict if callers pass dataclass instances
    serialisable = {}
    for ticker, pos in positions.items():
        serialisable[ticker] = asdict(pos) if isinstance(pos, PositionRecord) else pos
    _set_segment("active_positions", serialisable)


def load_positions() -> Dict[str, Any]:
    """
    Return the active positions map from cache/disk.

    Returns
    -------
    dict
        ticker → position dict
    """
    return _get_segment("active_positions") or {}


# ── cooldowns ─────────────────────────────────────────────────────────────────

def save_cooldowns(registry: Dict[str, float]) -> None:
    """
    Persist the cooldown expiry registry.

    Parameters
    ----------
    registry : dict
        ticker → unix timestamp (float) after which the cooldown expires.
    """
    _set_segment("cooldown_registry", registry)


def load_cooldowns() -> Dict[str, float]:
    """Return the cooldown registry from cache/disk."""
    return _get_segment("cooldown_registry") or {}


# ── daily loss ledger ─────────────────────────────────────────────────────────

def save_daily_loss(ledger: Any) -> None:
    """
    Persist the daily loss ledger.

    Parameters
    ----------
    ledger : DailyLossLedger | dict
        Accepts both the dataclass and a plain dict.
    """
    payload = asdict(ledger) if isinstance(ledger, DailyLossLedger) else ledger
    _set_segment("daily_loss_ledger", payload)


def load_daily_loss() -> Dict[str, Any]:
    """Return the daily loss ledger from cache/disk."""
    return _get_segment("daily_loss_ledger") or asdict(
        DailyLossLedger(utc_date=time.strftime("%Y-%m-%d", time.gmtime()))
    )


# ── kill switch ───────────────────────────────────────────────────────────────

def save_kill_switch(state: Any) -> None:
    """
    Persist the kill-switch state.

    Parameters
    ----------
    state : KillSwitchState | dict
        Accepts both the dataclass and a plain dict.
    """
    payload = asdict(state) if isinstance(state, KillSwitchState) else state
    _set_segment("kill_switch_state", payload)


def load_kill_switch() -> Dict[str, Any]:
    """Return the kill-switch state from cache/disk."""
    return _get_segment("kill_switch_state") or asdict(KillSwitchState())


# ── analytics checkpoint ──────────────────────────────────────────────────────

def save_analytics_checkpoint(data: Any) -> None:
    """
    Persist the analytics rolling cache.

    Parameters
    ----------
    data : AnalyticsCheckpoint | dict
    """
    payload = asdict(data) if isinstance(data, AnalyticsCheckpoint) else data
    _set_segment("analytics_checkpoint", payload)


def load_analytics_checkpoint() -> Dict[str, Any]:
    """Return the analytics checkpoint from cache/disk."""
    return _get_segment("analytics_checkpoint") or asdict(AnalyticsCheckpoint())


# ── mission control heartbeat ─────────────────────────────────────────────────

def save_heartbeat(data: Any) -> None:
    """
    Persist the mission-control liveness heartbeat.

    Parameters
    ----------
    data : MissionControlHeartbeat | dict
    """
    payload = asdict(data) if isinstance(data, MissionControlHeartbeat) else data
    _set_segment("mission_control_heartbeat", payload)


def load_heartbeat() -> Dict[str, Any]:
    """Return the mission-control heartbeat from cache/disk."""
    return _get_segment("mission_control_heartbeat") or asdict(MissionControlHeartbeat())


# ── summary printer ───────────────────────────────────────────────────────────

def print_state_summary(path_override: Optional[str] = None) -> None:
    """
    Print a human-readable diagnostic snapshot of the current state.

    Designed to be called from mission_control startup and health-check loops.
    """
    state = load_state(path_override)
    W = 72

    def _bar(c: str = "═") -> str:
        return c * W

    def _row(label: str, value: Any, pad: int = 42) -> str:
        dots = "." * max(pad - len(label), 1)
        return f"  {label}{dots}{value}"

    def _ts(epoch: Optional[float]) -> str:
        if epoch is None:
            return "never"
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch))

    positions  = state.get("active_positions", {})
    cooldowns  = state.get("cooldown_registry", {})
    ledger     = state.get("daily_loss_ledger", {})
    ks         = state.get("kill_switch_state", {})
    analytics  = state.get("analytics_checkpoint", {})
    heartbeat  = state.get("mission_control_heartbeat", {})

    print()
    print(_bar())
    print("  MONSTER TRADING AI — State Manager Snapshot".center(W))
    print(_bar())
    print(_row("Schema Version", state.get("schema_version", "?")))
    print(_row("Saved At",       state.get("saved_at", "unsaved")))
    print()

    # Positions
    print("  ┌─ ACTIVE POSITIONS " + "─" * 51 + "┐")
    if positions:
        for ticker, pos in positions.items():
            qty   = pos.get("quantity", "?")
            price = pos.get("avg_price", "?")
            cls   = pos.get("asset_class", "?")
            print(f"  │  {ticker:<10} qty={qty:<12} avg_px={price:<12} class={cls}")
    else:
        print("  │  (no open positions)")
    print("  └" + "─" * 70 + "┘")

    # Cooldowns
    now = time.time()
    active_cds = {t: exp for t, exp in cooldowns.items() if exp > now}
    print()
    print("  ┌─ COOLDOWN REGISTRY " + "─" * 50 + "┐")
    if active_cds:
        for ticker, exp in active_cds.items():
            remaining = max(0.0, exp - now)
            print(f"  │  {ticker:<12} expires in {remaining:.0f}s  ({_ts(exp)})")
    else:
        print("  │  (no active cooldowns)")
    print("  └" + "─" * 70 + "┘")

    # Daily Loss
    print()
    print("  ┌─ DAILY LOSS LEDGER " + "─" * 50 + "┐")
    print(_row("  │  Date",               ledger.get("utc_date", "?")))
    print(_row("  │  Realized Loss USD",  f"${ledger.get('realized_loss_usd', 0):.2f}"))
    print(_row("  │  Unrealized Loss USD",f"${ledger.get('unrealized_loss_usd', 0):.2f}"))
    print(_row("  │  Warning Triggered",  ledger.get("warning_triggered", False)))
    print(_row("  │  Lockout Triggered",  ledger.get("lockout_triggered", False)))
    print(_row("  │  Trades Today",       ledger.get("trades_today", 0)))
    print("  └" + "─" * 70 + "┘")

    # Kill Switch
    ks_label = "🔴 ACTIVE" if ks.get("active") else "🟢 INACTIVE"
    print()
    print("  ┌─ KILL SWITCH " + "─" * 56 + "┐")
    print(_row("  │  Status",       ks_label))
    print(_row("  │  Activated At", _ts(ks.get("activated_at"))))
    print(_row("  │  Reason",       ks.get("reason") or "—"))
    print("  └" + "─" * 70 + "┘")

    # Analytics
    print()
    print("  ┌─ ANALYTICS CHECKPOINT " + "─" * 47 + "┐")
    print(_row("  │  Total Trades",    analytics.get("total_trades", 0)))
    print(_row("  │  Wins / Losses",
               f"{analytics.get('total_wins', 0)} / {analytics.get('total_losses', 0)}"))
    print(_row("  │  Win Streak",      analytics.get("win_streak", 0)))
    print(_row("  │  Loss Streak",     analytics.get("loss_streak", 0)))
    print(_row("  │  Peak Portfolio",  f"${analytics.get('peak_portfolio_usd', 0):,.2f}"))
    print(_row("  │  Max Drawdown",    f"{analytics.get('max_drawdown_pct', 0):.2f} %"))
    print("  └" + "─" * 70 + "┘")

    # Heartbeat
    print()
    print("  ┌─ MISSION CONTROL HEARTBEAT " + "─" * 42 + "┐")
    print(_row("  │  Last Loop",         _ts(heartbeat.get("last_successful_loop"))))
    print(_row("  │  Last News Pull",    _ts(heartbeat.get("last_news_pull"))))
    print(_row("  │  Last Broker Send",  _ts(heartbeat.get("last_broker_send"))))
    print(_row("  │  Last Health Check", _ts(heartbeat.get("last_health_check"))))
    print(_row("  │  Loop Count",        heartbeat.get("loop_count", 0)))
    print(_row("  │  Error Count",       heartbeat.get("error_count", 0)))
    print("  └" + "─" * 70 + "┘")

    print()
    print(_bar())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TESTS  (20 / 20)
# ══════════════════════════════════════════════════════════════════════════════

class _StateManagerTests(unittest.TestCase):
    """
    20 smoke tests covering every major behaviour of state_manager.py.
    Run with:  python state_manager.py
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    def setUp(self) -> None:
        """Create a fresh temp directory and reset module globals for each test."""
        import tempfile as _tf
        self._tmpdir = _tf.mkdtemp(prefix="sm_test_")
        self._path   = os.path.join(self._tmpdir, "state.json")
        self._reset_module()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._reset_module()

    def _reset_module(self) -> None:
        """Reset all module-level singletons between tests."""
        global _memory_cache, _cache_dirty, _state_file_path, _plugins
        _memory_cache     = None
        _cache_dirty      = False
        _state_file_path  = self._path if hasattr(self, "_path") else _STATE_FILE_DEFAULT
        _plugins          = []

    def _fresh_load(self) -> Dict[str, Any]:
        return load_state(path_override=self._path)

    # ── Test 01 — fresh boot with no file ────────────────────────────────────
    def test_01_fresh_boot_no_file(self) -> None:
        state = self._fresh_load()
        self.assertIsInstance(state, dict)
        self.assertIn("active_positions", state)
        self.assertIn("kill_switch_state", state)
        self.assertEqual(state["schema_version"], STATE_SCHEMA_VERSION)

    # ── Test 02 — atomic save creates primary file ────────────────────────────
    def test_02_atomic_save_creates_file(self) -> None:
        state = self._fresh_load()
        save_state(state, path_override=self._path)
        self.assertTrue(os.path.exists(self._path))

    # ── Test 03 — saved_at timestamp is written ───────────────────────────────
    def test_03_saved_at_written(self) -> None:
        state = self._fresh_load()
        save_state(state, path_override=self._path)
        raw = _safe_read(self._path)
        self.assertIsNotNone(raw["saved_at"])
        self.assertIn("T", raw["saved_at"])

    # ── Test 04 — schema_version written to disk ──────────────────────────────
    def test_04_schema_version_on_disk(self) -> None:
        state = self._fresh_load()
        save_state(state, path_override=self._path)
        raw = _safe_read(self._path)
        self.assertEqual(raw["schema_version"], STATE_SCHEMA_VERSION)

    # ── Test 05 — load returns same data after save ───────────────────────────
    def test_05_roundtrip_positions(self) -> None:
        self._fresh_load()
        pos = {"AAPL": {"ticker": "AAPL", "quantity": 10, "avg_price": 175.0,
                        "broker": "alpaca", "asset_class": "EQUITY",
                        "open_timestamp": time.time()}}
        save_positions(pos)
        self._reset_module()
        loaded = load_positions()
        self.assertIn("AAPL", loaded)
        self.assertAlmostEqual(loaded["AAPL"]["avg_price"], 175.0)

    # ── Test 06 — backup file rotated on every save ───────────────────────────
    def test_06_backup_rotation(self) -> None:
        state = self._fresh_load()
        save_state(state, path_override=self._path)  # write primary
        save_state(state, path_override=self._path)  # should create backup
        self.assertTrue(os.path.exists(_backup_path(self._path)))

    # ── Test 07 — corruption recovery from backup ─────────────────────────────
    def test_07_corruption_recovery(self) -> None:
        state = self._fresh_load()
        state["active_positions"]["BTC"] = {"ticker": "BTC", "quantity": 0.5,
                                             "avg_price": 62000.0, "broker": "binance",
                                             "asset_class": "CRYPTO",
                                             "open_timestamp": time.time()}
        save_state(state, path_override=self._path)
        save_state(state, path_override=self._path)  # ensure backup exists

        # Corrupt the primary
        with open(self._path, "w", encoding="utf-8") as fh:
            fh.write("{{CORRUPTED JSON!!!")

        self._reset_module()
        recovered = self._fresh_load()
        self.assertIn("BTC", recovered.get("active_positions", {}))

    # ── Test 08 — both files missing returns default ──────────────────────────
    def test_08_both_files_missing_returns_default(self) -> None:
        state = self._fresh_load()
        self.assertEqual(state["active_positions"], {})
        self.assertFalse(state["kill_switch_state"]["active"])

    # ── Test 09 — migration no-op for current version ────────────────────────
    def test_09_migration_noop(self) -> None:
        data = _default_state()
        result = _apply_migrations(data)
        self.assertEqual(result["schema_version"], STATE_SCHEMA_VERSION)

    # ── Test 10 — future schema version passes through unchanged ──────────────
    def test_10_future_schema_passthrough(self) -> None:
        data = _default_state()
        data["schema_version"] = STATE_SCHEMA_VERSION + 99
        result = _apply_migrations(data)
        self.assertEqual(result["schema_version"], STATE_SCHEMA_VERSION + 99)

    # ── Test 11 — missing key backfilled on load ──────────────────────────────
    def test_11_missing_key_backfilled(self) -> None:
        state = self._fresh_load()
        # Manually strip a key and save raw
        del state["mission_control_heartbeat"]
        _atomic_write(self._path, state)
        self._reset_module()
        loaded = self._fresh_load()
        self.assertIn("mission_control_heartbeat", loaded)

    # ── Test 12 — thread-safe concurrent saves ────────────────────────────────
    def test_12_thread_safe_concurrent_saves(self) -> None:
        self._fresh_load()
        errors: List[Exception] = []

        def _worker(idx: int) -> None:
            try:
                pos = {f"TKR{idx}": {"ticker": f"TKR{idx}", "quantity": idx,
                                      "avg_price": float(idx * 10), "broker": "paper",
                                      "asset_class": "EQUITY",
                                      "open_timestamp": time.time()}}
                save_positions(pos)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertTrue(os.path.exists(self._path))

    # ── Test 13 — singleton cache reuse (no double disk read) ─────────────────
    def test_13_singleton_cache_reuse(self) -> None:
        s1 = self._fresh_load()
        s2 = load_state()          # should hit cache
        self.assertIs(s1, s2)

    # ── Test 14 — plugin fires after save ─────────────────────────────────────
    def test_14_plugin_fires_after_save(self) -> None:
        fired: List[Dict] = []
        register_plugin(lambda s: fired.append(s))
        state = self._fresh_load()
        save_state(state, path_override=self._path)
        self.assertEqual(len(fired), 1)
        self.assertIn("schema_version", fired[0])

    # ── Test 15 — plugin silenced after max errors ────────────────────────────
    def test_15_plugin_silenced_after_errors(self) -> None:
        call_count: List[int] = [0]

        def _bad_plugin(s: Dict) -> None:
            call_count[0] += 1
            raise RuntimeError("deliberate failure")

        register_plugin(_bad_plugin)
        state = self._fresh_load()
        for _ in range(_MAX_PLUGIN_ERRORS + 5):
            save_state(state, path_override=self._path)

        self.assertLessEqual(call_count[0], _MAX_PLUGIN_ERRORS)
        self.assertTrue(_plugins[0].silenced)

    # ── Test 16 — save/load cooldowns ─────────────────────────────────────────
    def test_16_cooldowns_roundtrip(self) -> None:
        self._fresh_load()
        exp = time.time() + 900
        save_cooldowns({"TSLA": exp})
        self._reset_module()
        cds = load_cooldowns()
        self.assertIn("TSLA", cds)
        self.assertAlmostEqual(cds["TSLA"], exp, places=2)

    # ── Test 17 — save/load kill switch ───────────────────────────────────────
    def test_17_kill_switch_roundtrip(self) -> None:
        self._fresh_load()
        ks = KillSwitchState(active=True, activated_at=time.time(),
                             reason="daily loss limit hit", triggered_by="risk_guardian")
        save_kill_switch(ks)
        self._reset_module()
        loaded = load_kill_switch()
        self.assertTrue(loaded["active"])
        self.assertEqual(loaded["reason"], "daily loss limit hit")

    # ── Test 18 — save/load analytics checkpoint ──────────────────────────────
    def test_18_analytics_roundtrip(self) -> None:
        self._fresh_load()
        ac = AnalyticsCheckpoint(
            rolling_pnl=[100.0, -50.0, 200.0],
            win_streak=3, total_trades=10, total_wins=7,
            peak_portfolio_usd=26500.0, max_drawdown_pct=1.2,
        )
        save_analytics_checkpoint(ac)
        self._reset_module()
        loaded = load_analytics_checkpoint()
        self.assertEqual(loaded["win_streak"], 3)
        self.assertAlmostEqual(loaded["peak_portfolio_usd"], 26500.0)

    # ── Test 19 — save/load heartbeat ─────────────────────────────────────────
    def test_19_heartbeat_roundtrip(self) -> None:
        self._fresh_load()
        now = time.time()
        hb = MissionControlHeartbeat(
            last_successful_loop=now, last_news_pull=now - 30,
            loop_count=42, error_count=1,
        )
        save_heartbeat(hb)
        self._reset_module()
        loaded = load_heartbeat()
        self.assertEqual(loaded["loop_count"], 42)
        self.assertAlmostEqual(loaded["last_successful_loop"], now, places=1)

    # ── Test 20 — print_state_summary runs without error ─────────────────────
    def test_20_print_state_summary(self) -> None:
        import io, contextlib
        self._fresh_load()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_state_summary(path_override=self._path)
        output = buf.getvalue()
        self.assertIn("MONSTER TRADING AI", output)
        self.assertIn("KILL SWITCH", output)
        self.assertIn("ANALYTICS", output)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_state_summary()

    print("═" * 72)
    print("  Running 20 smoke tests …".center(72))
    print("═" * 72)

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(_StateManagerTests)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    total   = result.testsRun
    passed  = total - len(result.failures) - len(result.errors)
    failed  = len(result.failures) + len(result.errors)

    print()
    print("═" * 72)
    if failed == 0:
        print(f"  ✓  {passed}/{total}  ALL CLEAR — state_manager.py is production-ready.")
    else:
        print(f"  ✗  {passed}/{total} passed — {failed} failure(s). See above.")
    print("═" * 72)
    print()

    sys.exit(0 if failed == 0 else 1)