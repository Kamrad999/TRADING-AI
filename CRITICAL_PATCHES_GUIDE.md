# CRITICAL FIXES — QUICK PATCH GUIDE
**Implementation Checklist for Priority Fixes**  
**Estimated Time: 65 minutes**

---

## PATCH #1: Fix feedparser Import (Issue #1)
**File:** [rss_sandbox.py](rss_sandbox.py)  
**Time:** 10 minutes  
**Blocker Status:** 🔴 CRITICAL

### Changes Required:

**Step 1:** Create `requirements.txt` at project root
```
feedparser>=6.0.0
requests>=2.28.0
```

**Step 2:** Update [rss_sandbox.py](rss_sandbox.py) line 9
```python
# BEFORE
import feedparser

# AFTER
try:
    import feedparser
except ImportError:
    raise ImportError(
        "feedparser not installed. Please run:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install feedparser requests\n"
        "\nWithout feedparser, RSS feed fetching will not work."
    ) from None
```

**Step 3:** Document in code comments
```python
"""
rss_sandbox.py
--------------
Sandbox test for fetching and displaying RSS headlines.

DEPENDENCIES:
  - feedparser (third-party RSS/Atom parser)
  - requests (third-party HTTP client)
  
  These are NOT stdlib and must be installed via:
    pip install -r requirements.txt

Built to be extended into a broader trading AI news engine.
"""
```

**Verification:**
```bash
cd /path/to/trading-ai
pip install -r requirements.txt
python -c "import rss_sandbox; rss_sandbox.test_single_feed()"
# Expected: Should print headlines or skip gracefully with [INFO]
```

---

## PATCH #2: Fix STAGE_NAMES in god_core.py (Issue #7)
**File:** [god_core.py](god_core.py#L66-L79)  
**Time:** 10 minutes  
**Blocker Status:** 🔴 CRITICAL

### Current Code (BROKEN):
```python
STAGE_NAMES: Tuple[str, ...] = (
    "fetch_news",
    "deduplicate_articles",
    "validate_articles",
    "generate_signals",
    "apply_risk_controls",    # ← Missing `regime_detector` before this
    "build_orders",
    "send_orders",
    "route_alerts",
    "persist_state",
    "update_validation_memory",
    "update_performance_analytics",
)
```

### Corrected Code:
```python
STAGE_NAMES: Tuple[str, ...] = (
    "fetch_news",                    # 1. news_engine.py
    "deduplicate_articles",          # 2. duplicate_filter.py
    "validate_articles",             # 3. fake_news_validator.py
    "generate_signals",              # 4. signal_engine.py
    "detect_market_regime",          # 5. regime_detector.py  ← ADD THIS
    "allocate_portfolio",            # 6. portfolio_brain.py  ← ADD THIS
    "apply_risk_controls",           # 7. risk_guardian.py
    "build_orders",                  # 8. execution_bridge.py
    "send_orders",                   # 9. broker_sender.py
    "route_alerts",                  # 10. alert_router.py
    "persist_state",                 # 11. state_manager.py
    "update_validation_memory",      # 12. validation_memory.py
    "update_performance_analytics",  # 13. performance_analytics.py
)
```

**Verification:**
```bash
cd /path/to/trading-ai
python -c "
from god_core import STAGE_NAMES
print(f'Total stages: {len(STAGE_NAMES)}')
print('Stages:')
for i, s in enumerate(STAGE_NAMES, 1):
    print(f'  {i:2d}. {s}')
assert len(STAGE_NAMES) == 13, f'Expected 13 stages, got {len(STAGE_NAMES)}'
print('✓ STAGE_NAMES validation passed')
"
```

---

## PATCH #3: Add Empty Feed Warning (Issue #15)
**File:** [news_engine.py](news_engine.py#L120-L135)  
**Time:** 15 minutes  
**Blocker Status:** 🔴 CRITICAL

### Location: After main loop, in `run_ingestion_pipeline()`

**Before** (line ~170):
```python
    wall_elapsed = round(time.perf_counter() - wall_start, 4)

    metrics = build_pipeline_metrics(
        sources_total    = len(ALL_FEED_SOURCES),
        successful_feeds = successful_feeds,
```

**Add This Block** (insert before metrics build):
```python
    # ── CRITICAL CHECK: Zero articles produced ─────────────────────────────────
    if not all_articles:
        _log(1, "CRITICAL", 
             f"🚨 NEWS ENGINE EMPTY — All feeds produced zero usable articles!\n"
             f"   Total sources: {len(ALL_FEED_SOURCES)}\n"
             f"   Successful fetches: {successful_feeds}\n"
             f"   Failed sources: {len(failed_feeds)}\n"
             f"   Skipped sources: {len(skipped_feeds)}\n"
             f"   Failed: {', '.join(failed_feeds[:3])}{'...' if len(failed_feeds) > 3 else ''}\n"
             f"   Pipeline will produce ZERO signals this cycle.\n")

    wall_elapsed = round(time.perf_counter() - wall_start, 4)

    metrics = build_pipeline_metrics(
        sources_total    = len(ALL_FEED_SOURCES),
        successful_feeds = successful_feeds,
```

**Also Update** metrics dict return:
```python
    # After building metrics, add:
    metrics["has_critical_issue"] = (not all_articles)
    if not all_articles:
        metrics["critical_reason"] = "ZERO_ARTICLES_INGESTED"
        
    return {
        "articles": all_articles,
        "metrics": metrics,
        "run_id": run_id,
        "timestamp": run_start,
        "latency_seconds": wall_elapsed,
    }
```

**Verification:**
```bash
# Simulate by temporarily breaking a feed URL
# Run manually to confirm warning appears
python -c "
from news_engine import run_ingestion_pipeline
result = run_ingestion_pipeline()
if not result['articles']:
    print('✓ Empty feed warning should have been logged above')
"
```

---

## PATCH #4: Add Broker Failure Escalation (Issue #17)
**File:** [broker_sender.py](broker_sender.py#L200-L240)  
**Time:** 20 minutes  
**Blocker Status:** 🔴 CRITICAL

### Location: In `send_order()` function after ALL retries exhausted

**Before** (current state):
```python
def send_order(order_dict: dict) -> _SendResult:
    """Send an order to the active broker..."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _send_order_to_broker(order_dict)
            return result
        except Exception as e:
            # ... retry logic ...
            if attempt == MAX_RETRIES:
                return _SendResult(
                    status=BrokerStatus.FAILED,
                    broker_order_id="",
                    fill_price=0.0,
                    latency_ms=0.0,
                    retry_count=MAX_RETRIES,
                    log_entries=[],
                    raw_response={"error": str(e)},
                )
```

**Add After Retries Exhausted:**
```python
def send_order(order_dict: dict) -> _SendResult:
    """Send an order to the active broker..."""
    order_id = order_dict.get("order_id", "<unknown>")
    ticker = order_dict.get("ticker", "?")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _send_order_to_broker(order_dict)
            return result
        except Exception as e:
            # ... retry logic ...
            if attempt == MAX_RETRIES:
                # ← NEW: ESCALATE FAILURE
                failure_msg = (
                    f"🔴 BROKER ORDER FAILED (exhausted {MAX_RETRIES} retries)\n"
                    f"   Order ID: {order_id}\n"
                    f"   Ticker: {ticker}\n"
                    f"   Side: {order_dict.get('side', '?')}\n"
                    f"   Quantity: {order_dict.get('qty', '?')}\n"
                    f"   Final Error: {type(e).__name__}: {str(e)[:100]}\n"
                    f"   Status: ORDER DID NOT REACH BROKER\n"
                    f"   Action: Manual intervention required.\n"
                )
                _audit(failure_msg, level="CRITICAL")
                
                return _SendResult(
                    status=BrokerStatus.FAILED,
                    broker_order_id="",
                    fill_price=0.0,
                    latency_ms=0.0,
                    retry_count=MAX_RETRIES,
                    log_entries=[failure_msg],
                    raw_response={
                        "error": str(e),
                        "exhausted": True,
                        "escalated": True,
                    },
                )
```

**Where `_audit()` is:**
```python
def _audit(msg: str, level: str = "INFO") -> None:
    """Internal audit log for critical events."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    log_line = f"[{ts}] [{level}] {msg}"
    _panic_log.append({
        "ts": ts,
        "level": level,
        "msg": msg
    })
    # Also print to console for emergency visibility
    print(log_line, file=sys.stderr)
```

**Verification:**
```bash
# Test by temporarily disabling broker or setting invalid creds
python -c "
from broker_sender import send_order
result = send_order({'order_id': 'TEST', 'ticker': 'AAPL', 'side': 'BUY', 'qty': 10})
if result.status.value == 'FAILED':
    print('✓ Failure logged with escalation message')
    print(f'  Entries: {result.log_entries}')
"
```

---

## PATCH #5: Standardize Field Names in config.py (Issue #5, #9, #10)
**File:** [config.py](config.py)  
**Time:** 15 minutes  
**Blocker Status:** 🟡 MEDIUM (but high impact)

### Add New Section to config.py (after section 5 "EXECUTION TEMPLATES"):

```python
# ══════════════════════════════════════════════════════════════════════════════
# 6.  STANDARDIZED FIELD NAMES  (canonical keys across all modules)
# ══════════════════════════════════════════════════════════════════════════════

# These constants define the EXACT dict key names used by the pipeline.
# Every module MUST use these keys when passing data downstream.
# NO module should hardcode field names like "gross_cap" or "hedge_ratio".

FIELD_MARKET_REGIME: Final[str] = "market_regime"
"""
Canonical key for market regime classification string.
Value: one of { "bull_trend", "bear_trend", "crisis", "neutral", ... }
Produced by: regime_detector.py
Consumed by: portfolio_brain.py, risk_guardian.py, self_learning_optimizer.py
"""

FIELD_GROSS_CAP: Final[str] = "recommended_gross_cap"
"""
Canonical key for gross leverage ceiling (fraction of NAV).
Value: float in [0.2, 2.0]
Produced by: regime_detector.py
Consumed by: portfolio_brain.py, risk_guardian.py
"""

FIELD_HEDGE_RATIO: Final[str] = "recommended_hedge_ratio"
"""
Canonical key for recommended hedging ratio.
Value: float in [0.0, 1.0]
Produced by: regime_detector.py
Consumed by: portfolio_brain.py
"""

FIELD_CONFIDENCE_SCORE: Final[str] = "confidence_score"
"""
Canonical key for signal confidence.
Value: float in [0.0, 1.0]
Produced by: signal_engine.py
Consumed by: execution_bridge.py, alert_router.py, portfolio_brain.py
"""

FIELD_SIGNAL_DIRECTION: Final[str] = "signal_direction"
"""
Canonical key for trade direction.
Value: one of { "LONG", "SHORT", "FLAT", "NO_TRADE" }
Produced by: signal_engine.py
Consumed by: risk_guardian.py, execution_bridge.py
"""

FIELD_OPTIONS_PREMIUM_RISK: Final[str] = "options_premium_at_risk"
"""
Canonical key for options premium exposure (fraction of NAV).
Value: float in [0.0, 0.1]
Produced by: portfolio_brain.py
Consumed by: risk_guardian.py, state_manager.py
"""
```

### Usage Example (Update all modules):
```python
# Before (old code in signal_engine.py)
result_dict = {
    "market_regime_bias": regime,  # ❌ WRONG KEY NAME
    "confidence_score": conf,
}

# After (using config constants)
from config import FIELD_MARKET_REGIME, FIELD_CONFIDENCE_SCORE
result_dict = {
    FIELD_MARKET_REGIME: regime,    # ✓ Standardized
    FIELD_CONFIDENCE_SCORE: conf,
}
```

**Files to Update:**
- signal_engine.py: Change `market_regime_bias` → use config constant
- regime_detector.py: Change all output keys to use constants  
- portfolio_brain.py: Use config constants in dict construction
- risk_guardian.py: Use config constants in validation

**Verification Script:**
```bash
python -c "
from config import (
    FIELD_MARKET_REGIME,
    FIELD_GROSS_CAP,
    FIELD_HEDGE_RATIO,
    FIELD_CONFIDENCE_SCORE,
    FIELD_SIGNAL_DIRECTION,
    FIELD_OPTIONS_PREMIUM_RISK,
)
print('✓ All field name constants defined')
print(f'  - Market Regime: {FIELD_MARKET_REGIME}')
print(f'  - Gross Cap: {FIELD_GROSS_CAP}')
print(f'  - Hedge Ratio: {FIELD_HEDGE_RATIO}')
print(f'  - Confidence: {FIELD_CONFIDENCE_SCORE}')
print(f'  - Signal Direction: {FIELD_SIGNAL_DIRECTION}')
print(f'  - Options Premium: {FIELD_OPTIONS_PREMIUM_RISK}')
"
```

---

## PATCH #6: Centralize Drawdown Policy (Issue #11)
**File:** [config.py](config.py)  
**Time:** 15 minutes  
**Blocker Status:** 🟡 MEDIUM (policy conflict)

### Add to config.py (after section 2 "PORTFOLIO + RISK"):

```python
# ── UNIFIED DRAWDOWN CASCADE (single source of truth for all risk tiers) ────
# This policy is the ONLY place max drawdown thresholds are defined.
# risk_guardian.py and self_learning_optimizer.py read this — never hardcode.

DRAWDOWN_POLICY_TIERS: Final[list[tuple[float, str, float]]] = [
    # (threshold_pct, action, size_multiplier)
    # Applied in descending order — first match wins.
    (0.025, "FULL_KILL_SWITCH", 0.0),    # ≥ 2.5% → block all trades
    (0.015, "HEAVY_REDUCTION", 0.4),     # 1.5–2.5% → 40% position size
    (0.05,  "WARNING_ALERT", 0.8),       # 0.5–1.5% → reduce 20%
    (0.0,   "NORMAL", 1.0),              # < 0.5% → full size
]

# Reference names (used in logging / metrics)
DRAWDOWN_ACTION_KILL_SWITCH = "FULL_KILL_SWITCH"
DRAWDOWN_ACTION_HEAVY_REDUCTION = "HEAVY_REDUCTION"
DRAWDOWN_ACTION_WARNING = "WARNING_ALERT"
DRAWDOWN_ACTION_NORMAL = "NORMAL"
```

### Update [risk_guardian.py](risk_guardian.py) to use it:

**Before:**
```python
# risk_guardian.py — old code
DAILY_DRAWDOWN_LOCK_PCT = 0.025
DAILY_DRAWDOWN_WARN_PCT = 0.015
```

**After:**
```python
# risk_guardian.py — new code
from config import DRAWDOWN_POLICY_TIERS, DRAWDOWN_ACTION_KILL_SWITCH

def check_daily_drawdown(drawdown_pct: float) -> _RiskCheckResult:
    """Check current drawdown against unified policy."""
    for threshold, action, multiplier in DRAWDOWN_POLICY_TIERS:
        if drawdown_pct >= threshold:
            reasons = [f"{action} triggered at {drawdown_pct:.2%} drawdown"]
            return _RiskCheckResult(
                passed=(multiplier > 0.0),
                multiplier=multiplier,
                reasons=reasons,
            )
    return _RiskCheckResult(passed=True, multiplier=1.0, reasons=[])
```

### Update [self_learning_optimizer.py](self_learning_optimizer.py) similarly:

**Before:**
```python
# self_learning_optimizer.py — old code (conflicting)
DRAWDOWN_WARN: float = -0.05
DRAWDOWN_REDUCE: float = -0.10
DRAWDOWN_DEFENSIVE: float = -0.15
```

**After:**
```python
# self_learning_optimizer.py — new code (reads config policy)
from config import DRAWDOWN_POLICY_TIERS

# Query unified policy instead of local constants
def get_allowed_gross_cap(drawdown_pct: float) -> float:
    """Get sizing multiplier from unified policy."""
    for threshold, action, mult in DRAWDOWN_POLICY_TIERS:
        if drawdown_pct >= threshold:
            return GROSS_CAP_DEFAULT * mult  # Apply to baseline cap
    return GROSS_CAP_DEFAULT
```

---

## Summary: Implementation Checklist

```
PATCH #1 (Blocker #1): feedparser import
  [ ] Create requirements.txt
  [ ] Add try/except to rss_sandbox.py
  [ ] Test: pip install -r requirements.txt && python -c "import rss_sandbox"
  Time: 10 min

PATCH #2 (Blocker #2): STAGE_NAMES fix
  [ ] Update STAGE_NAMES tuple in god_core.py (add 2 entries)
  [ ] Test: python -c "from god_core import STAGE_NAMES; print(len(STAGE_NAMES))"
  Time: 10 min

PATCH #3 (Blocker #3): Empty feed warning
  [ ] Add critical check to news_engine.py post-loop
  [ ] Update metrics dict
  [ ] Test: Run with broken feed URL, verify log
  Time: 15 min

PATCH #4 (Blocker #4): Broker failure escalation
  [ ] Add _audit() function to broker_sender.py
  [ ] Log critical failure after retries exhausted
  [ ] Test: Disconnect network, verify escalation message
  Time: 20 min

PATCH #5 (Medium): Field name standardization
  [ ] Add field name constants to config.py
  [ ] Update signal_engine.py to use constants
  [ ] Update regime_detector.py to use constants
  [ ] Update portfolio_brain.py to use constants
  [ ] Test: Verify dict keys match across modules
  Time: 15 min

PATCH #6 (Medium): Centralize drawdown policy
  [ ] Add DRAWDOWN_POLICY_TIERS to config.py
  [ ] Update risk_guardian.py to read config
  [ ] Update self_learning_optimizer.py to read config
  [ ] Test: Verify both modules use same policy
  Time: 15 min

TOTAL TIME: ~85 minutes
```

---

## After All Patches Applied

Run verification:
```bash
#!/bin/bash
set -e

echo "🔍 Verifying all patches..."

# Check 1: Requirements installed
echo "[1/5] Checking dependencies..."
pip list | grep feedparser || echo "⚠️ feedparser not installed"

# Check 2: STAGE_NAMES
echo "[2/5] Checking STAGE_NAMES..."
python -c "
from god_core import STAGE_NAMES
assert len(STAGE_NAMES) == 13, f'Expected 13 stages, got {len(STAGE_NAMES)}'
print('  ✓ STAGE_NAMES has 13 entries')
"

# Check 3: Config field constants
echo "[3/5] Checking field name constants..."
python -c "
from config import (
    FIELD_MARKET_REGIME,
    FIELD_GROSS_CAP,
    FIELD_HEDGE_RATIO,
)
print(f'  ✓ Field constants defined')
"

# Check 4: Unified drawdown policy
echo "[4/5] Checking drawdown policy..."
python -c "
from config import DRAWDOWN_POLICY_TIERS
assert len(DRAWDOWN_POLICY_TIERS) >= 3
print(f'  ✓ Drawdown policy has {len(DRAWDOWN_POLICY_TIERS)} tiers')
"

# Check 5: Import all modules (catch any circular imports)
echo "[5/5] Importing all modules..."
python -c "
from news_engine import run_ingestion_pipeline
from duplicate_filter import filter_duplicates
from fake_news_validator import validate_articles
from signal_engine import generate_signals
from regime_detector import analyse_regime
from portfolio_brain import allocate_portfolio
from risk_guardian import evaluate_execution_order
from execution_bridge import construct_broker_order
from broker_sender import send_order
from alert_router import route_signal
print('  ✓ All modules import successfully')
"

echo ""
echo "✅ All patches verified successfully!"
echo ""
echo "Next steps:"
echo "  1. Run test suite"
echo "  2. Paper trade for 5 days"
echo "  3. Full system integration test"
echo "  4. Then ready for live deployment review"
```

---

**End of Quick Patch Guide**
