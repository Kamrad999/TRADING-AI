# AMATIS FAILURE DISCIPLINE
## Phase 2.9999 — Institutional Failure Handling

**Date:** 2026-05-16

---

## Current State

**Bare except:** 2 (fixed in Phase 2.999)  
**Silent failures:** 3 identified  
**Structured exceptions:** Created in Phase 2.999

---

## Requirements

1. NO bare except ✅ (completed)
2. NO silent failures (in progress)
3. NO swallowed async exceptions
4. ALL critical failures emit events
5. ALL failures contain correlation ID, component, severity, recovery hints

---

## Implementation

### FailureContext

```python
@dataclass
class FailureContext:
    correlation_id: str
    component: str
    severity: ErrorSeverity
    recovery_hint: Optional[str]
    causal_chain: List[str]
```

### Recovery Policies

```python
class RecoveryPolicy(Enum):
    RETRY = "retry"
    FAIL_CLOSED = "fail_closed"
    DEGRADE = "degrade"
    PANIC = "panic"
```

### Panic Mode

```python
class PanicMode:
    """System-wide panic mode for catastrophic failures."""
    
    def __init__(self):
        self._panic_active = False
        self._panic_reason: Optional[str] = None
    
    def trigger(self, reason: str) -> None:
        self._panic_active = True
        self._panic_reason = reason
        # Emit panic event
        # Kill all trading
        # Enter safe state
```

---

## Status

**Exception hierarchy:** ✅ Created (Phase 2.999)  
**Failure context:** ⏳ Pending  
**Recovery policies:** ⏳ Pending  
**Panic mode:** ⏳ Pending

**Estimated Effort:** 16 hours

---

*SECTION 4 — FAILURE DISCIPLINE 🟡*
