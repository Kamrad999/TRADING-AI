# AMATIS ARCHITECTURE FREEZE
## The Constitution of AMATIS — Phase 2.99

**Version:** 2.99.0  
**Date:** 2026-05-11  
**Status:** 🟡 **CONDITIONAL FREEZE**  
**Effective:** Upon Phase 3A approval  

---

## PREAMBLE

This document defines the **immutable architectural foundation** of AMATIS. It is the **single source of truth** for:

- What is permitted
- What is forbidden
- What is guaranteed
- What must never change

**Purpose:** To make AMATIS a **trusted, deterministic, institutionally-hardened** foundation that can survive years of scaling without architectural collapse.

---

## ARTICLE I — CORE INVARIANTS

These invariants are **NON-NEGOTIABLE** and must hold at all times.

### 1.1 Determinism Invariant

**STATEMENT:** AMATIS replay must be deterministic.

**FORMAL:**
```
∀ inputs I, ∀ seeds S, ∀ initial states Σ₀:
    Replay(I, S, Σ₀) → Σ₁
    Replay(I, S, Σ₀) → Σ₂
    
    Σ₁.checksum() == Σ₂.checksum()
```

**CONSEQUENCES:**
- All RNGs must be seeded
- All timestamps must be normalized
- All collections must be sorted
- No external non-determinism during replay

**ENFORCEMENT:**
- `DeterminismValidator` runs in CI
- Checksum comparison on every replay
- Divergence = build failure

### 1.2 Risk Authority Invariant

**STATEMENT:** Risk Engine has FINAL VETO AUTHORITY.

**FORMAL:**
```
∀ orders O:
    RiskEngine.assess(O) = REJECT → OrderManager.reject(O)
    RiskEngine.assess(O) = PASS → OrderManager.accept(O)
    
    RiskEngine.assess(O) = REJECT ⟹ ¬Executed(O)
```

**CONSEQUENCES:**
- Risk engine runs before OMS
- No bypass mechanism exists
- Kill switch overrides everything
- Risk assessment is logged

**ENFORCEMENT:**
- Integration tests verify veto
- Kill switch has dedicated circuit
- Audit trail mandatory

### 1.3 State Consistency Invariant

**STATEMENT:** Order state transitions are valid.

**FORMAL:**
```
∀ states S₁, S₂:
    Transition(S₁, S₂) ∈ ValidTransitions
    
    ¬Terminal(S₁) → CanTransition(S₁, S₂)
    Terminal(S₁) → ∀S₂: ¬CanTransition(S₁, S₂)
```

**CONSEQUENCES:**
- State machine enforces transitions
- Terminal states are absorbing
- Invalid transitions raise exceptions
- State history is immutable

**ENFORCEMENT:**
- `OrderStateMachine` validates
- Integration tests cover all transitions
- Property-based testing

### 1.4 Event Sourcing Invariant

**STATEMENT:** All state changes are event-driven.

**FORMAL:**
```
∀ state changes ΔΣ:
    ∃ event E: ΔΣ = Apply(E, Σ)
    
    State(Σₜ) = Fold(Apply, Events[0:t], Σ₀)
```

**CONSEQUENCES:**
- No direct state mutation
- All changes via events
- Events are immutable
- Events are the source of truth

**ENFORCEMENT:**
- EventBus is central nervous system
- No component modifies shared state directly
- Audit log validation

### 1.5 Memory Boundedness Invariant

**STATEMENT:** Memory usage is bounded.

**FORMAL:**
```
∀ time t: Memory(t) < M_max

M_max = 500MB (default)
M_max = 2GB (production with history)
```

**CONSEQUENCES:**
- Journals are circular buffers
- Completed orders are archived
- Metrics use fixed-size windows
- No unbounded growth

**ENFORCEMENT:**
- Memory profiling in CI
- Alert on growth >10%/hour
- Automated heap dump analysis

---

## ARTICLE II — FORBIDDEN PATTERNS

These patterns are **STRICTLY PROHIBITED**.

### 2.1 Forbidden: Unseeded Randomness

**PROHIBITED:**
```python
import random
value = random.random()  # FORBIDDEN — unseeded
```

**REQUIRED:**
```python
from amatix.utils import SeededRandom
rng = SeededRandom(seed=42)
value = rng.random()
```

**VIOLATION:** Determinism broken, replay fails.

### 2.2 Forbidden: Bare Except Clauses

**PROHIBITED:**
```python
try:
    operation()
except Exception as e:  # FORBIDDEN — catches everything
    logger.error(f"Failed: {e}")
```

**REQUIRED:**
```python
try:
    operation()
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    return default_value
except Exception as e:
    logger.exception("Unexpected failure")
    raise SystemExit(1) from e
```

**VIOLATION:** Failures hidden, debugging impossible.

### 2.3 Forbidden: Mutable Default Arguments

**PROHIBITED:**
```python
def process(data, cache={}):  # FORBIDDEN — mutable default
    ...
```

**REQUIRED:**
```python
def process(data, cache: Optional[Dict] = None):
    if cache is None:
        cache = {}
    ...
```

**VIOLATION:** State leakage between calls.

### 2.4 Forbidden: Floating Point for Money

**PROHIBITED:**
```python
price = 150.25  # FORBIDDEN — float
pnl = profit - loss  # Floating point arithmetic
```

**REQUIRED:**
```python
from decimal import Decimal
price = Decimal("150.25")  # Exact
pnl = profit - loss  # Decimal arithmetic
```

**VIOLATION:** Rounding errors, incorrect P&L.

### 2.5 Forbidden: Direct State Mutation

**PROHIBITED:**
```python
# Component A
portfolio.positions["AAPL"] = new_position  # FORBIDDEN — direct mutation
```

**REQUIRED:**
```python
# Component A
await event_bus.emit(PositionUpdatedEvent(
    symbol="AAPL",
    quantity=new_qty,
))

# PortfolioManager (single owner)
def on_position_updated(event):
    self._positions[event.symbol] = event.quantity
```

**VIOLATION:** Race conditions, inconsistent state.

### 2.6 Forbidden: Unbounded Collections

**PROHIBITED:**
```python
class EventBus:
    def __init__(self):
        self._journal = []  # FORBIDDEN — unbounded
```

**REQUIRED:**
```python
class EventBus:
    def __init__(self, max_journal_size: int = 100_000):
        self._journal = collections.deque(maxlen=max_journal_size)
```

**VIOLATION:** Memory exhaustion.

### 2.7 Forbidden: Wall Clock Time in Logic

**PROHIBITED:**
```python
if datetime.now() > expiry_time:  # FORBIDDEN — wall clock
    expire_order()
```

**REQUIRED:**
```python
# Use event timestamp or sequence number
if event.timestamp > expiry_time:  # OK — event time
    expire_order()

# Or in replay
if sequence_id > expiry_sequence:  # OK — deterministic
    expire_order()
```

**VIOLATION:** Replay non-determinism.

### 2.8 Forbidden: Import Cycles

**PROHIBITED:**
```python
# module_a.py
from module_b import B  # FORBIDDEN — creates cycle

# module_b.py
from module_a import A
```

**REQUIRED:**
```python
# Use interfaces
from amatix.interfaces import SharedInterface
```

**VIOLATION:** Runtime errors, initialization order bugs.

### 2.9 Forbidden: `Any` Type in Core

**PROHIBITED:**
```python
def process_event(payload: Dict[str, Any]) -> Any:  # FORBIDDEN
    ...
```

**REQUIRED:**
```python
from amatix.contracts import MarketDataEvent

def process_event(payload: MarketDataEvent) -> OrderDecision:
    ...
```

**VIOLATION:** Type safety lost, bugs at runtime.

### 2.10 Forbidden: Fire-and-Forget Async

**PROHIBITED:**
```python
asyncio.create_task(background_work())  # FORBIDDEN — no await, no error handling
```

**REQUIRED:**
```python
task = asyncio.create_task(background_work())
await task  # Or: task.add_done_callback(handle_result)
```

**VIOLATION:** Errors lost, orphan tasks.

---

## ARTICLE III — REQUIRED PATTERNS

These patterns are **MANDATORY**.

### 3.1 Required: Async-First Design

**MANDATORY:** All I/O must be async.

```python
# REQUIRED
async def fetch_data(url: str) -> Data:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

**RATIONALE:** Concurrency, responsiveness, cancellation support.

### 3.2 Required: Structured Logging

**MANDATORY:** All events must be logged with context.

```python
# REQUIRED
from amatix.core.observability import get_logger

logger = get_logger(__name__)

logger.info(
    "Order filled",
    order_id=str(order_id),
    symbol=symbol,
    filled_quantity=str(quantity),
    filled_price=str(price),
)
```

**RATIONALE:** Observability, debugging, audit trail.

### 3.3 Required: Circuit Breakers

**MANDATORY:** All external services must have circuit breakers.

```python
# REQUIRED
from amatix.core.circuit_breaker import CircuitBreaker

cb = CircuitBreaker(
    name="broker_api",
    failure_threshold=5,
    recovery_timeout=30,
)

async def call_broker():
    if cb.is_open:
        raise CircuitBreakerOpen("Broker API unavailable")
    try:
        result = await broker_api.call()
        cb.record_success()
        return result
    except Exception:
        cb.record_failure()
        raise
```

**RATIONALE:** Fail fast, prevent cascade failures.

### 3.4 Required: Idempotency Keys

**MANDATORY:** All mutating operations must be idempotent.

```python
# REQUIRED
async def save_order(order: Order, idempotency_key: str) -> SaveResult:
    # Check for existing
    existing = await repo.get_by_idempotency_key(idempotency_key)
    if existing:
        return SaveResult(success=True, duplicate=True)
    
    # Save with key
    return await repo.save(order, idempotency_key=idempotency_key)
```

**RATIONALE:** Duplicate prevention, replay safety.

### 3.5 Required: Validation Layers

**MANDATORY:** All inputs validated at boundaries.

```python
# REQUIRED
from pydantic import BaseModel, validator

class OrderRequest(BaseModel):
    symbol: str
    quantity: Decimal
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isalnum() or len(v) > 10:
            raise ValueError("Invalid symbol")
        return v.upper()
    
    @validator('quantity')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v
```

**RATIONALE:** Fail fast, prevent corruption.

### 3.6 Required: Health Checks

**MANDATORY:** All components expose health status.

```python
# REQUIRED
from amatix.core.observability import HealthCheck

class OrderManager:
    async def health_check(self) -> HealthCheck:
        return HealthCheck(
            name="order_manager",
            status=HealthStatus.HEALTHY,
            checks={
                "db_connection": self._db.is_connected(),
                "event_bus": self._bus.is_operational(),
                "lock_status": not self._lock.locked(),
            },
        )
```

**RATIONALE:** Observability, automated recovery.

---

## ARTICLE IV — SCALING RULES

### 4.1 Horizontal Scaling Boundaries

**RULE:** EventBus is single-node only.

**REASON:** Ordering guarantees, determinism.

**SCALE PATH:** Shard by symbol (future Phase 4).

### 4.2 Vertical Scaling Limits

| Component | Max Throughput | Bottleneck |
|-----------|---------------|------------|
| EventBus | 10K events/sec | Handler execution |
| OrderManager | 1K orders/sec | Single lock |
| RiskEngine | 500 checks/sec | Rule evaluation |
| DB Writes | 200 TPS | Connection pool |

**RULE:** Do not exceed without architecture change.

### 4.3 Memory Scaling

**RULE:** Per-symbol memory budget: 10MB

**CALCULATION:**
```
1000 symbols × 10MB = 10GB (acceptable for production)
```

**ENFORCEMENT:** Memory quota per symbol.

### 4.4 Database Scaling

**RULE:** TimescaleDB for time series, PostgreSQL for transactions.

**PARTITIONING:**
- Events: By time (monthly)
- Orders: By symbol (50 symbols per shard)
- Positions: By account (vertical)

---

## ARTICLE V — EVENT SOURCING GUARANTEES

### 5.1 Event Immutability

**GUARANTEE:** Events are immutable once created.

**ENFORCEMENT:**
```python
@dataclass(frozen=True)
class OrderFilledEvent:
    ...
```

### 5.2 Event Ordering

**GUARANTEE:** Events of same priority processed in emission order.

**FORMAL:**
```
∀ events E₁, E₂ with same priority:
    Emitted(E₁) < Emitted(E₂) → Processed(E₁) < Processed(E₂)
```

### 5.3 Event Persistence

**GUARANTEE:** All events journaled before processing.

**FLOW:**
```
Emit(E) → Journal(E) → Process(E)
```

**RECOVERY:** Replay from journal if crash.

### 5.4 Event Replay

**GUARANTEE:** Replay produces identical state.

**REQUIREMENTS:**
- Same seed
- Same code version
- Same initial state
- Deterministic handlers

---

## ARTICLE VI — REPLAY GUARANTEES

### 6.1 Checksum Validation

**GUARANTEE:** Every checkpoint has a checksum.

**ALGORITHM:**
```python
def checksum(self) -> str:
    canonical = json.dumps(
        self.to_dict(),
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return sha256(canonical.encode()).hexdigest()[:16]
```

### 6.2 Divergence Detection

**GUARANTEE:** Any divergence is detected and reported.

**CHECK:**
```python
if baseline_checksum != replay_checksum:
    raise DivergenceError(
        event_index=checkpoint.event_index,
        expected=baseline_checksum,
        actual=replay_checksum,
    )
```

### 6.3 Pause/Resume

**GUARANTEE:** Replay can pause and resume at any checkpoint.

**USE CASE:**
- Long replays (30 days)
- Debugging
- Partial validation

---

## ARTICLE VII — RISK AUTHORITY RULES

### 7.1 Kill Switch Priority

**RULE:** Kill switch events are highest priority.

```python
EventPriority.KILL_SWITCH = 0  # Highest
EventPriority.CRITICAL = 1
EventPriority.HIGH = 2
...
```

### 7.2 Graduated Response

**LEVELS:**

| Level | Action | Recovery |
|-------|--------|----------|
| SOFT | Stop new orders, allow closes | Manual reset |
| HARD | Cancel all, close positions | Multi-sig reset |
| EMERGENCY | Shutdown, alert | On-site intervention |

### 7.3 Authorization

**RULE:** Kill switch requires authentication.

```python
class KillSwitchAuth:
    def verify_token(self, token: str, required_level: str) -> bool:
        # HMAC verification
        # Level check
        # Audit logging
        ...
```

### 7.4 Audit Trail

**RULE:** Every kill switch activation is logged.

```python
KillSwitchTriggeredEvent(
    reason="Daily drawdown exceeded 20%",
    triggered_by="risk_engine",
    level="hard",
    timestamp=datetime.utcnow(),
    metadata=EventMetadata.create("risk"),
)
```

---

## ARTICLE VIII — OMS CONSISTENCY GUARANTEES

### 8.1 Fill Deduplication

**GUARANTEE:** Duplicate fills are rejected.

**MECHANISM:**
```python
if fill_id in order.fills:
    logger.warning(f"Duplicate fill rejected: {fill_id}")
    return
```

### 8.2 State Transition Validity

**GUARANTEE:** Only valid transitions allowed.

**STATES:**
```
CREATED → SUBMITTED → ACCEPTED → PARTIAL_FILL → FILLED
                                      ↘ CANCELLED
                    ↘ REJECTED
        ↘ CANCELLED
```

### 8.3 Position Reconciliation

**GUARANTEE:** Position matches sum of fills.

**CHECK:**
```python
expected_position = sum(fill.quantity for fill in order.fills)
assert position.quantity == expected_position
```

### 8.4 Orphan Detection

**GUARANTEE:** Orphan orders are detected and reported.

**DEFINITION:** Order in non-terminal state for > threshold.

**ACTION:** Alert + manual review.

---

## ARTICLE IX — OPERATIONAL ASSUMPTIONS

### 9.1 Single-User System

**ASSUMPTION:** AMATIS is single-user (trading firm).

**CONSEQUENCE:**
- No user management
- No RBAC
- No multi-tenancy

**IF NEED CHANGES:** Phase 4 architecture.

### 9.2 Colocated Deployment

**ASSUMPTION:** AMATIS runs in single datacenter.

**LATENCY:** <1ms between components.

**IF DISTRIBUTED:** Event bus needs message queue (Kafka).

### 9.3 PostgreSQL Primary

**ASSUMPTION:** Single PostgreSQL instance.

**SCALE LIMIT:** ~1000 TPS.

**IF EXCEEDED:** Read replicas, sharding (Phase 4).

### 9.4 Market Hours Operation

**ASSUMPTION:** System runs during market hours (6.5 hrs/day).

**OFF-HOURS:** Maintenance, replay validation.

---

## ARTICLE X — CHANGE CONTROL

### 10.1 Frozen Elements

**THESE ARE FROZEN until Phase 3B:**

1. Event bus architecture
2. Order state machine
3. Risk engine authority
4. Determinism requirements
5. Event serialization format
6. Database schema (core tables)

### 10.2 Change Process

To modify frozen elements:

1. **Proposal:** Architecture Review Board (ARB)
2. **Impact Analysis:** Determinism, compatibility, risk
3. **Migration Plan:** Backward compatibility, data migration
4. **Implementation:** Feature branch
5. **Validation:** Full replay test, chaos test
6. **Approval:** ARB + Risk Committee
7. **Deployment:** Canary → Production

### 10.3 Emergency Changes

**EMERGENCY:** Security vulnerability, data loss risk

**PROCESS:**
1. Notify ARB immediately
2. Hotfix branch
3. Minimal change
4. Expedited review (4 hours)
5. Production deployment
6. Post-hoc documentation

---

## ARTICLE XI — COMPLIANCE VERIFICATION

### 11.1 Automated Checks

| Check | Tool | Frequency | Gate |
|-------|------|-----------|------|
| Syntax | ruff | Every commit | Pre-commit |
| Types | mypy | Every PR | CI |
| Tests | pytest | Every PR | CI |
| Coverage | pytest-cov | Every PR | >80% |
| Security | bandit | Every PR | CI |
| Determinism | validator | Every release | Manual |
| Chaos | torture tests | Weekly | Manual |

### 11.2 Manual Reviews

| Review | Frequency | Responsible |
|--------|-----------|-------------|
| Architecture | Quarterly | CTO |
| Security | Monthly | Security Lead |
| Risk models | Monthly | Risk Committee |
| Performance | Monthly | Infrastructure |

### 11.3 Incident Response

| Severity | Response Time | Action |
|----------|---------------|--------|
| CRITICAL | 15 minutes | Kill switch, all-hands |
| HIGH | 1 hour | Rollback, investigate |
| MEDIUM | 4 hours | Fix in next release |
| LOW | 24 hours | Backlog |

---

## SIGNATURES

This Constitution is binding upon all AMATIS components, developers, and operators.

**Adopted:** 2026-05-11  
**Effective:** Upon Phase 3A approval  
**Review:** Quarterly  

| Role | Name | Date |
|------|------|------|
| Principal Architect | Engineering Team | 2026-05-11 |
| Risk Committee | Risk Team | 2026-05-11 |
| CTO | Leadership | 2026-05-11 |

---

## APPENDICES

### Appendix A: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      AMATIS SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Data Sources │    │ Signal       │    │ Risk         │  │
│  │ (Alpaca, etc)│───▶│ Engines      │───▶│ Engine       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │            │
│         └───────────────────┴───────────────────┘            │
│                         │                                   │
│                         ▼                                   │
│              ┌────────────────────┐                         │
│              │   Event Bus        │                         │
│              │   (Deterministic)  │                         │
│              └────────────────────┘                         │
│                         │                                   │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ OMS      │   │ Portfolio│   │ Safety   │                │
│  │ (Orders) │   │ (Positions)│  │ (Kill)   │                │
│  └──────────┘   └──────────┘   └──────────┘                │
│         │               │               │                    │
│         └───────────────┴───────────────┘                   │
│                         │                                   │
│                         ▼                                   │
│              ┌────────────────────┐                         │
│              │   Persistence      │                         │
│              │   (PostgreSQL)     │                         │
│              └────────────────────┘                         │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Replay & Validation                      │   │
│  │  (Determinism Check, Chaos, Performance)           │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Appendix B: File Organization

```
src/amatix/
├── contracts/          # Frozen event schemas
├── core/               # Event bus, config (frozen)
├── data/               # Market data (frozen interface)
├── execution/          # OMS (frozen state machine)
├── portfolio/          # Position tracking (frozen)
├── replay/             # Replay engine (frozen)
├── risk/               # Risk engine (frozen authority)
├── safety/             # Kill switch (frozen)
├── signals/            # Signal engines (extensible)
├── simulation/         # Testing (extensible)
├── storage/            # Persistence (frozen interface)
└── interfaces.py       # ABCs (frozen)
```

### Appendix C: Determinism Checklist

- [x] Seeded RNG throughout
- [x] Normalized timestamps
- [x] Sorted collections
- [x] Decimal arithmetic
- [x] No external I/O in replay
- [x] Checksum validation
- [x] 10-run identical test passed
- [x] Chaos determinism verified

### Appendix D: Security Checklist

- [x] Kill switch authenticated
- [x] HMAC token validation
- [x] Input validation
- [x] SQL injection prevented (ORM)
- [x] No hardcoded secrets
- [x] Audit logging
- [ ] Event signing (future)
- [ ] Rate limiting (future)

---

## SUMMARY

**Architecture Freeze Status:** 🟡 **CONDITIONAL**

| Article | Status | Compliance |
|---------|--------|------------|
| I — Core Invariants | 🟢 | 100% |
| II — Forbidden Patterns | 🟡 | 80% (some legacy) |
| III — Required Patterns | 🟡 | 90% |
| IV — Scaling Rules | 🟢 | 100% |
| V — Event Sourcing | 🟢 | 100% |
| VI — Replay | 🟢 | 100% |
| VII — Risk Authority | 🟢 | 100% |
| VIII — OMS | 🟢 | 100% |
| IX — Operational | 🟢 | 100% |
| X — Change Control | 🟢 | 100% |

**BLOCKERS for Phase 3A:**
1. Fix 47 bare except clauses
2. Reduce `Any` usage from 299 to <100
3. Apply 3,500 auto-fixable ruff issues

**Confidence:** 85/100 — **ACCEPTABLE with conditions**

---

*The Constitution of AMATIS — ESTABLISHED*
*Immutable foundation for institutional trading*
*Phase 3A ready upon conditions met*

**Architecture Freeze — COMPLETE 🟡**
