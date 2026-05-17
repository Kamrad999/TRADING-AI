# AMATIS HARDENING PLAN — PHASE 2.75
## Institutional-Grade Implementation Roadmap

**Objective:** Convert findings from VERIFICATION_AUDIT into hardened, production-ready code.

**Timeline:** 4 weeks  
**Priority:** CRITICAL fixes first, then HIGH, then testing  

---

## WEEK 1 — CRITICAL FIXES

### Day 1-2: Event Bus Hardening

#### Task 1.1: Bounded Event Journal
**File:** `src/amatix/core/event_bus.py`

**Requirements:**
- Max journal size: 100,000 events (configurable)
- Overflow strategy: async write to SQLite disk buffer
- Automatic compression of events > 24 hours old
- Memory pressure monitoring

**Implementation:**
```python
class EventJournal:
    def __init__(self, max_memory_size: int = 100000):
        self._memory_buffer = deque(maxlen=max_memory_size)
        self._disk_buffer = SQLiteEventStore("./journal_overflow.db")
        self._compression_threshold = 24 * 60 * 60  # 24 hours
```

**Tests:**
- Journal fills to limit, overflows to disk
- Replay reads from both memory and disk
- Compression reduces memory by 80%+

---

#### Task 1.2: Fire-and-Forgot Elimination
**Files:** 
- `src/amatix/risk/engine.py`
- `src/amatix/core/orchestrator.py`
- `src/amatix/data/market/stream_manager.py`

**Requirements:**
- All `asyncio.create_task()` must be tracked
- Emergency events must be awaited with timeout
- Failed emissions must retry with backoff
- Shutdown must wait for all pending tasks

**Implementation:**
```python
class TaskManager:
    def __init__(self):
        self._pending: Set[asyncio.Task] = set()
    
    def create_tracked(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task
    
    async def wait_all(self, timeout: float = 30.0):
        await asyncio.wait_for(
            asyncio.gather(*self._pending, return_exceptions=True),
            timeout=timeout
        )
```

---

#### Task 1.3: Backpressure Handling
**File:** `src/amatix/core/event_bus.py`

**Requirements:**
- Max queue depth: 10,000 events
- Backpressure strategies: shed load, block, or error
- Circuit breaker on sustained backpressure
- Metrics on dropped events

**Implementation:**
```python
class BackpressureController:
    def __init__(self, max_depth: int = 10000):
        self._max_depth = max_depth
        self._current_depth = 0
        self._shed_counter = 0
    
    def acquire_slot(self) -> bool:
        if self._current_depth >= self._max_depth:
            self._shed_counter += 1
            return False
        self._current_depth += 1
        return True
```

---

### Day 3-4: OMS Hardening

#### Task 2.1: Fill Deduplication
**File:** `src/amatix/execution/oms/order_manager.py`

**Requirements:**
- Track all processed execution_ids
- Reject duplicate fills
- Alert on duplicate detection
- Persist execution_id set to database

**Implementation:**
```python
@dataclass
class OrderEntry:
    # ... existing fields ...
    processed_execution_ids: Set[str] = field(default_factory=set)

async def update_fill(self, order_id: UUID, execution: Execution) -> bool:
    async with self._lock:
        entry = self._orders.get(order_id)
        if not entry:
            raise ValueError(f"Order {order_id} not found")
        
        # Deduplication check
        if execution.execution_id in entry.processed_execution_ids:
            logger.warning("Duplicate fill rejected", 
                         execution_id=execution.execution_id)
            get_metrics().counter("oms_duplicate_fills_rejected")
            return False
        
        entry.processed_execution_ids.add(execution.execution_id)
        # ... rest of fill logic
```

---

#### Task 2.2: Broker Reconciliation
**File:** `src/amatix/execution/oms/order_manager.py`

**Requirements:**
- Periodic reconciliation task (every 60s)
- Query broker for unknown order statuses
- Detect orphaned orders (>60s in SUBMITTED)
- Alert on discrepancies

**Implementation:**
```python
async def reconcile_with_broker(self, broker: ExecutionEngine) -> ReconciliationReport:
    """Reconcile OMS state with broker state."""
    discrepancies = []
    
    async with self._lock:
        for entry in self._orders.values():
            if entry.state_machine.current_state == OrderState.SUBMITTED:
                # Check if orphaned
                elapsed = (datetime.utcnow() - entry.updated_at).total_seconds()
                if elapsed > 60:
                    # Query broker
                    broker_status = await broker.get_order_status(
                        entry.broker_order_id
                    )
                    if broker_status != OrderStatus.SUBMITTED:
                        discrepancies.append({
                            "order_id": entry.order_id,
                            "oms_state": entry.state_machine.current_state,
                            "broker_state": broker_status,
                            "orphaned_for_seconds": elapsed,
                        })
    
    return ReconciliationReport(discrepancies=discrepancies)
```

---

#### Task 2.3: Partial Fill Torture Tests
**File:** `tests/torture/test_oms_fills.py`

**Test Scenarios:**
```python
async def test_partial_fill_sequence():
    """Test 1%, 5%, 10%, remainder fill sequence."""
    order = create_order(qty=100)
    entry = await oms.create_order(order)
    
    # Fill 1
    await oms.update_fill(entry.order_id, create_fill(qty=1, price=100))
    assert entry.filled_quantity == 1
    assert entry.remaining_quantity == 99
    
    # Fill 5
    await oms.update_fill(entry.order_id, create_fill(qty=5, price=101))
    assert entry.filled_quantity == 6
    assert entry.remaining_quantity == 94
    
    # ... continue to full fill

async def test_duplicate_fill_rejection():
    """Test that duplicate execution_ids are rejected."""
    order = create_order(qty=100)
    entry = await oms.create_order(order)
    
    execution_id = "exec_123"
    fill = create_fill(qty=10, execution_id=execution_id)
    
    # First fill accepted
    assert await oms.update_fill(entry.order_id, fill)
    
    # Duplicate rejected
    assert not await oms.update_fill(entry.order_id, fill)

async def test_out_of_order_fill_delivery():
    """Test fills arriving out of chronological order."""
    # This can happen with network delays
    pass

async def test_fill_exceeding_order_quantity():
    """Test handling of broker bug: fill > order qty."""
    order = create_order(qty=100)
    entry = await oms.create_order(order)
    
    # Broker bug: says 110 filled
    with pytest.raises(FillValidationError):
        await oms.update_fill(entry.order_id, create_fill(qty=110))
```

---

### Day 5-7: Risk Engine Hardening

#### Task 3.1: Kill Switch Hardening
**File:** `src/amatix/risk/engine.py`

**Requirements:**
- Synchronous (awaited) emission with timeout
- Retry logic with exponential backoff
- Fallback to direct component notification
- Immutable audit trail

**Implementation:**
```python
async def _activate_kill_switch(self, reason: str) -> bool:
    """Activate kill switch with guaranteed delivery."""
    if self._kill_switch_active:
        return True
    
    self._kill_switch_active = True
    
    # Emit with guaranteed delivery
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(
                self._event_bus.emit_new(
                    EventType.KILL_SWITCH_TRIGGERED,
                    {
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat(),
                        "drawdown": self._current_snapshot.current_drawdown 
                                   if self._current_snapshot else None,
                    },
                    priority=EventPriority.CRITICAL,
                    source="risk_engine",
                ),
                timeout=5.0  # 5 second timeout for critical event
            )
            logger.critical("Kill switch activated and propagated", reason=reason)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Kill switch emission timeout (attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Kill switch emission failed (attempt {attempt + 1}): {e}")
    
    # Fallback: direct component notification
    logger.critical("Kill switch event emission failed after retries, using fallback")
    await self._fallback_kill_notification(reason)
    return True
```

---

#### Task 3.2: Parallel Rule Evaluation
**File:** `src/amatix/risk/engine.py`

**Requirements:**
- Evaluate independent rules in parallel
- Critical rules first
- Timeout per rule
- Aggregate results

**Implementation:**
```python
async def assess_order(
    self,
    order: Order,
    portfolio: Dict[str, Any],
    market: Dict[str, Any],
) -> RiskAssessment:
    start_time = time.time()
    
    # Check kill switch first (blocking)
    if self._kill_switch_active:
        return self._create_kill_switch_assessment(order)
    
    # Separate critical and non-critical rules
    critical_rules = [r for r in self._rules if r.severity == RiskSeverity.CRITICAL]
    other_rules = [r for r in self._rules if r.severity != RiskSeverity.CRITICAL]
    
    # Evaluate critical rules sequentially first
    assessment = RiskAssessment.create(verdict=RiskVerdict.APPROVED)
    for rule in critical_rules:
        violation = await self._evaluate_rule_with_timeout(rule, order, portfolio, market)
        if violation:
            assessment.violations.append(violation)
            if rule.block_on_violation:
                assessment.verdict = RiskVerdict.REJECTED
                return assessment
    
    # Evaluate other rules in parallel
    if other_rules:
        violation_tasks = [
            self._evaluate_rule_with_timeout(rule, order, portfolio, market)
            for rule in other_rules
        ]
        violations = await asyncio.gather(*violation_tasks, return_exceptions=True)
        
        for violation in violations:
            if isinstance(violation, Exception):
                logger.error(f"Rule evaluation error: {violation}")
                continue
            if violation:
                assessment.violations.append(violation)
    
    # Calculate final risk score
    assessment.risk_score = self._calculate_risk_score(assessment.violations)
    
    return assessment

async def _evaluate_rule_with_timeout(
    self,
    rule: BaseRiskRule,
    order: Order,
    portfolio: Dict[str, Any],
    market: Dict[str, Any],
    timeout: float = 1.0,
) -> Optional[RiskViolation]:
    """Evaluate a single rule with timeout."""
    try:
        return await asyncio.wait_for(
            rule.evaluate(order, portfolio, market),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Rule {rule.name} evaluation timeout")
        return RiskViolation(
            rule_name=rule.name,
            severity=RiskSeverity.CRITICAL,
            message=f"Rule evaluation timeout after {timeout}s",
            current_value="timeout",
            limit_value=f"{timeout}s",
        )
```

---

## WEEK 2 — HIGH SEVERITY FIXES

### Day 8-10: Database Hardening

#### Task 4.1: Transaction Boundaries
**File:** `src/amatix/storage/postgres/repositories/order_repository.py`

**Implementation:**
```python
class OrderRepository:
    async def save_order_with_fills(
        self,
        order: OrderRecord,
        fills: List[FillRecord],
    ) -> None:
        """Save order and fills atomically."""
        async with self._session.begin():
            self._session.add(order)
            for fill in fills:
                self._session.add(fill)
            # Both succeed or both fail
```

---

#### Task 4.2: Optimistic Locking
**File:** `src/amatix/storage/postgres/models.py`

**Implementation:**
```python
class OrderRecord(Base):
    # ... existing fields ...
    version = Column(Integer, default=0, nullable=False)
    
    __mapper_args__ = {
        "version_id_col": version
    }
```

---

#### Task 4.3: Repository Implementations
**Files:**
- `src/amatix/storage/postgres/repositories/order_repository.py`
- `src/amatix/storage/postgres/repositories/signal_repository.py`
- `src/amatix/storage/postgres/repositories/position_repository.py`

---

### Day 11-12: Auth & Security

#### Task 5.1: Kill Switch Auth
**File:** `src/amatix/risk/engine.py`

**Implementation:**
```python
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend

class KillSwitchAuth:
    def __init__(self, secret_key: bytes):
        self._secret = secret_key
    
    def generate_token(self, user_id: str, expiry: datetime) -> str:
        """Generate HMAC-signed token with expiry."""
        payload = f"{user_id}:{expiry.isoformat()}"
        signature = hmac.HMAC(self._secret, hashes.SHA256(), backend=default_backend())
        signature.update(payload.encode())
        return f"{payload}:{signature.finalize().hex()}"
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify token, return user_id if valid."""
        try:
            payload, signature_hex = token.rsplit(":", 1)
            user_id, expiry_str = payload.split(":", 1)
            expiry = datetime.fromisoformat(expiry_str)
            
            if datetime.utcnow() > expiry:
                return None
            
            # Verify signature
            expected = hmac.HMAC(self._secret, hashes.SHA256(), backend=default_backend())
            expected.update(payload.encode())
            
            if hmac.compare_digest(expected.finalize().hex(), signature_hex):
                return user_id
            return None
        except Exception:
            return None
```

---

### Day 13-14: Observability

#### Task 6.1: Latency Histograms
**File:** `src/amatix/core/observability.py`

**Implementation:**
```python
from prometheus_client import Histogram

# Histograms for latency tracking
RISK_ASSESSMENT_LATENCY = Histogram(
    "risk_assessment_latency_seconds",
    "Risk assessment latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

EVENT_PROCESSING_LATENCY = Histogram(
    "event_processing_latency_seconds",
    "Event processing latency",
    ["event_type"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)
```

---

## WEEK 3 — TESTING EXPANSION

### Day 15-17: Stress Tests

#### Task 7.1: Event Bus Stress Test
**File:** `tests/stress/test_event_bus.py`

**Test:**
```python
async def test_event_storm():
    """Inject 50,000 events in 5 seconds."""
    bus = EventBus()
    received = 0
    
    @bus.on(EventType.SIGNAL_GENERATED)
    async def handler(event):
        nonlocal received
        received += 1
    
    # Inject events
    start = time.time()
    for i in range(50000):
        await bus.emit_new(
            EventType.SIGNAL_GENERATED,
            {"index": i},
        )
    
    # Wait for processing
    await asyncio.sleep(2)
    
    duration = time.time() - start
    rate = received / duration
    
    assert received == 50000, f"Only {received}/{50000} events processed"
    assert rate > 5000, f"Rate {rate} below threshold"
```

---

#### Task 7.2: Memory Pressure Test
**File:** `tests/stress/test_memory.py`

**Test:**
```python
async def test_journal_memory_bounds():
    """Verify journal doesn't exceed memory limit."""
    bus = EventBus(max_journal_size=1000)
    
    for i in range(10000):
        await bus.emit_new(EventType.SIGNAL_GENERATED, {"index": i})
    
    metrics = bus.get_metrics()
    assert metrics["journal_size"] <= 1000
    assert metrics["journal_overflow_count"] == 9000
```

---

### Day 18-19: Property-Based Tests

#### Task 8.1: State Machine Properties
**File:** `tests/property/test_state_machine.py`

**Test:**
```python
from hypothesis import given, strategies as st

@given(st.lists(st.sampled_from(OrderState), min_size=1, max_size=20))
def test_state_machine_valid_transitions_only(states):
    """Verify only valid transitions are accepted."""
    sm = OrderStateMachine()
    
    for state in states:
        if sm.can_transition(state):
            sm.transition(state)
        else:
            with pytest.raises(InvalidStateTransition):
                sm.transition(state)
```

---

### Day 20-21: Chaos Tests

#### Task 9.1: Random Failure Injection
**File:** `tests/chaos/test_random_failures.py`

**Test:**
```python
async def test_random_broker_disconnects():
    """Randomly disconnect broker during operations."""
    for _ in range(100):
        order = create_random_order()
        entry = await oms.create_order(order)
        
        # Randomly disconnect
        if random.random() < 0.3:
            await broker.disconnect()
            await asyncio.sleep(random.uniform(0.1, 1.0))
            await broker.connect()
        
        # Verify order integrity
        assert await oms.get_order(entry.order_id) is not None
```

---

## WEEK 4 — INTEGRATION & VALIDATION

### Day 22-24: Integration Tests

#### Task 10.1: End-to-End Flow
**File:** `tests/integration/test_full_flow.py`

**Test:**
```python
async def test_market_data_to_fill():
    """Complete flow: market data → signal → order → fill."""
    # Setup system
    app = AMATISApplication()
    await app.initialize()
    
    # Inject market data
    await app._event_bus.emit_new(
        EventType.MARKET_DATA_RECEIVED,
        {"symbol": "AAPL", "price": 150.0},
    )
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Verify signal generated
    signals = await signal_repository.get_recent()
    assert len(signals) > 0
    
    # Verify order created
    orders = await order_repository.get_active()
    assert len(orders) > 0
    
    await app.shutdown()
```

---

### Day 25-26: Replay Determinism

#### Task 11.1: Deterministic Replay Validation
**File:** `tests/replay/test_determinism.py`

**Test:**
```python
async def test_replay_produces_identical_state():
    """Run live, record events, replay, compare states."""
    # Run live for 100 events
    live_events = []
    live_state = await run_scenario(live_events)
    
    # Replay
    replay_state = await replay_scenario(live_events)
    
    # Compare
    assert live_state.positions == replay_state.positions
    assert live_state.orders == replay_state.orders
    assert live_state.portfolio_value == replay_state.portfolio_value
```

---

### Day 27-28: Final Validation

#### Task 12.1: Production Readiness Checklist

**Validation:**
- [ ] All CRITICAL issues resolved
- [ ] All HIGH issues resolved or mitigated
- [ ] 85%+ test coverage
- [ ] Stress tests pass at 10K events/sec
- [ ] Memory usage stable over 24h
- [ ] Kill switch responds < 100ms
- [ ] Replay produces identical states
- [ ] OMS handles 1000 partial fills correctly
- [ ] All brokers reconcile correctly

---

## DELIVERABLES

### Week 1:
1. Hardened event bus with bounded journal
2. Fire-and-forget eliminated
3. OMS with fill deduplication
4. Broker reconciliation
5. Hardened kill switch

### Week 2:
6. Database transaction boundaries
7. Optimistic locking
8. Repository implementations
9. Kill switch auth
10. Observability improvements

### Week 3:
11. Stress test suite
12. Property-based tests
13. Chaos tests

### Week 4:
14. Integration tests
15. Replay determinism validation
16. Production readiness report

---

## SUCCESS CRITERIA

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Critical Issues | 5 | 0 | 0 |
| High Issues | 6 | 0 | ≤2 |
| Test Coverage | ~40% | 85%+ | 85%+ |
| Event Rate | Untested | 10K/sec | 10K/sec |
| Memory Growth | Unbounded | Bounded | Bounded |
| Kill Switch Latency | Unknown | <100ms | <100ms |
| Replay Determinism | Unproven | Proven | 100% match |

---

## POST-HARDENING ASSESSMENT

After completing this plan, re-run VERIFICATION_AUDIT to confirm:

1. All CRITICAL findings resolved
2. Production readiness score > 85/100
3. Confidence to answer: "Yes, real capital can be trusted"

---

*Hardening plan generated from VERIFICATION_AUDIT findings*
*Each task directly addresses a documented risk*
