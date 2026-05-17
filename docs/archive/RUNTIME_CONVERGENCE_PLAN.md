# Runtime Convergence Plan
## Phase 3.0 — Infrastructure Migration

**Date:** 2026-05-16

---

## Legacy Infrastructure Identification

### Event Bus
- **Legacy:** `core/event_bus.py` (18 imports)
- **Intermediate:** `core/event_bus_hardened.py` (0 imports)
- **Canonical:** `core/event_bus_v2.py` (0 imports)

### Order Manager
- **Legacy:** `execution/oms/order_manager.py` (3 imports)
- **Intermediate:** `execution/oms/order_manager_hardened.py` (0 imports)
- **Canonical:** Not yet created

### Lifecycle
- **Legacy:** `core/lifecycle.py` (0 imports)
- **Canonical:** `core/memory_lifecycle.py` (0 imports)

---

## Migration Strategy

### Step 1: Migrate EventBus to EventBusV2
- Update all 18 imports
- Ensure API compatibility
- Test event flow

### Step 2: Migrate OrderManager to OrderManagerHardened
- Update all 3 imports
- Ensure API compatibility
- Test order lifecycle

### Step 3: Integrate MemoryLifecycleManager
- Initialize in app.py
- Register bounded collections
- Add cleanup policies

### Step 4: Remove Legacy Files
- Deprecate event_bus.py
- Deprecate order_manager.py
- Deprecate lifecycle.py

---

## Files to Update

Event Bus (18 files):
1. simulation/validation_runner.py
2. simulation/replay_engine.py
3. simulation/market_regimes.py
4. simulation/chaos_replay.py
5. signals/pipeline.py
6. signals/engines/news_engine.py
7. signals/engines/momentum_engine.py
8. signals/engines/base.py
9. safety/kill_switch.py
10. replay/engine.py
11. risk/engine.py
12. portfolio/manager.py
13. execution/oms/order_manager_hardened.py
14. execution/oms/order_manager.py
15. __init__.py
16. app.py
17. backtesting/engine.py
18. data/news/collector.py
19. data/market/stream_manager.py
20. core/orchestrator.py
21. data/market/providers/base.py
22. core/__init__.py

Order Manager (3 files):
1. execution/__init__.py
2. execution/oms/__init__.py
3. app.py

---

## Execution Order

1. Update EventBus imports (18 files)
2. Update OrderManager imports (3 files)
3. Integrate MemoryLifecycleManager in app.py
4. Test runtime
5. Remove legacy files
