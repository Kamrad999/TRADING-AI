# AMATIS Codebase Inventory
## Phase 3.0 — Institutional Consolidation

**Date:** 2026-05-16  
**Total Python Files:** 54  
**Total Markdown Files:** 54+

---

## SECTION 1: File Classification

### Canonical Runtime Infrastructure (ACTIVE)

**Core:**
- `core/event_bus_v2.py` ✅ Canonical event bus
- `core/memory_lifecycle.py` ✅ Canonical lifecycle
- `core/exceptions.py` ✅ Canonical exceptions
- `core/config.py` ✅ Canonical configuration
- `core/observability.py` ✅ Canonical observability
- `core/circuit_breaker.py` ✅ Canonical circuit breaker
- `core/orchestrator.py` ✅ Canonical orchestrator
- `core/event_models.py` ✅ Canonical event models

**Execution:**
- `execution/oms/order_manager_hardened.py` ✅ Canonical OMS
- `execution/oms/state_machine.py` ✅ Canonical state machine

**Risk:**
- `risk/engine.py` ✅ Canonical risk engine
- `risk/models.py` ✅ Canonical risk models

**Signals:**
- `signals/pipeline.py` ✅ Canonical signal pipeline
- `signals/engines/base.py` ✅ Canonical base engine
- `signals/engines/momentum_engine.py` ✅ Canonical momentum
- `signals/engines/news_engine.py` ✅ Canonical news

**Data:**
- `data/market/models.py` ✅ Canonical market models
- `data/market/providers/base.py` ✅ Canonical provider base
- `data/market/providers/alpaca.py` ✅ Canonical Alpaca
- `data/market/providers/yahoo.py` ✅ Canonical Yahoo
- `data/market/cache.py` ✅ Canonical cache
- `data/market/stream_manager.py` ✅ Canonical stream manager
- `data/news/models.py` ✅ Canonical news models
- `data/news/collector.py` ✅ Canonical news collector
- `data/news/deduplicator.py` ✅ Canonical deduplicator
- `data/news/sources.py` ✅ Canonical news sources
- `data/news/validator.py` ✅ Canonical validator

**Replay:**
- `replay/engine.py` ✅ Canonical replay engine

**Simulation:**
- `simulation/replay_engine.py` ✅ Canonical simulation replay
- `simulation/analytics.py` ✅ Canonical analytics
- `simulation/chaos_replay.py` ✅ Canonical chaos replay
- `simulation/market_regimes.py` ✅ Canonical regimes
- `simulation/validation_runner.py` ✅ Canonical validation

**Chaos:**
- `chaos/engine.py` ✅ Canonical chaos engine
- `chaos/injectors.py` ✅ Canonical injectors

**Safety:**
- `safety/kill_switch.py` ✅ Canonical kill switch

**Portfolio:**
- `portfolio/manager.py` ✅ Canonical portfolio manager

**Backtesting:**
- `backtesting/engine.py` ✅ Canonical backtest engine

**Memory:**
- `memory/decision_journal.py` ✅ Canonical decision journal

**Observability:**
- `observability/metrics.py` ✅ Canonical metrics

**Interfaces:**
- `interfaces.py` ✅ Canonical interfaces

**App:**
- `app.py` ✅ Canonical application

---

### Legacy Infrastructure (DEPRECATED - TO REMOVE)

**Core:**
- `core/event_bus.py` ❌ Legacy event bus (18 imports)
- `core/event_bus_hardened.py` ❌ Intermediate event bus (0 imports)
- `core/lifecycle.py` ❌ Legacy lifecycle (0 imports)

**Execution:**
- `execution/oms/order_manager.py` ❌ Legacy OMS (3 imports)

---

### Duplicate Implementations

**Event Buses:**
1. `core/event_bus.py` (legacy)
2. `core/event_bus_hardened.py` (intermediate)
3. `core/event_bus_v2.py` (canonical)
→ **Action:** Remove 1 and 2

**Order Managers:**
1. `execution/oms/order_manager.py` (legacy)
2. `execution/oms/order_manager_hardened.py` (canonical)
→ **Action:** Remove 1

**Lifecycle:**
1. `core/lifecycle.py` (legacy)
2. `core/memory_lifecycle.py` (canonical)
→ **Action:** Remove 1

**Replay Engines:**
1. `replay/engine.py` (canonical for production)
2. `simulation/replay_engine.py` (canonical for simulation)
→ **Action:** Keep both (different purposes)

---

### Documentation Files (54+ MARKDOWN FILES - EXCESSIVE)

**Phase Reports (Historical):**
- `AMATIS_PHASE1.md`
- `AMATIS_PHASE2.md`
- `AMATIS_PHASE2_5.md`
- `AMATIS_HARDENING_PLAN.md`
- `AMATIS_HARDENING_REPORT.md`
- `AMATIS_PRODUCTION_AUDIT.md`
- `AMATIS_SIMULATION_VALIDATION_REPORT.md`
- `AMATIS_STATIC_ANALYSIS_REPORT.md`
- `AMATIS_TECHNICAL_AUDIT.md`
- `AMATIS_VERIFICATION_AUDIT.md`
- `AMATIS_DETERMINISM_PROOF.md`
- `AMATIS_SECURITY_AUDIT.md`
- `SYSTEM_AUDIT_REPORT.md`
- `AUDIT_SUMMARY.md`
- `AGENT_SUMMARY.md`

**Purification Reports (Phase 2.9999):**
- `AMATIS_ARCHITECTURAL_DISCIPLINE.md`
- `AMATIS_ARCHITECTURAL_ENFORCEMENT_V2.md`
- `AMATIS_CODE_QUALITY.md`
- `AMATIS_DOMAIN_MODEL_AUDIT.md`
- `AMATIS_EVENT_BUS_PURIFICATION.md`
- `AMATIS_EXCEPTION_HARDENING.md`
- `AMATIS_FAILURE_DISCIPLINE.md`
- `AMATIS_FAILURE_DISCIPLINE_V2.md`
- `AMATIS_FOUNDATION_LOCKDOWN.md`
- `AMATIS_LONGRUN_VALIDATION.md`
- `AMATIS_MEMORY_LIFECYCLE_REPORT.md`
- `AMATIS_OBSERVABILITY_V2.md`
- `AMATIS_OMS_PURITY_REPORT.md`
- `AMATIS_PURITY_CERTIFICATION.md`
- `AMATIS_TYPE_SYSTEM_PURIFICATION.md`
- `AMATIS_TYPE_SYSTEM_PURIFICATION_V2.md`
- `AMATIS_TYPE_SYSTEM_REPORT.md`

**Architecture Docs:**
- `AMATIS_ARCHITECTURE.md`
- `AMATIS_ARCHITECTURE_FREEZE.md`
- `AMATIS_ARCHITECTURE_FREEZE_AUDIT.md`
- `AMATIS_EVENT_CONTRACTS.md`

**Final Reports:**
- `FINAL_FOUNDATION_CERTIFICATION.md`
- `FINAL_VALIDATION_REPORT.md`
- `PHASE_2_99_FINAL_VERDICT.md`

**Guides:**
- `BACKTEST_ENGINE_GUIDE.md`
- `CRITICAL_PATCHES_GUIDE.md`
- `DEPLOYMENT_READY.md`
- `GITHUB_SETUP.md`

**Runtime Plans:**
- `RUNTIME_CONVERGENCE_PLAN.md`

**Duplicates in trading-ai subdirectory:**
- `trading-ai/AUDIT_SUMMARY.md`
- `trading-ai/CRITICAL_PATCHES_GUIDE.md`
- `trading-ai/DEPLOYMENT_READY.md`
- `trading-ai/FINAL_VALIDATION_REPORT.md`

→ **Action:** Consolidate to single FINAL_REPORT.md, archive historical docs

---

## SECTION 2: Dependency/Runtime Topology

### Critical Subsystems

**Event Bus:**
- Canonical: `core/event_bus_v2.py`
- Current imports: 18 files still using legacy `core/event_bus.py`
- Migration status: 60% complete (app.py, core/__init__.py, execution/__init__.py migrated)

**Order Management:**
- Canonical: `execution/oms/order_manager_hardened.py`
- Current imports: 3 files still using legacy `execution/oms/order_manager.py`
- Migration status: 60% complete (app.py, execution/__init__.py migrated)

**Lifecycle:**
- Canonical: `core/memory_lifecycle.py`
- Current imports: 0 files using legacy `core/lifecycle.py`
- Migration status: 100% (no legacy usage)

---

### Duplicate Systems

**Event Bus Triplication:**
1. `event_bus.py` - Original implementation (18 active imports)
2. `event_bus_hardened.py` - Intermediate hardening (0 imports)
3. `event_bus_v2.py` - Institutional-grade canonical (partial migration)

**Order Manager Duplication:**
1. `order_manager.py` - Original implementation (3 active imports)
2. `order_manager_hardened.py` - Hardened canonical (partial migration)

**Lifecycle Duplication:**
1. `lifecycle.py` - Original implementation (0 imports)
2. `memory_lifecycle.py` - Institutional-grade canonical (ready)

---

### Orphan Modules

**Potential Orphans:**
- `core/event_bus_hardened.py` - 0 imports, intermediate version
- `core/lifecycle.py` - 0 imports, legacy version
- `execution/oms/order_manager.py` - 3 imports, legacy version

---

### Broken Imports

**Potential Issues:**
- `core/orchestrator.py` - References `ChildEventBus` (removed from event_bus_v2)
- `core/orchestrator.py` - Type hint still references `EventBus` instead of `HardenedEventBusV2`

---

## SECTION 3: Action Plan

### Immediate Actions

1. **Complete EventBus Migration:**
   - Update remaining 18 imports from `event_bus.py` to `event_bus_v2.py`
   - Fix `orchestrator.py` to remove `ChildEventBus` references
   - Remove `event_bus.py` and `event_bus_hardened.py`

2. **Complete OrderManager Migration:**
   - Update remaining 3 imports from `order_manager.py` to `order_manager_hardened.py`
   - Remove `order_manager.py`

3. **Remove Legacy Lifecycle:**
   - Remove `lifecycle.py` (already 0 imports)

4. **Consolidate Documentation:**
   - Archive 40+ historical markdown files
   - Keep only: README.md, FINAL_REPORT.md, ARCHITECTURE.md
   - Move historical docs to `docs/archive/`

---

## SECTION 4: Runtime Topology Map

```
app.py (Canonical)
├── HardenedEventBusV2 (Canonical)
│   ├── RiskEngine
│   ├── SignalPipeline
│   │   ├── MomentumEngine
│   │   └── NewsSignalEngine
│   ├── HardenedOrderManager
│   ├── AlpacaDataProvider
│   └── KillSwitch
├── MemoryLifecycleManager (Not yet integrated)
└── Orchestrator (Needs fix)
```

---

## SECTION 5: Classification Summary

| Category | Count | Action |
|----------|-------|--------|
| Canonical Runtime | 40 | Keep |
| Legacy Infrastructure | 4 | Remove |
| Duplicate Implementations | 3 | Remove |
| Documentation Files | 54+ | Consolidate |
| **Total** | **101** | |

---

## SECTION 6: Critical Issues

1. **Dual Runtime:** Event bus and OMS have active legacy implementations
2. **Broken References:** Orchestrator references removed `ChildEventBus`
3. **Documentation Bloat:** 54+ markdown files, mostly historical
4. **Incomplete Migration:** 60% complete on critical subsystems

---

*SECTION 1 — CODEBASE INVENTORY COMPLETE*
