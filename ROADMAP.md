# AMATIS Roadmap

## Current Status: Phase 2 Complete — Canonical Institutional Runtime Foundation

**Phase 2 Status:** ✅ COMPLETE  
**Consolidation Score:** 95/100  
**Verdict:** AMATIS is operating as ONE unified institutional runtime system

---

## Completed Work (Phase 2)

### Runtime Convergence
- ✅ Migrated to canonical `HardenedEventBusV2` (18 files)
- ✅ Migrated to canonical `HardenedOrderManager` (3 files)
- ✅ Removed legacy infrastructure (4 files)
- ✅ Fixed orchestrator to remove `ChildEventBus` references

### Codebase Consolidation
- ✅ Archived 40+ historical markdown files to `docs/archive/`
- ✅ Removed duplicate `trading-ai` subdirectory
- ✅ Cleaned root structure
- ✅ Updated `.gitignore` for institutional-grade security

### Import & Dependency Purification
- ✅ No wildcard imports
- ✅ No circular dependencies
- ✅ Clear dependency direction
- ✅ Bounded module ownership

### Runtime Simplification
- ✅ No unnecessary abstraction layers
- ✅ No fake interfaces
- ✅ No wrapper hell
- ✅ Clear layering and ownership

---

## Remaining Technical Debt

### Type System (Priority: HIGH)
- **Current:** 341 `Any` usages
- **Target:** <50 `Any` usages
- **Action:** Enable mypy strict mode, add Protocol definitions

### Memory Lifecycle (Priority: HIGH)
- **Current:** `MemoryLifecycleManager` not integrated
- **Target:** Fully integrated and active
- **Action:** Integrate into `app.py`, enforce bounded collections

### Observability (Priority: MEDIUM)
- **Current:** Basic logging and metrics
- **Target:** Distributed tracing, metrics dashboard, alerting
- **Action:** Add OpenTelemetry, Prometheus, Grafana

### CI/CD Enforcement (Priority: MEDIUM)
- **Current:** No quality gates
- **Target:** Forbidden import rules, architectural enforcement
- **Action:** Configure pre-commit hooks, CI quality gates

---

## Phase 3: Production Hardening (Future)

### 3.1 Type System Hardening
- Enable mypy strict mode
- Add Protocol definitions for all interfaces
- Reduce `Any` usages to <50
- Add type tests

### 3.2 Memory Lifecycle Integration
- Integrate `MemoryLifecycleManager` into runtime
- Enforce bounded collections
- Add cleanup policies
- Add TTL support

### 3.3 Observability Expansion
- Add distributed tracing (OpenTelemetry)
- Add metrics collection (Prometheus)
- Add metrics dashboard (Grafana)
- Add alerting

### 3.4 CI/CD Enforcement
- Configure pre-commit hooks
- Add forbidden import rules
- Add architectural enforcement
- Add quality gates

### 3.5 Runtime Validation
- 90-day accelerated replay
- Chaos injection tests
- Memory stability validation
- Graceful shutdown validation

---

## Phase 4: Intelligence Systems (Future - NOT STARTED)

**Note:** Phase 4 is explicitly forbidden until Phase 3 is complete.

### 4.1 Multi-Agent Architecture
- Agent framework
- Agent communication
- Agent supervision

### 4.2 Advanced Signal Generation
- ML-based signals
- Ensemble methods
- Adaptive strategies

### 4.3 Reinforcement Learning
- RL-based execution
- RL-based risk management
- RL-based portfolio optimization

---

## Timeline

**Phase 2:** ✅ COMPLETE (2026-05-16)  
**Phase 3:** Q3 2026  
**Phase 4:** TBD (after Phase 3 complete)

---

## Dependencies

**Phase 3 requires:**
- Python 3.11+
- mypy (strict mode)
- OpenTelemetry
- Prometheus
- Grafana
- CI/CD pipeline

**Phase 4 requires:**
- Phase 3 complete
- ML frameworks (PyTorch, TensorFlow)
- RL frameworks (Stable Baselines3, Ray)
- Distributed computing (Ray, Dask)

---

## Governance

**Decision Process:**
- All changes require architectural review
- No new features until Phase 3 complete
- No AI/agent development until Phase 4 approved
- Institutional-grade discipline enforced

**Quality Gates:**
- All tests must pass
- Type checking must pass (strict mode)
- No `Any` usages >50
- No circular dependencies
- No wildcard imports

---

## Contact

**Repository:** https://github.com/Kamrad999/TRADING-AI  
**Issues:** GitHub Issues  
**Discussions:** GitHub Discussions

---

*Last Updated: 2026-05-16*
