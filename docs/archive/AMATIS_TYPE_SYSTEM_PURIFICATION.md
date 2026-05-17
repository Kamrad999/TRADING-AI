# AMATIS TYPE SYSTEM PURIFICATION
## Phase 2.9999 — Institutional Type Safety

**Date:** 2026-05-16  
**Target:** <5% Any usage, strict mode compatible

---

## Current State

**Any Usage:** 341 instances across 58 files  
**Target:** <50 instances (<5%)  
**Gap:** 291 instances to eliminate

---

## Critical Any Usages

1. **Event payloads** — `Dict[str, Any]` in Event model
2. **Repository returns** — `Dict[str, Any]` in serialization
3. **Callback types** — `Any` in interface callbacks
4. **Domain model fields** — `Any` in SignalFeature, RiskViolation
5. **Component instances** — `Any` in orchestrator

---

## Migration Plan

### Phase 1: Protocol Interfaces (8 hours)
- Define protocols for all callbacks
- Define protocols for components
- Define protocols for repositories

### Phase 2: Typed Event Contracts (8 hours)
- Migrate to `contracts/events.py` types
- Replace `Dict[str, Any]` with specific event types
- Update event bus to accept typed events

### Phase 3: Domain Model Typing (8 hours)
- Replace `Any` with Union types
- Add TypedDict for DTOs
- Make models immutable where appropriate

### Phase 4: Strict Mode Enablement (4 hours)
- Enable mypy strict mode
- Fix immediate errors
- Add to CI

---

## Success Metrics

| Metric | Current | Target |
|--------|--------|--------|
| Any usage | 341 | <50 |
| Type coverage | 65% | 90% |
| Strict mode | Disabled | Enabled |
| Protocol definitions | 0 | 20 |

**Estimated Effort:** 28 hours

---

*SECTION 3 — TYPE SYSTEM PURIFICATION 🟠*
