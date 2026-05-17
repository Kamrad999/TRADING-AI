# AMATIS ARCHITECTURAL ENFORCEMENT
## Phase 2.9999 — CI Quality Gate

**Date:** 2026-05-16

---

## Requirements

Enforce THE CONSTITUTION through CI.

Detect:
- Forbidden imports
- Circular dependencies
- Unsafe async patterns
- Mutable shared state
- Forbidden Any usage
- Architecture violations
- Event contract violations

---

## CI Configuration

```yaml
name: AMATIS Quality Gate V2
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - Type coverage check (must not regress)
      - Any count check (must not increase)
      - Forbidden imports check
      - Circular dependency check
      - Architecture purity check
      - Replay determinism test
      - Torture tests
```

---

## Forbidden Imports

- No direct broker imports outside execution layer
- No direct DB imports outside storage layer
- No circular imports (enforced by import-linter)

---

## Layering Enforcement

- Core → Simulation → Risk → Execution → Storage
- No reverse dependencies

---

## Status

**CI configuration:** ⏳ Pending  
**Forbidden import rules:** ⏳ Pending  
**Layering enforcement:** ⏳ Pending

**Estimated Effort:** 12 hours

---

*SECTION 7 — ARCHITECTURAL ENFORCEMENT 🟡*
