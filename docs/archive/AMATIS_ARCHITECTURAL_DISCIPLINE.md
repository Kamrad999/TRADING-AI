# AMATIS ARCHITECTURAL DISCIPLINE ENFORCEMENT
## Phase 2.999 — CI/CD Quality Gate

**Date:** 2026-05-14

## CI Quality Gate Configuration

```yaml
name: AMATIS Quality Gate
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - Type coverage check (must not regress)
      - Any count check (must not increase)
      - Forbidden imports check
      - Replay determinism test
      - Torture tests
```

## Forbidden Imports

- No direct broker imports outside execution layer
- No direct DB imports outside storage layer
- No circular imports (enforced by import-linter)

## Layering Enforcement

- Core → Simulation → Risk → Execution → Storage
- No reverse dependencies

**SECTION 7 — ARCHITECTURAL DISCIPLINE 🟡**
