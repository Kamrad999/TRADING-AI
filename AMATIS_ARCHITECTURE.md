# AMATIS: Adaptive Multi-Agent Trading Intelligence System

## Genesis Foundation — Phase 0

AMATIS is an institutional-grade, event-driven, modular trading intelligence platform engineered for evolution into autonomous multi-agent systems.

> **⚠️ IMPORTANT**: This is Phase 0 — The Foundation. We are building the operating system, not the trading strategies. No AI, no RL, no prediction systems yet.

---

## 🎯 Engineering Principles

| Principle | Description |
|-------------|-------------|
| **Reliability > Features** | A reliable foundation beats broken features |
| **Modularity > Speed** | Clean interfaces enable future evolution |
| **Explainability > Magic** | Every decision must be auditable |
| **Composability > Monoliths** | Components plug together cleanly |
| **Event-Driven > Direct Coupling** | Decoupled architecture enables scaling |
| **Risk-First** | Risk system has final veto authority |

---

## 🏗️ Architecture

```
AMATIS/
├── src/amatix/
│   ├── core/              # Foundation layer
│   │   ├── event_bus.py        # Event-driven backbone
│   │   ├── event_models.py     # Event dataclasses
│   │   ├── orchestrator.py   # System conductor
│   │   ├── circuit_breaker.py # Resilience patterns
│   │   ├── observability.py   # Logging, metrics, tracing
│   │   └── config.py          # Settings management
│   │
│   ├── interfaces.py      # ABCs for all components
│   │
│   ├── memory/            # Decision journaling
│   │   ├── decision_journal.py
│   │   └── feature_attribution.py
│   │
│   └── [future modules]   # data/, signals/, execution/, etc.
│
├── tests/                 # Comprehensive test suite
├── infra/                 # Docker, K8s, infrastructure
├── docs/                  # Documentation
└── pyproject.toml         # Modern Python packaging
```

---

## 📦 Core Components

### Event Bus (`amatix.core.event_bus`)

The central nervous system enabling decoupled component communication.

```python
from amatix.core import EventBus, EventType

bus = EventBus()

@bus.on(EventType.SIGNAL_GENERATED)
async def handle_signal(event):
    print(f"Signal: {event.payload}")

await bus.emit_new(
    EventType.SIGNAL_GENERATED,
    {"symbol": "AAPL", "direction": "LONG"},
    source="my_strategy"
)
```

**Features:**
- Async-first with sync handler support
- Priority-based processing (CRITICAL events first)
- Middleware chain for transformation/filtering
- Event replay for debugging
- Journaling for audit trails

### Orchestrator (`amatix.core.orchestrator`)

System lifecycle manager with graceful degradation.

```python
from amatix.core import Orchestrator

orch = Orchestrator()
orch.register("risk", risk_engine, critical=True)
orch.register("execution", exec_engine)

await orch.start()  # Initializes all components
await orch.stop()   # Graceful shutdown
```

### Interfaces (`amatix.interfaces`)

Clean ABCs defining component contracts:

- `DataProvider` — Market data sources
- `SignalEngine` — Signal generation
- `RiskEngine` — Risk management (with veto authority)
- `ExecutionEngine` — Order execution
- `Strategy` — Trading strategies
- `PortfolioManager` — Position sizing
- `Agent` — Future multi-agent support

### Decision Journal (`amatix.memory`)

Explainability and audit foundation:

```python
from amatix.memory import DecisionJournal, DecisionRecord, DecisionType

journal = DecisionJournal()

record = DecisionRecord.create(
    decision_type=DecisionType.ORDER_SUBMITTED,
    trace_id=event.context.trace_id,
    signal=signal,
    order=order,
)

await journal.record(record)
```

### Observability (`amatix.core.observability`)

Structured logging, metrics, and health checks:

```python
from amatix.core.observability import get_logger, get_metrics, timed

logger = get_logger(__name__)
logger.info("signal_generated", symbol="AAPL", confidence=0.85)

metrics = get_metrics()
metrics.counter("signals_generated", labels={"strategy": "momentum"})

@timed("execution_latency")
async def execute_order(order):
    pass
```

### Circuit Breaker (`amatix.core.circuit_breaker`)

Resilience pattern for external service calls:

```python
from amatix.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(
    "alpaca",
    CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60)
)

@breaker
async def call_api():
    return await alpaca.get_positions()
```

---

## 🔧 Development Setup

### Prerequisites
- Python 3.11+
- PostgreSQL (for production)
- Redis (for caching)

### Installation

```bash
# Clone and enter directory
git clone <repo>
cd amatis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Copy environment template
cp env.template .env
# Edit .env with your configuration
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=amatix --cov-report=html

# Unit tests only (fast)
pytest tests/unit/

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Linting
ruff check .
ruff format .

# Type checking
mypy src/amatix

# Security checks
bandit -r src/
safety scan
```

---

## 📋 Phase 0 Completion Checklist

### ✅ TASK 1 — Repository Hard Reset
- [x] Removed duplicate `trading-ai/` folder
- [x] Cleaned obsolete artifacts
- [x] Preserved valuable domain logic (risk concepts, signal ideas)

### ✅ TASK 2 — Modern Python Foundation
- [x] `pyproject.toml` with modern packaging
- [x] Ruff configuration (linting + formatting)
- [x] MyPy strict configuration
- [x] Pytest with coverage
- [x] Pre-commit hooks
- [x] `.gitignore` for AMATIS
- [x] Environment template

### ✅ TASK 3 — Event-Driven Core
- [x] Event dataclasses with typing
- [x] Event bus with priority queues
- [x] Middleware support
- [x] Event replay capability
- [x] Async/sync handler support

### ✅ TASK 4 — Core Interfaces
- [x] `DataProvider` ABC
- [x] `SignalEngine` ABC
- [x] `RiskEngine` ABC (with veto authority)
- [x] `ExecutionEngine` ABC
- [x] `Strategy` ABC
- [x] `PortfolioManager` ABC
- [x] `Agent` ABC (future)

### ✅ TASK 5 — Observability Foundation
- [x] Structured logging (JSON)
- [x] Metrics collection (Prometheus-compatible)
- [x] Tracing preparation
- [x] Health check registry
- [x] Performance decorators

### ✅ TASK 6 — Decision Journal System
- [x] Decision record dataclass
- [x] Feature attribution tracking
- [x] Context snapshots
- [x] Rationale recording
- [x] Outcome tracking
- [x] RL training data export

---

## 🚀 Next Phases (Future Work)

### Phase 1: Data Infrastructure
- Real-time market data connectors
- Historical data storage (PostgreSQL + TimescaleDB)
- WebSocket feeds
- Data validation and cleaning

### Phase 2: Signal Engines
- News signal engine (refactored from legacy)
- Price-based signal engines
- Technical indicators
- Signal composition framework

### Phase 3: Risk & Execution
- Risk engine implementation (7-layer system)
- Broker connectors (Alpaca, IBKR, Binance)
- Order management system
- Position tracking

### Phase 4: ML Foundation
- Feature engineering pipeline
- Model serving infrastructure
- Sentiment analysis (FinBERT)
- RL environment preparation

### Phase 5: Multi-Agent System
- Agent orchestration
- Meta-learning framework
- Strategy reputation system
- Alpha decay detection

---

## 📚 Documentation

- `AMATIS_ARCHITECTURE.md` — This file
- `docs/architecture.md` — Detailed architecture docs
- `docs/api.md` — API reference
- `docs/deployment.md` — Production deployment guide

---

## 🛡️ Security

- All secrets via environment variables
- No hardcoded credentials
- Secrets scanning in CI/CD
- Audit logging for all decisions

---

## 📄 License

MIT License — See `LICENSE` file

---

## 🤝 Contributing

This is a foundational rewrite. Contributions should:
1. Maintain interface contracts
2. Include comprehensive tests
3. Follow the engineering principles above
4. Document architectural decisions

---

**Built for institutional-grade reliability. Engineered for evolution.**
