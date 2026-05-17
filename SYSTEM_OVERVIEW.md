# AMATIS System Overview

## What is AMATIS?

AMATIS is an **institutional-grade algorithmic trading system** designed for:
- High-frequency signal generation
- Risk-aware order execution
- Deterministic replay and validation
- Chaos resilience testing
- Long-running production deployment

---

## Architecture

### Canonical Runtime Topology

```
app.py (Canonical Bootstrap)
├── HardenedEventBusV2 (Canonical Event Bus)
│   ├── RiskEngine (Risk Assessment)
│   ├── SignalPipeline (Signal Orchestration)
│   │   ├── MomentumEngine (Technical Analysis)
│   │   └── NewsSignalEngine (NLP Analysis)
│   ├── HardenedOrderManager (Canonical OMS)
│   ├── AlpacaDataProvider (Market Data)
│   └── KillSwitch (Emergency Stop)
├── MemoryLifecycleManager (Ready for Integration)
└── Orchestrator (System Coordination)
```

### Layering

**Core Layer:**
- Event bus (HardenedEventBusV2)
- Configuration management
- Observability
- Circuit breakers
- Memory lifecycle

**Data Layer:**
- Market data providers (Alpaca, Yahoo)
- Data normalization
- Data caching
- Stream management
- News collection

**Signals Layer:**
- Signal pipeline
- Signal engines (momentum, news)
- Signal filtering
- Signal aggregation

**Risk Layer:**
- Risk engine
- Risk rules
- Risk assessment
- Risk adjustment

**Execution Layer:**
- Order manager (HardenedOrderManager)
- Order state machine
- Order reconciliation
- Fill validation

**Portfolio Layer:**
- Portfolio manager
- Position tracking
- P&L calculation
- Exposure management

**Simulation Layer:**
- Replay engine
- Accelerated replay
- Chaos replay
- Analytics
- Validation runner

---

## Key Features

### Event-Driven Architecture
- Decoupled components
- Event bus with delivery guarantees
- Dead letter queue
- Deterministic sequencing
- Event journaling

### Risk Management
- Pre-trade risk assessment
- Position limits
- Exposure limits
- Drawdown protection
- Emergency kill switch

### Order Management
- Order lifecycle management
- State machine validation
- Fill deduplication
- Broker reconciliation
- Orphan order detection

### Replay & Validation
- Deterministic replay
- Accelerated replay (up to 1000x)
- Chaos injection
- Market regime simulation
- Validation metrics

### Observability
- Structured logging
- Metrics collection
- Performance tracking
- Error tracking
- Health monitoring

---

## Data Flow

```
Market Data → Signal Pipeline → Risk Engine → Order Manager → Broker
                ↓                    ↓              ↓
            Event Bus ←─────── Kill Switch ←───────
```

1. **Market Data:** Providers emit price/quote events
2. **Signal Pipeline:** Engines analyze data, emit signals
3. **Risk Engine:** Assesses risk, approves/rejects orders
4. **Order Manager:** Manages order lifecycle, tracks fills
5. **Broker:** Executes orders, reports fills
6. **Kill Switch:** Emergency stop, overrides all

---

## Configuration

### Environment Variables

```bash
AMATIS_MODE=production|simulation|backtest
AMATIS_LOG_LEVEL=INFO|DEBUG|ERROR
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
```

### Configuration File

See `env.template` for full configuration options.

---

## Deployment

### Production Mode
```bash
export AMATIS_MODE=production
python -m amatix.app
```

### Simulation Mode
```bash
export AMATIS_MODE=simulation
python -m amatix.app
```

### Backtest Mode
```bash
export AMATIS_MODE=backtest
python -m amatix.app
```

---

## Monitoring

### Health Checks
- Component health status
- Event bus metrics
- Order manager metrics
- Risk engine metrics
- Data provider connectivity

### Metrics
- Event counts
- Handler errors
- Queue sizes
- Dead letter queue size
- Latency metrics
- Throughput metrics

### Logging
- Structured JSON logs
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Contextual logging with trace IDs
- Error logging with stack traces

---

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Replay Validation
```bash
python -m amatix.simulation.validation_runner
```

### Chaos Testing
```bash
python -m amatix.simulation.chaos_replay
```

---

## Security

### API Keys
- Never commit API keys
- Use environment variables
- Use `.env` file (gitignored)
- See `env.template` for reference

### Kill Switch
- Multi-signature support
- Token-based authentication
- Emergency stop capability
- Audit logging

### Risk Limits
- Position limits enforced
- Exposure limits enforced
- Drawdown protection
- Circuit breakers

---

## Performance

### Latency
- Signal generation: <10ms
- Risk assessment: <5ms
- Order submission: <50ms
- Event processing: <1ms

### Throughput
- Events per second: 10,000+
- Orders per second: 100+
- Signals per second: 1,000+

### Memory
- Bounded collections
- TTL-based cleanup
- Memory pressure detection
- Leak detection

---

## Troubleshooting

### Common Issues

**Event bus not starting:**
- Check configuration
- Check dependencies
- Check logs

**Orders not executing:**
- Check risk engine
- Check kill switch
- Check broker connectivity

**Replay not deterministic:**
- Check event journal
- Check state checksums
- Check event ordering

---

## Documentation

- [Architecture](AMATIS_ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [Archived Reports](docs/archive/)

---

## License

See LICENSE file.

---

*Last Updated: 2026-05-16*
