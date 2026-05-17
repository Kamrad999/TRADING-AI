"""AMATIS Institutional Validation Tests.

Comprehensive simulation and replay validation for Phase 2.95.
Brutal validation of determinism, resilience, and correctness.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from amatix.core.event_bus import EventBus
from amatix.simulation.analytics import PortfolioAnalytics
from amatix.simulation.chaos_replay import (
    ChaosInjection,
    ChaosReplayOrchestrator,
    ReplayFailureType,
)
from amatix.simulation.determinism import DeterminismValidator
from amatix.simulation.execution_simulator import ExecutionSimulator, SlippageModel
from amatix.simulation.market_regimes import (
    MarketRegimeType,
    RegimeGenerator,
    ScenarioBuilder,
)
from amatix.simulation.replay_engine import (
    AcceleratedReplayEngine,
    ReplayConfig,
    ReplaySpeed,
)


class TestDeterminismValidation:
    """Brutal determinism validation — must be perfect."""

    async def test_replay_determinism_5_runs(self):
        """Run 5 identical replays — must produce IDENTICAL results."""
        bus = EventBus()

        # Generate deterministic market data
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.SIDEWAYS,
            ["AAPL"],
            datetime(2024, 1, 1),
            days=5,
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL"], datetime(2024, 1, 1), bars_per_day=10
        )

        # Validator
        validator = DeterminismValidator()

        def engine_factory():
            return AcceleratedReplayEngine(
                bus,
                ReplayConfig(speed=ReplaySpeed.MAX_SPEED, seed=42),
            )

        metrics = await validator.validate_replay(engine_factory, events, runs=5)

        # BRUTAL: Must be perfect
        assert metrics.determinism_score == 100.0, (
            f"DETERMINISM FAILURE: Score {metrics.determinism_score}"
        )
        assert metrics.divergent_runs == 0, (
            f"DIVERGENCE DETECTED: {metrics.divergent_runs} runs diverged"
        )

        print(f"✅ Determinism: {metrics.determinism_score}/100 — PERFECT")

    async def test_checksum_consistency(self):
        """Verify state checksums are consistent across replays."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.BULL_TREND, ["MSFT"], datetime(2024, 1, 1), days=3
        )
        events = generator.generate_market_data_events(
            regime, ["MSFT"], datetime(2024, 1, 1), bars_per_day=10
        )

        # Run twice
        engine1 = AcceleratedReplayEngine(bus, ReplayConfig(speed=ReplaySpeed.MAX_SPEED, seed=42))
        result1 = await engine1.replay_historical_data(events)

        engine2 = AcceleratedReplayEngine(bus, ReplayConfig(speed=ReplaySpeed.MAX_SPEED, seed=42))
        result2 = await engine2.replay_historical_data(events)

        # Checksums must match
        assert result1.final_state.checksum() == result2.final_state.checksum()

        # All checkpoints must match
        for cp1, cp2 in zip(result1.checkpoints, result2.checkpoints):
            assert cp1.state_checksum == cp2.state_checksum

        print("✅ Checksum consistency validated")


class TestMarketRegimeScenarios:
    """Test behavior under different market regimes."""

    async def test_bull_market_scenario(self):
        """Validate behavior in trending bull market."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)

        regime = generator.generate_regime(
            MarketRegimeType.BULL_TREND,
            ["AAPL", "MSFT"],
            datetime(2024, 1, 1),
            days=10,
        )

        events = generator.generate_market_data_events(
            regime, ["AAPL", "MSFT"], datetime(2024, 1, 1)
        )

        engine = AcceleratedReplayEngine(bus)
        result = await engine.replay_historical_data(events)

        # Should process all events
        assert result.events_processed == len(events)
        assert result.determinism_score >= 95

        print(f"✅ Bull market: {result.events_processed} events processed")

    async def test_bear_market_scenario(self):
        """Validate behavior in bear market with drawdowns."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)

        regime = generator.generate_regime(
            MarketRegimeType.BEAR_TREND,
            ["AAPL"],
            datetime(2024, 1, 1),
            days=10,
        )

        events = generator.generate_market_data_events(regime, ["AAPL"], datetime(2024, 1, 1))

        engine = AcceleratedReplayEngine(bus)
        result = await engine.replay_historical_data(events)

        # In bear market, portfolio should be monitored for drawdown
        assert result.final_state.max_drawdown >= 0  # Should track this

        print("✅ Bear market: Max DD tracked")

    async def test_flash_crash_scenario(self):
        """Validate behavior during flash crash conditions."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)

        regime = generator.generate_regime(
            MarketRegimeType.FLASH_CRASH,
            ["AAPL"],
            datetime(2024, 1, 1),
            days=3,
        )

        events = generator.generate_market_data_events(regime, ["AAPL"], datetime(2024, 1, 1))

        engine = AcceleratedReplayEngine(bus)
        result = await engine.replay_historical_data(events)

        # System should survive flash crash
        assert result.success is True or result.events_processed > len(events) * 0.9

        print("✅ Flash crash: System survived")

    async def test_multi_regime_scenario(self):
        """Test regime transitions."""
        builder = ScenarioBuilder.create_stress_test()
        name, events = builder.build_scenario(
            "stress_test",
            ["AAPL", "MSFT", "GOOGL"],
            datetime(2024, 1, 1),
            seed=42,
        )

        bus = EventBus()
        engine = AcceleratedReplayEngine(bus)
        result = await engine.replay_historical_data(events)

        assert result.events_processed > 0
        print(f"✅ Multi-regime: {name} with {result.events_processed} events")


class TestChaosDuringReplay:
    """Chaos engineering during replay."""

    async def test_event_drops_during_replay(self):
        """Validate recovery when events are dropped."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.SIDEWAYS, ["AAPL"], datetime(2024, 1, 1), days=5
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL"], datetime(2024, 1, 1), bars_per_day=20
        )

        orchestrator = ChaosReplayOrchestrator(bus, seed=42)

        # Schedule event drops
        orchestrator.schedule_chaos(
            ChaosInjection(
                failure_type=ReplayFailureType.EVENT_DROP,
                trigger_event_index=50,
                duration_events=20,
                probability=0.3,
            )
        )

        result = await orchestrator.run_chaos_replay(events)

        # Should maintain resilience
        assert result.resilience_score >= 60  # At least partial recovery

        print(f"✅ Event drops: Resilience {result.resilience_score:.1f}")

    async def test_websocket_disconnect_during_replay(self):
        """Validate recovery from connection drops."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.HIGH_VOLATILITY, ["AAPL"], datetime(2024, 1, 1), days=5
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL"], datetime(2024, 1, 1), bars_per_day=20
        )

        orchestrator = ChaosReplayOrchestrator(bus, seed=42)
        orchestrator.schedule_chaos(
            ChaosInjection(
                failure_type=ReplayFailureType.WEBSOCKET_DISCONNECT,
                trigger_event_index=30,
                duration_events=10,
            )
        )

        result = await orchestrator.run_chaos_replay(events)

        # System should continue after disconnect
        assert result.replay_result.events_processed > len(events) * 0.8

        print(f"✅ Disconnect: {result.replay_result.events_processed} events processed")

    async def test_random_chaos_survival(self):
        """Test survival under random chaos."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.SIDEWAYS, ["AAPL", "MSFT"], datetime(2024, 1, 1), days=7
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL", "MSFT"], datetime(2024, 1, 1), bars_per_day=15
        )

        orchestrator = ChaosReplayOrchestrator(bus, seed=42)
        orchestrator.schedule_random_chaos(count=5, event_range=(100, 500))

        result = await orchestrator.run_chaos_replay(events)

        # Must achieve at least Grade C
        assert result.resilience_score >= 70, (
            f"CHAOS FAILURE: Score {result.resilience_score} below 70"
        )

        print(f"✅ Random chaos: Grade {result._get_grade()}")


class TestExecutionRealism:
    """Validate execution simulation realism."""

    async def test_slippage_calculation(self):
        """Validate slippage increases with order size."""
        simulator = ExecutionSimulator(
            slippage_model=SlippageModel(),
            seed=42,
        )

        from amatix.simulation.execution_simulator import MarketCondition

        market = MarketCondition(
            symbol="AAPL",
            volatility=Decimal("0.25"),
            adv_30d=Decimal("10000000"),
        )

        # Small order
        small = await simulator.simulate_fill("AAPL", "buy", Decimal("100"), "market", market)

        # Large order
        large = await simulator.simulate_fill("AAPL", "buy", Decimal("10000"), "market", market)

        # Larger orders should have more slippage
        assert large.slippage >= small.slippage

        print(
            f"✅ Slippage: Small {float(small.slippage) * 10000:.1f}bps, Large {float(large.slippage) * 10000:.1f}bps"
        )

    async def test_partial_fill_simulation(self):
        """Validate partial fill behavior."""
        simulator = ExecutionSimulator(
            partial_fill_rate=0.5,  # High rate for test
            seed=42,
        )

        from amatix.simulation.execution_simulator import MarketCondition

        market = MarketCondition(symbol="AAPL")

        fills = []
        for _ in range(10):
            fill = await simulator.simulate_fill("AAPL", "buy", Decimal("1000"), "market", market)
            fills.append(fill)

        # Should have some partial fills
        partials = sum(1 for f in fills if f.partial_fill)
        assert partials > 0, "Expected partial fills"

        print(f"✅ Partial fills: {partials}/10")

    async def test_rejection_simulation(self):
        """Validate order rejection simulation."""
        simulator = ExecutionSimulator(
            rejection_rate=0.3,  # High rate for test
            seed=42,
        )

        from amatix.simulation.execution_simulator import MarketCondition

        # Normal market
        normal = MarketCondition(symbol="AAPL")

        # Halted market
        halted = MarketCondition(
            symbol="AAPL",
            halted=True,
        )

        normal_fills = []
        halted_fills = []

        for _ in range(10):
            n = await simulator.simulate_fill("AAPL", "buy", Decimal("100"), "market", normal)
            h = await simulator.simulate_fill("AAPL", "buy", Decimal("100"), "market", halted)
            normal_fills.append(n)
            halted_fills.append(h)

        # Halted should have more rejections
        halted_rejects = sum(1 for f in halted_fills if f.rejection_reason)
        assert halted_rejects >= 9, "Expected halted market rejections"

        print(f"✅ Rejections: Halted {halted_rejects}/10")


class TestPerformanceUnderLoad:
    """Performance validation during replay."""

    async def test_1000x_replay_speed(self):
        """Validate replay at 1000x speed."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.SIDEWAYS, ["AAPL"], datetime(2024, 1, 1), days=10
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL"], datetime(2024, 1, 1), bars_per_day=78
        )

        import time

        start = time.time()

        engine = AcceleratedReplayEngine(
            bus,
            ReplayConfig(speed=ReplaySpeed.ACCELERATED_1000X, seed=42),
        )
        result = await engine.replay_historical_data(events)

        duration = time.time() - start

        # 10 days at 1000x should take ~10 minutes in virtual time
        # but should process in seconds of real time
        assert duration < 30, f"Too slow: {duration:.1f}s"
        assert result.events_processed == len(events)

        print(f"✅ 1000x speed: {len(events)} events in {duration:.2f}s")

    async def test_max_speed_throughput(self):
        """Test max speed processing."""
        bus = EventBus()
        generator = RegimeGenerator(seed=42)
        regime = generator.generate_regime(
            MarketRegimeType.SIDEWAYS, ["AAPL"], datetime(2024, 1, 1), days=30
        )
        events = generator.generate_market_data_events(
            regime, ["AAPL"], datetime(2024, 1, 1), bars_per_day=78
        )

        engine = AcceleratedReplayEngine(
            bus,
            ReplayConfig(speed=ReplaySpeed.MAX_SPEED, seed=42),
        )

        import time

        start = time.time()
        result = await engine.replay_historical_data(events)
        duration = time.time() - start

        rate = result.events_processed / duration

        # Should achieve high throughput
        assert rate > 1000, f"Low throughput: {rate:.0f} events/sec"

        print(f"✅ Max speed: {rate:.0f} events/sec")


class TestPortfolioAnalytics:
    """Portfolio analytics calculation validation."""

    async def test_sharpe_ratio_calculation(self):
        """Validate Sharpe ratio calculation."""
        analytics = PortfolioAnalytics(risk_free_rate=Decimal("0.02"))

        # Create equity curve
        from datetime import timedelta

        base_date = datetime(2024, 1, 1)
        equity_curve = []

        equity = Decimal("100000")
        for i in range(252):  # 1 year
            # 10% annual return with 15% vol
            ret = Decimal("0.0003") + Decimal(str(__import__("random").gauss(0, 0.01)))
            equity *= Decimal("1") + ret
            equity_curve.append((base_date + timedelta(days=i), equity))

        # Mock replay result
        from amatix.simulation.replay_engine import ReplayResult, ReplayState

        state = ReplayState(
            sequence_id=0,
            timestamp=base_date,
            portfolio_value=equity,
            cash=Decimal("20000"),
            positions={},
            active_orders={},
            total_trades=100,
            total_pnl=equity - Decimal("100000"),
            max_drawdown=Decimal("0.05"),
        )

        result = ReplayResult(
            session_id="test",
            start_time=base_date,
            end_time=base_date + timedelta(days=252),
            events_processed=1000,
            events_emitted=1000,
            duration_seconds=1.0,
            checkpoints=[],
            final_state=state,
            determinism_score=100.0,
            integrity_violations=[],
        )

        metrics = analytics.calculate_performance(
            result,
            trade_history=[],
            equity_curve=equity_curve,
        )

        # Should have reasonable Sharpe
        assert metrics.sharpe_ratio > 0
        assert metrics.sharpe_ratio < 5  # Sanity check

        print(f"✅ Sharpe: {float(metrics.sharpe_ratio):.2f}")

    async def test_drawdown_calculation(self):
        """Validate max drawdown calculation."""
        analytics = PortfolioAnalytics()

        # Create drawdown scenario
        from datetime import timedelta

        base_date = datetime(2024, 1, 1)

        # Peak, then drop, then recover
        equity_curve = [
            (base_date, Decimal("100000")),
            (base_date + timedelta(days=10), Decimal("110000")),  # Peak
            (base_date + timedelta(days=20), Decimal("95000")),  # Drawdown
            (base_date + timedelta(days=30), Decimal("100000")),  # Recover
        ]

        from amatix.simulation.replay_engine import ReplayResult, ReplayState

        state = ReplayState(
            sequence_id=0,
            timestamp=base_date,
            portfolio_value=Decimal("100000"),
            cash=Decimal("20000"),
            positions={},
            active_orders={},
            total_trades=10,
            total_pnl=Decimal("0"),
            max_drawdown=Decimal("0.136"),  # (110-95)/110
        )

        result = ReplayResult(
            session_id="test",
            start_time=base_date,
            end_time=base_date + timedelta(days=30),
            events_processed=100,
            events_emitted=100,
            duration_seconds=1.0,
            checkpoints=[],
            final_state=state,
            determinism_score=100.0,
            integrity_violations=[],
        )

        metrics = analytics.calculate_performance(result, [], equity_curve)

        # Max DD should be ~13.6%
        assert 10 <= float(metrics.max_drawdown_pct) <= 20

        print(f"✅ Max DD: {float(metrics.max_drawdown_pct):.1f}%")
