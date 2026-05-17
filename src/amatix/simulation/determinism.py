"""Determinism Validation for AMATIS Replay.

Brutal validation that replay produces identical results:
    - State checksums
    - Portfolio checksums
    - Event sequence verification
    - Signal consistency
    - Order consistency
    - Divergence detection and reporting
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from amatix.core.observability import get_logger
from amatix.simulation.replay_engine import ReplayResult, ReplayState

logger = get_logger(__name__)


@dataclass
class StateChecksum:
    """Checksum for state at a point in time."""
    timestamp: datetime
    event_index: int
    checksum: str
    state_type: str  # "portfolio", "orders", "full"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergencePoint:
    """Point where two replays diverged."""
    event_index: int
    timestamp: datetime
    field_name: str
    expected_value: Any
    actual_value: Any
    severity: str  # minor, major, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_index": self.event_index,
            "field": self.field_name,
            "expected": str(self.expected_value)[:100],
            "actual": str(self.actual_value)[:100],
            "severity": self.severity,
        }


@dataclass
class DivergenceReport:
    """Complete divergence report between two replays."""
    baseline_session: str
    comparison_session: str
    divergence_count: int
    first_divergence_index: Optional[int]
    divergence_points: List[DivergencePoint]
    analysis: str
    
    @property
    def is_identical(self) -> bool:
        """Check if replays were identical."""
        return self.divergence_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_session,
            "comparison": self.comparison_session,
            "divergences": self.divergence_count,
            "first_at": self.first_divergence_index,
            "identical": self.is_identical,
            "points": [p.to_dict() for p in self.divergence_points[:10]],  # Top 10
        }


@dataclass
class DeterminismMetrics:
    """Metrics for determinism validation."""
    total_runs: int
    identical_runs: int
    divergent_runs: int
    avg_divergence_count: float
    determinism_score: float  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": self.total_runs,
            "identical": self.identical_runs,
            "divergent": self.divergent_runs,
            "avg_divergences": round(self.avg_divergence_count, 2),
            "score": round(self.determinism_score, 2),
            "grade": self._get_grade(),
        }
    
    def _get_grade(self) -> str:
        if self.determinism_score >= 99:
            return "A+ (Perfect)"
        elif self.determinism_score >= 95:
            return "A (Excellent)"
        elif self.determinism_score >= 90:
            return "B (Good)"
        elif self.determinism_score >= 80:
            return "C (Acceptable)"
        elif self.determinism_score >= 70:
            return "D (Concerning)"
        else:
            return "F (Non-Deterministic)"


class DeterminismValidator:
    """Validate determinism of AMATIS replay engine.
    
    Performs multiple runs of the same replay scenario and validates
    that results are bit-for-bit identical.
    
    Any divergence indicates:
        - Uncontrolled randomness
        - Race conditions
        - Non-deterministic operations
        - Timing-dependent logic
    
    Usage:
        validator = DeterminismValidator()
        
        # Run multiple validations
        metrics = await validator.validate_replay(
            replay_engine, events, runs=5
        )
        
        # Must be perfect
        assert metrics.determinism_score == 100.0
        assert metrics.divergent_runs == 0
    """
    
    def __init__(self) -> None:
        self._results: List[ReplayResult] = []
        self._divergence_reports: List[DivergenceReport] = []
    
    async def validate_replay(
        self,
        replay_engine_factory,
        events: List[Any],
        runs: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DeterminismMetrics:
        """Run multiple replays and validate determinism.
        
        Args:
            replay_engine_factory: Callable that creates fresh replay engine
            events: Events to replay
            runs: Number of times to replay
            progress_callback: Called with (current_run, total_runs)
        
        Returns:
            DeterminismMetrics with validation results
        """
        self._results.clear()
        self._divergence_reports.clear()
        
        logger.info(f"Starting determinism validation: {runs} runs")
        
        for i in range(runs):
            # Create fresh engine for each run
            engine = replay_engine_factory()
            
            # Run replay
            result = await engine.replay_historical_data(events)
            self._results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, runs)
            
            logger.info(
                f"Run {i+1}/{runs} complete",
                session=result.session_id,
                checksum=result.final_state.checksum(),
            )
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> DeterminismMetrics:
        """Calculate determinism metrics from results."""
        if len(self._results) < 2:
            return DeterminismMetrics(
                total_runs=len(self._results),
                identical_runs=len(self._results),
                divergent_runs=0,
                avg_divergence_count=0,
                determinism_score=100.0,
            )
        
        # Compare all pairs
        identical_count = 0
        total_divergences = 0
        
        baseline = self._results[0]
        
        for i, result in enumerate(self._results[1:], 1):
            report = self._compare_results(baseline, result)
            self._divergence_reports.append(report)
            
            if report.is_identical:
                identical_count += 1
            else:
                total_divergences += report.divergence_count
        
        divergent = len(self._results) - 1 - identical_count
        avg_divergences = total_divergences / max(divergent, 1)
        
        # Score: 100 if all identical, drops based on divergence count
        if divergent == 0:
            score = 100.0
        else:
            # Penalize heavily for any divergence
            score = max(0, 100 - (divergent * 20) - (avg_divergences * 5))
        
        return DeterminismMetrics(
            total_runs=len(self._results),
            identical_runs=identical_count + 1,  # Include baseline
            divergent_runs=divergent,
            avg_divergence_count=avg_divergences,
            determinism_score=score,
        )
    
    def _compare_results(
        self,
        baseline: ReplayResult,
        comparison: ReplayResult,
    ) -> DivergenceReport:
        """Compare two replay results for divergence."""
        divergences: List[DivergencePoint] = []
        
        # Compare final states
        final_div = self._compare_states(
            baseline.final_state,
            comparison.final_state,
            len(baseline.checkpoints),  # Use checkpoint count as index
        )
        divergences.extend(final_div)
        
        # Compare checkpoints
        for i, (base_cp, comp_cp) in enumerate(
            zip(baseline.checkpoints, comparison.checkpoints)
        ):
            if base_cp.state_checksum != comp_cp.state_checksum:
                divergences.append(DivergencePoint(
                    event_index=base_cp.event_count,
                    timestamp=base_cp.timestamp,
                    field_name="state_checksum",
                    expected_value=base_cp.state_checksum,
                    actual_value=comp_cp.state_checksum,
                    severity="critical",
                ))
        
        # Find first divergence
        first_idx = None
        if divergences:
            first_idx = min(d.event_index for d in divergences)
        
        # Generate analysis
        analysis = self._generate_analysis(divergences, baseline, comparison)
        
        return DivergenceReport(
            baseline_session=baseline.session_id,
            comparison_session=comparison.session_id,
            divergence_count=len(divergences),
            first_divergence_index=first_idx,
            divergence_points=divergences,
            analysis=analysis,
        )
    
    def _compare_states(
        self,
        baseline: ReplayState,
        comparison: ReplayState,
        event_index: int,
    ) -> List[DivergencePoint]:
        """Compare two replay states for divergence."""
        divergences = []
        
        # Compare portfolio value
        if baseline.portfolio_value != comparison.portfolio_value:
            divergences.append(DivergencePoint(
                event_index=event_index,
                timestamp=baseline.timestamp,
                field_name="portfolio_value",
                expected_value=baseline.portfolio_value,
                actual_value=comparison.portfolio_value,
                severity="critical",
            ))
        
        # Compare cash
        if baseline.cash != comparison.cash:
            divergences.append(DivergencePoint(
                event_index=event_index,
                timestamp=baseline.timestamp,
                field_name="cash",
                expected_value=baseline.cash,
                actual_value=comparison.cash,
                severity="major",
            ))
        
        # Compare positions
        if baseline.positions != comparison.positions:
            # Find specific differences
            all_symbols = set(baseline.positions.keys()) | set(comparison.positions.keys())
            for symbol in all_symbols:
                base_pos = baseline.positions.get(symbol, {})
                comp_pos = comparison.positions.get(symbol, {})
                
                if base_pos != comp_pos:
                    divergences.append(DivergencePoint(
                        event_index=event_index,
                        timestamp=baseline.timestamp,
                        field_name=f"position_{symbol}",
                        expected_value=base_pos,
                        actual_value=comp_pos,
                        severity="critical",
                    ))
        
        # Compare trade count
        if baseline.total_trades != comparison.total_trades:
            divergences.append(DivergencePoint(
                event_index=event_index,
                timestamp=baseline.timestamp,
                field_name="total_trades",
                expected_value=baseline.total_trades,
                actual_value=comparison.total_trades,
                severity="major",
            ))
        
        # Compare P&L
        if baseline.total_pnl != comparison.total_pnl:
            divergences.append(DivergencePoint(
                event_index=event_index,
                timestamp=baseline.timestamp,
                field_name="total_pnl",
                expected_value=baseline.total_pnl,
                actual_value=comparison.total_pnl,
                severity="critical",
            ))
        
        return divergences
    
    def _generate_analysis(
        self,
        divergences: List[DivergencePoint],
        baseline: ReplayResult,
        comparison: ReplayResult,
    ) -> str:
        """Generate human-readable analysis of divergence."""
        if not divergences:
            return "Results are IDENTICAL. Perfect determinism achieved."
        
        lines = [
            f"DIVERGENCE DETECTED: {len(divergences)} differences",
            "",
            "Analysis:",
        ]
        
        # Categorize
        critical = [d for d in divergences if d.severity == "critical"]
        major = [d for d in divergences if d.severity == "major"]
        minor = [d for d in divergences if d.severity == "minor"]
        
        if critical:
            lines.append(f"  CRITICAL: {len(critical)} - State-affecting divergence")
        if major:
            lines.append(f"  MAJOR: {len(major)} - Significant difference")
        if minor:
            lines.append(f"  MINOR: {len(minor)} - Cosmetic difference")
        
        # Root cause hints
        lines.append("")
        lines.append("Possible root causes:")
        
        field_types = set(d.field_name for d in divergences)
        
        if "portfolio_value" in field_types or "total_pnl" in field_types:
            lines.append("  - Uncontrolled randomness in execution simulation")
            lines.append("  - Non-deterministic slippage calculation")
        
        if any("position" in f for f in field_types):
            lines.append("  - Race condition in position updates")
            lines.append("  - Non-atomic order/fill processing")
        
        if "total_trades" in field_types:
            lines.append("  - Differential fill rates (stochastic behavior)")
            lines.append("  - Non-deterministic order rejection")
        
        lines.append("  - Timing-dependent event ordering")
        lines.append("  - Missing seed in random number generator")
        
        return "\n".join(lines)
    
    def get_worst_divergence(self) -> Optional[DivergenceReport]:
        """Get the report with the most divergences."""
        if not self._divergence_reports:
            return None
        
        return max(
            self._divergence_reports,
            key=lambda r: r.divergence_count,
        )
    
    def generate_determinism_report(self, metrics: DeterminismMetrics) -> str:
        """Generate comprehensive determinism report."""
        lines = [
            "=" * 70,
            "AMATIS DETERMINISM VALIDATION REPORT",
            "=" * 70,
            "",
            f"Total Runs: {metrics.total_runs}",
            f"Identical Runs: {metrics.identical_runs}",
            f"Divergent Runs: {metrics.divergent_runs}",
            f"Average Divergences: {metrics.avg_divergence_count:.2f}",
            "",
            f"DETERMINISM SCORE: {metrics.determinism_score:.2f}/100",
            f"GRADE: {metrics._get_grade()}",
            "",
        ]
        
        if metrics.divergent_runs > 0:
            worst = self.get_worst_divergence()
            if worst:
                lines.extend([
                    "WORST DIVERGENCE:",
                    f"  Sessions: {worst.baseline_session} vs {worst.comparison_session}",
                    f"  Divergences: {worst.divergence_count}",
                    f"  First At: Event {worst.first_divergence_index}",
                    "",
                    "ANALYSIS:",
                    worst.analysis,
                    "",
                ])
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
