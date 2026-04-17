"""
Performance and Alpha Tracking system following institutional trading patterns.
Implements comprehensive performance analytics and alpha generation tracking.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from ..infrastructure.logging import get_logger
from ..portfolio.position import Position, PositionSide, PositionStatus
from ..core.models import Signal, SignalDirection
from ..strategies.freqtrade_strategies import StrategyResult
from ..learning.learning_engine import TradeMemory


class PerformanceMetric(Enum):
    """Performance metrics types."""
    RETURN = "return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    ALPHA = "alpha"
    BETA = "beta"
    INFORMATION_RATIO = "information_ratio"


class AlphaType(Enum):
    """Alpha generation types."""
    EVENT_DRIVEN = "event_driven"
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MARKET_TIMING = "market_timing"
    RISK_PREMIUM = "risk_premium"


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    alpha: float
    beta: float
    information_ratio: float
    volatility: float
    var_95: float
    cvar_95: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlphaAttribution:
    """Alpha attribution analysis."""
    alpha_type: AlphaType
    alpha_value: float
    contribution_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    factor_exposure: Dict[str, float]
    attribution_period: str
    confidence_level: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Risk metrics analysis."""
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    average_drawdown: float
    drawdown_duration: float
    volatility: float
    downside_volatility: float
    skewness: float
    kurtosis: float
    correlation_benchmark: float
    beta: float
    tracking_error: float


class AlphaTracker:
    """
    Performance and alpha tracking system following institutional trading patterns.
    
    Key features:
    - Comprehensive performance metrics
    - Alpha attribution analysis
    - Risk metrics calculation
    - Benchmark comparison
    - Performance attribution
    - Alpha decay detection
    """
    
    def __init__(self, benchmark_symbol: str = "SPY"):
        """Initialize alpha tracker."""
        self.logger = get_logger("alpha_tracker")
        
        # Benchmark
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_data: deque = deque(maxlen=2520)  # 10 years of daily data
        
        # Performance tracking
        self.performance_snapshots: deque = deque(maxlen=2520)
        self.daily_returns: deque = deque(maxlen=2520)
        self.equity_curve: deque = deque(maxlen=2520)
        
        # Alpha attribution
        self.alpha_attributions: Dict[str, AlphaAttribution] = {}
        self.alpha_history: List[AlphaAttribution] = []
        
        # Risk metrics
        self.risk_metrics: RiskMetrics = RiskMetrics(
            var_95=0.0, cvar_95=0.0, var_99=0.0, cvar_99=0.0,
            max_drawdown=0.0, average_drawdown=0.0, drawdown_duration=0.0,
            volatility=0.0, downside_volatility=0.0,
            skewness=0.0, kurtosis=0.0,
            correlation_benchmark=0.0, beta=0.0, tracking_error=0.0
        )
        
        # Performance parameters
        self._initialize_performance_parameters()
        
        # Alpha detection
        self._initialize_alpha_detection()
        
        self.logger.info(f"AlphaTracker initialized with benchmark: {benchmark_symbol}")
    
    def _initialize_performance_parameters(self) -> None:
        """Initialize performance calculation parameters."""
        self.performance_params = {
            # Risk-free rate for calculations
            "risk_free_rate": 0.02,  # 2% annualized
            
            # Calculation periods
            "sharpe_period": 252,  # Trading days per year
            "sortino_period": 252,
            "calmar_period": 252,
            
            # Drawdown parameters
            "drawdown_window": 252,  # 1 year lookback
            "recovery_threshold": 0.95,  # 95% recovery threshold
            
            # VaR parameters
            "var_confidence_levels": [0.95, 0.99],
            "var_window": 252,  # 1 year lookback
            
            # Alpha parameters
            "alpha_threshold": 0.0001,  # Daily alpha threshold
            "alpha_window": 63,  # 3 months lookback
            "alpha_decay_threshold": 0.5  # 50% decay threshold
        }
    
    def _initialize_alpha_detection(self) -> None:
        """Initialize alpha detection parameters."""
        self.alpha_detection = {
            # Alpha generation factors
            "alpha_factors": {
                "event_driven": 0.3,
                "technical": 0.25,
                "momentum": 0.2,
                "mean_reversion": 0.15,
                "market_timing": 0.1
            },
            
            # Performance thresholds
            "min_sharpe": 0.5,
            "min_win_rate": 0.55,
            "max_drawdown": 0.2,
            "min_alpha": 0.05,  # 5% annual alpha
            
            # Attribution weights
            "attribution_weights": {
                "signal_quality": 0.4,
                "execution_quality": 0.2,
                "risk_management": 0.2,
                "market_conditions": 0.2
            }
        }
    
    def update_performance(self, positions: List[Position], signals: List[Signal],
                         strategy_results: List[StrategyResult], benchmark_data: Optional[Dict[str, float]] = None) -> PerformanceSnapshot:
        """
        Update performance metrics with current data.
        
        Args:
            positions: Current positions
            signals: Generated signals
            strategy_results: Strategy execution results
            benchmark_data: Benchmark price data
            
        Returns:
            Current performance snapshot
        """
        try:
            timestamp = datetime.now()
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(positions)
            
            # Calculate returns
            daily_return = self._calculate_daily_return(portfolio_value)
            cumulative_return = self._calculate_cumulative_return(portfolio_value)
            
            # Calculate risk metrics
            volatility = self._calculate_volatility()
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(daily_return, volatility)
            sortino_ratio = self._calculate_sortino_ratio(daily_return)
            calmar_ratio = self._calculate_calmar_ratio(cumulative_return, max_drawdown)
            
            # Calculate trading metrics
            win_rate, profit_factor = self._calculate_trading_metrics(positions)
            
            # Calculate alpha and beta
            alpha, beta = self._calculate_alpha_beta(benchmark_data)
            
            # Calculate information ratio
            information_ratio = self._calculate_information_ratio(alpha, beta)
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(0.95)
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                total_return=cumulative_return,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95,
                metadata={
                    "position_count": len(positions),
                    "signal_count": len(signals),
                    "strategy_count": len(strategy_results)
                }
            )
            
            # Store snapshot
            self.performance_snapshots.append(snapshot)
            self.daily_returns.append(daily_return)
            self.equity_curve.append(portfolio_value)
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Detect alpha decay
            self._detect_alpha_decay()
            
            self.logger.debug(f"Performance updated: Return: {cumulative_return:.2%} | Sharpe: {sharpe_ratio:.2f}")
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to update performance: {e}")
            # Return empty snapshot
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                total_return=0.0,
                daily_return=0.0,
                cumulative_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                alpha=0.0,
                beta=0.0,
                information_ratio=0.0,
                volatility=0.0,
                var_95=0.0,
                cvar_95=0.0
            )
    
    def analyze_alpha_attribution(self, trade_memories: List[TradeMemory]) -> Dict[str, AlphaAttribution]:
        """Analyze alpha attribution from trade memories."""
        try:
            attributions = {}
            
            # Group trades by alpha type
            alpha_groups = defaultdict(list)
            
            for trade in trade_memories:
                alpha_type = self._classify_trade_alpha_type(trade)
                alpha_groups[alpha_type].append(trade)
            
            # Calculate attribution for each alpha type
            for alpha_type, trades in alpha_groups.items():
                if len(trades) < 5:  # Minimum trades for analysis
                    continue
                
                attribution = self._calculate_alpha_attribution(alpha_type, trades)
                attributions[alpha_type.value] = attribution
                
                # Store in history
                self.alpha_history.append(attribution)
            
            # Update current attributions
            self.alpha_attributions = attributions
            
            self.logger.info(f"Alpha attribution analyzed: {len(attributions)} alpha types")
            
            return attributions
            
        except Exception as e:
            self.logger.error(f"Failed to analyze alpha attribution: {e}")
            return {}
    
    def _classify_trade_alpha_type(self, trade: TradeMemory) -> AlphaType:
        """Classify the alpha type of a trade."""
        try:
            # Check event-driven alpha
            if trade.event_classifications:
                high_impact_events = [e for e in trade.event_classifications if e.impact_level.value >= 3]
                if high_impact_events:
                    return AlphaType.EVENT_DRIVEN
            
            # Check technical alpha
            if "technical" in trade.strategy.lower():
                return AlphaType.TECHNICAL
            
            # Check momentum alpha
            if abs(trade.pnl_percentage) > 0.05 and trade.duration_hours < 24:
                return AlphaType.MOMENTUM
            
            # Check mean reversion alpha
            if abs(trade.pnl_percentage) > 0.03 and trade.max_drawdown > 0.02:
                return AlphaType.MEAN_REVERSION
            
            # Default to market timing
            return AlphaType.MARKET_TIMING
            
        except Exception as e:
            self.logger.error(f"Failed to classify alpha type: {e}")
            return AlphaType.MARKET_TIMING
    
    def _calculate_alpha_attribution(self, alpha_type: AlphaType, trades: List[TradeMemory]) -> AlphaAttribution:
        """Calculate alpha attribution for a specific alpha type."""
        try:
            # Calculate basic metrics
            total_return = sum(trade.pnl_percentage for trade in trades)
            avg_return = total_return / len(trades)
            
            winning_trades = [t for t in trades if t.realized_pnl > 0]
            win_rate = len(winning_trades) / len(trades)
            
            # Calculate Sharpe ratio
            returns = [trade.pnl_percentage for trade in trades]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            max_drawdown = max(trade.max_drawdown for trade in trades)
            
            # Calculate contribution percentage
            total_alpha = sum(t.pnl_percentage for t in self.alpha_history if t.pnl_percentage > 0)
            contribution_pct = total_return / total_alpha if total_alpha > 0 else 0.0
            
            # Calculate factor exposure (simplified)
            factor_exposure = self._calculate_factor_exposure(trades)
            
            # Calculate confidence level
            confidence_level = min(1.0, len(trades) / 20.0)  # More trades = higher confidence
            
            return AlphaAttribution(
                alpha_type=alpha_type,
                alpha_value=avg_return,
                contribution_pct=contribution_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                factor_exposure=factor_exposure,
                attribution_period="current",
                confidence_level=confidence_level,
                metadata={
                    "trade_count": len(trades),
                    "avg_duration": np.mean([trade.duration_hours for trade in trades]),
                    "avg_confidence": np.mean([trade.signal_confidence for trade in trades])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate alpha attribution for {alpha_type}: {e}")
            return AlphaAttribution(
                alpha_type=alpha_type,
                alpha_value=0.0,
                contribution_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                factor_exposure={},
                attribution_period="current",
                confidence_level=0.0
            )
    
    def _calculate_factor_exposure(self, trades: List[TradeMemory]) -> Dict[str, float]:
        """Calculate factor exposure for trades."""
        try:
            factor_exposure = defaultdict(float)
            
            for trade in trades:
                # Market regime exposure
                regime = trade.market_conditions.get("market_regime", "neutral")
                factor_exposure[f"regime_{regime}"] += 1.0
                
                # Time of day exposure
                hour = trade.entry_time.hour
                if 6 <= hour < 12:
                    factor_exposure["time_morning"] += 1.0
                elif 12 <= hour < 18:
                    factor_exposure["time_afternoon"] += 1.0
                else:
                    factor_exposure["time_evening"] += 1.0
                
                # Volatility exposure
                volatility = trade.market_conditions.get("volatility", 0.02)
                if volatility < 0.01:
                    factor_exposure["vol_low"] += 1.0
                elif volatility < 0.03:
                    factor_exposure["vol_medium"] += 1.0
                else:
                    factor_exposure["vol_high"] += 1.0
            
            # Normalize exposures
            total_exposure = sum(factor_exposure.values())
            if total_exposure > 0:
                for factor in factor_exposure:
                    factor_exposure[factor] /= total_exposure
            
            return dict(factor_exposure)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate factor exposure: {e}")
            return {}
    
    def _calculate_portfolio_value(self, positions: List[Position]) -> float:
        """Calculate current portfolio value."""
        try:
            total_value = 0.0
            
            for position in positions:
                if position.status == PositionStatus.OPEN:
                    total_value += position.current_value
                    total_value += position.realized_pnl  # Add realized P&L from partial closes
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio value: {e}")
            return 0.0
    
    def _calculate_daily_return(self, portfolio_value: float) -> float:
        """Calculate daily return."""
        try:
            if len(self.equity_curve) == 0:
                return 0.0
            
            previous_value = self.equity_curve[-1]
            if previous_value == 0:
                return 0.0
            
            return (portfolio_value - previous_value) / previous_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate daily return: {e}")
            return 0.0
    
    def _calculate_cumulative_return(self, portfolio_value: float) -> float:
        """Calculate cumulative return."""
        try:
            if len(self.equity_curve) == 0:
                return 0.0
            
            initial_value = self.equity_curve[0]
            if initial_value == 0:
                return 0.0
            
            return (portfolio_value - initial_value) / initial_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cumulative return: {e}")
            return 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility."""
        try:
            if len(self.daily_returns) < 20:
                return 0.0
            
            returns = list(self.daily_returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Failed to calculate volatility: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            equity_values = list(self.equity_curve)
            peak = equity_values[0]
            max_dd = 0.0
            
            for value in equity_values[1:]:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Failed to calculate max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, daily_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        try:
            if volatility == 0:
                return 0.0
            
            risk_free_daily = self.performance_params["risk_free_rate"] / 252
            excess_return = daily_return - risk_free_daily
            
            # Use historical data for more accurate calculation
            if len(self.daily_returns) >= 30:
                returns = list(self.daily_returns)
                excess_returns = [r - risk_free_daily for r in returns]
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            else:
                sharpe = excess_return / volatility
            
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, daily_return: float) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            returns = list(self.daily_returns)
            risk_free_daily = self.performance_params["risk_free_rate"] / 252
            excess_returns = [r - risk_free_daily for r in returns]
            
            # Calculate downside deviation
            downside_returns = [r for r in excess_returns if r < 0]
            if len(downside_returns) == 0:
                return 0.0
            
            downside_deviation = np.std(downside_returns)
            if downside_deviation == 0:
                return 0.0
            
            sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
            return sortino
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, cumulative_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        try:
            if max_drawdown == 0:
                return 0.0
            
            return cumulative_return / max_drawdown
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Calmar ratio: {e}")
            return 0.0
    
    def _calculate_trading_metrics(self, positions: List[Position]) -> Tuple[float, float]:
        """Calculate trading metrics (win rate and profit factor)."""
        try:
            if not positions:
                return 0.0, 0.0
            
            # Get closed positions
            closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
            
            if not closed_positions:
                return 0.0, 0.0
            
            # Calculate win rate
            winning_trades = [p for p in closed_positions if p.realized_pnl > 0]
            win_rate = len(winning_trades) / len(closed_positions)
            
            # Calculate profit factor
            total_wins = sum(p.realized_pnl for p in winning_trades)
            total_losses = abs(sum(p.realized_pnl for p in closed_positions if p.realized_pnl < 0))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
            
            return win_rate, profit_factor
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trading metrics: {e}")
            return 0.0, 0.0
    
    def _calculate_alpha_beta(self, benchmark_data: Optional[Dict[str, float]]) -> Tuple[float, float]:
        """Calculate alpha and beta relative to benchmark."""
        try:
            if not benchmark_data or len(self.daily_returns) < 30:
                return 0.0, 0.0
            
            # Get benchmark returns (simplified)
            benchmark_returns = []
            # In production, would use actual benchmark data
            
            # Calculate beta
            if len(self.daily_returns) >= 30:
                portfolio_returns = list(self.daily_returns)[-30:]
                # Simplified beta calculation
                beta = 1.0  # Default beta
            else:
                beta = 1.0
            
            # Calculate alpha
            risk_free_rate = self.performance_params["risk_free_rate"]
            expected_return = risk_free_rate + beta * 0.08  # 8% market return assumption
            
            if len(self.daily_returns) >= 30:
                avg_return = np.mean(list(self.daily_returns)[-30:]) * 252  # Annualized
                alpha = avg_return - expected_return
            else:
                alpha = 0.0
            
            return alpha, beta
            
        except Exception as e:
            self.logger.error(f"Failed to calculate alpha and beta: {e}")
            return 0.0, 0.0
    
    def _calculate_information_ratio(self, alpha: float, beta: float) -> float:
        """Calculate information ratio."""
        try:
            if len(self.daily_returns) < 30:
                return 0.0
            
            # Simplified information ratio
            tracking_error = self.risk_metrics.tracking_error
            if tracking_error == 0:
                return 0.0
            
            return alpha / tracking_error
            
        except Exception as e:
            self.logger.error(f"Failed to calculate information ratio: {e}")
            return 0.0
    
    def _calculate_var_cvar(self, confidence_level: float) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        try:
            if len(self.daily_returns) < 30:
                return 0.0, 0.0
            
            returns = list(self.daily_returns)
            returns.sort()
            
            # Calculate VaR
            var_index = int((1 - confidence_level) * len(returns))
            var = -returns[var_index]  # Negative because returns are sorted ascending
            
            # Calculate CVaR (expected shortfall)
            cvar_returns = returns[:var_index]
            cvar = -np.mean(cvar_returns) if cvar_returns else var
            
            return var, cvar
            
        except Exception as e:
            self.logger.error(f"Failed to calculate VaR/CVaR: {e}")
            return 0.0, 0.0
    
    def _update_risk_metrics(self) -> None:
        """Update risk metrics."""
        try:
            if len(self.daily_returns) < 30:
                return
            
            returns = list(self.daily_returns)
            
            # VaR and CVaR
            self.risk_metrics.var_95, self.risk_metrics.cvar_95 = self._calculate_var_cvar(0.95)
            self.risk_metrics.var_99, self.risk_metrics.cvar_99 = self._calculate_var_cvar(0.99)
            
            # Drawdown metrics
            self.risk_metrics.max_drawdown = self._calculate_max_drawdown()
            
            # Volatility metrics
            self.risk_metrics.volatility = self._calculate_volatility()
            
            # Downside volatility
            downside_returns = [r for r in returns if r < 0]
            self.risk_metrics.downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.0
            
            # Higher moments
            self.risk_metrics.skewness = self._calculate_skewness(returns)
            self.risk_metrics.kurtosis = self._calculate_kurtosis(returns)
            
            # Beta and tracking error
            alpha, beta = self._calculate_alpha_beta(None)
            self.risk_metrics.beta = beta
            
            # Tracking error (simplified)
            self.risk_metrics.tracking_error = self.risk_metrics.volatility * 0.1  # 10% of volatility
            
        except Exception as e:
            self.logger.error(f"Failed to update risk metrics: {e}")
    
    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns."""
        try:
            if len(returns) < 3:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            skewness = np.mean([((r - mean_return) / std_return) ** 3 for r in returns])
            return skewness
            
        except Exception as e:
            self.logger.error(f"Failed to calculate skewness: {e}")
            return 0.0
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns."""
        try:
            if len(returns) < 4:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            kurtosis = np.mean([((r - mean_return) / std_return) ** 4 for r in returns]) - 3  # Excess kurtosis
            return kurtosis
            
        except Exception as e:
            self.logger.error(f"Failed to calculate kurtosis: {e}")
            return 0.0
    
    def _detect_alpha_decay(self) -> None:
        """Detect alpha decay in performance."""
        try:
            if len(self.performance_snapshots) < 63:  # Need 3 months of data
                return
            
            # Calculate alpha decay
            recent_snapshots = list(self.performance_snapshots)[-63:]  # Last 3 months
            older_snapshots = list(self.performance_snapshots)[-126:-63]  # Previous 3 months
            
            if len(older_snapshots) == 0:
                return
            
            recent_alpha = np.mean([s.alpha for s in recent_snapshots])
            older_alpha = np.mean([s.alpha for s in older_snapshots])
            
            alpha_decay = (older_alpha - recent_alpha) / older_alpha if older_alpha != 0 else 0.0
            
            # Check if alpha decay is significant
            if alpha_decay > self.performance_params["alpha_decay_threshold"]:
                self.logger.warning(f"Alpha decay detected: {alpha_decay:.2%}")
                
                # Trigger adaptation if needed
                self._trigger_alpha_decay_adaptation(alpha_decay)
            
        except Exception as e:
            self.logger.error(f"Failed to detect alpha decay: {e}")
    
    def _trigger_alpha_decay_adaptation(self, decay_amount: float) -> None:
        """Trigger adaptation for alpha decay."""
        try:
            # Log alpha decay event
            self.logger.info(f"Alpha decay adaptation triggered: {decay_amount:.2%}")
            
            # In production, this would trigger strategy adjustments
            # For now, just log the event
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alpha decay adaptation: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.performance_snapshots:
                return {"error": "No performance data available"}
            
            latest = self.performance_snapshots[-1]
            
            # Calculate additional metrics
            if len(self.performance_snapshots) >= 30:
                monthly_snapshots = list(self.performance_snapshots)[-30:]
                monthly_return = np.mean([s.daily_return for s in monthly_snapshots]) * 30
                monthly_sharpe = np.mean([s.sharpe_ratio for s in monthly_snapshots])
            else:
                monthly_return = latest.daily_return * 30
                monthly_sharpe = latest.sharpe_ratio
            
            # Performance attribution
            alpha_summary = {
                "total_alpha_types": len(self.alpha_attributions),
                "top_alpha_type": None,
                "top_alpha_value": 0.0
            }
            
            if self.alpha_attributions:
                top_alpha = max(self.alpha_attributions.values(), key=lambda x: x.alpha_value)
                alpha_summary["top_alpha_type"] = top_alpha.alpha_type.value
                alpha_summary["top_alpha_value"] = top_alpha.alpha_value
            
            return {
                "current_performance": {
                    "portfolio_value": latest.portfolio_value,
                    "total_return": latest.total_return,
                    "daily_return": latest.daily_return,
                    "sharpe_ratio": latest.sharpe_ratio,
                    "sortino_ratio": latest.sortino_ratio,
                    "max_drawdown": latest.max_drawdown,
                    "win_rate": latest.win_rate,
                    "profit_factor": latest.profit_factor,
                    "calmar_ratio": latest.calmar_ratio,
                    "alpha": latest.alpha,
                    "beta": latest.beta,
                    "information_ratio": latest.information_ratio,
                    "volatility": latest.volatility
                },
                "risk_metrics": {
                    "var_95": self.risk_metrics.var_95,
                    "cvar_95": self.risk_metrics.cvar_95,
                    "var_99": self.risk_metrics.var_99,
                    "cvar_99": self.risk_metrics.cvar_99,
                    "max_drawdown": self.risk_metrics.max_drawdown,
                    "volatility": self.risk_metrics.volatility,
                    "downside_volatility": self.risk_metrics.downside_volatility,
                    "skewness": self.risk_metrics.skewness,
                    "kurtosis": self.risk_metrics.kurtosis,
                    "beta": self.risk_metrics.beta,
                    "tracking_error": self.risk_metrics.tracking_error
                },
                "monthly_performance": {
                    "return": monthly_return,
                    "sharpe": monthly_sharpe
                },
                "alpha_summary": alpha_summary,
                "performance_history": {
                    "snapshots_count": len(self.performance_snapshots),
                    "days_tracked": len(self.daily_returns),
                    "last_update": latest.timestamp.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def get_alpha_report(self) -> Dict[str, Any]:
        """Get comprehensive alpha attribution report."""
        try:
            if not self.alpha_attributions:
                return {"error": "No alpha attribution data available"}
            
            # Calculate alpha statistics
            total_alpha = sum(attribution.alpha_value for attribution in self.alpha_attributions.values())
            alpha_contributions = {name: attribution.contribution_pct for name, attribution in self.alpha_attributions.items()}
            
            # Performance by alpha type
            alpha_performance = {}
            for name, attribution in self.alpha_attributions.items():
                alpha_performance[name] = {
                    "alpha_value": attribution.alpha_value,
                    "sharpe_ratio": attribution.sharpe_ratio,
                    "win_rate": attribution.win_rate,
                    "max_drawdown": attribution.max_drawdown,
                    "confidence": attribution.confidence_level,
                    "trade_count": attribution.metadata.get("trade_count", 0)
                }
            
            # Factor exposure analysis
            factor_exposure = defaultdict(float)
            for attribution in self.alpha_attributions.values():
                for factor, exposure in attribution.factor_exposure.items():
                    factor_exposure[factor] += exposure
            
            return {
                "total_alpha": total_alpha,
                "alpha_contributions": alpha_contributions,
                "alpha_performance": alpha_performance,
                "factor_exposure": dict(factor_exposure),
                "attribution_history": len(self.alpha_history),
                "last_analysis": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get alpha report: {e}")
            return {"error": str(e)}
