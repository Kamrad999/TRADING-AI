"""
Advanced Position Manager following Jesse patterns.
Handles sophisticated position lifecycle, scaling, and risk management.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection
from ..portfolio.position import Position, PositionSide, PositionStatus
from ..execution.execution_engine import ExecutionRequest, ExecutionResult, ScalingMethod
from ..market.market_microstructure import MicrostructureSignals, LiquidityState


class PositionState(Enum):
    """Position states following Jesse lifecycle."""
    PENDING = "pending"           # Awaiting execution
    OPENING = "opening"           # Partially filled, still opening
    OPEN = "open"                 # Fully opened
    SCALING_IN = "scaling_in"     # Adding to position
    SCALING_OUT = "scaling_out"   # Reducing position
    CLOSING = "closing"           # Partially filled, still closing
    CLOSED = "closed"             # Fully closed
    FAILED = "failed"             # Opening failed


class RiskLevel(Enum):
    """Risk levels for position management."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionConfig:
    """Configuration for position management."""
    max_position_size: float = 0.2        # 20% of portfolio
    max_risk_per_trade: float = 0.02     # 2% risk per trade
    stop_loss_pct: float = 0.05           # 5% stop loss
    take_profit_pct: float = 0.10         # 10% take profit
    trailing_stop_pct: float = 0.03       # 3% trailing stop
    scaling_enabled: bool = True
    auto_reduce_enabled: bool = True
    max_positions: int = 10
    correlation_limit: float = 0.7         # Max correlation between positions
    volatility_adjustment: bool = True


@dataclass
class PositionMetrics:
    """Metrics for position analysis."""
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_percentage: float
    max_unrealized: float
    max_drawdown: float
    current_drawdown: float
    time_in_position: float  # Hours
    risk_reward_ratio: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float


@dataclass
class ScalingPlan:
    """Plan for position scaling."""
    action: str  # "add" or "reduce"
    quantity: float
    method: ScalingMethod
    trigger_price: float
    trigger_time: datetime
    reason: str
    confidence: float


class PositionManager:
    """
    Advanced position manager following Jesse patterns.
    
    Key features:
    - Sophisticated position lifecycle management
    - Dynamic position scaling (DCA, Pyramid, Anti-Martingale)
    - Advanced risk management
    - Correlation analysis
    - Performance tracking and optimization
    - Market microstructure integration
    """
    
    def __init__(self, portfolio_value: float = 100000.0, config: Optional[PositionConfig] = None):
        """Initialize position manager."""
        self.logger = get_logger("position_manager")
        
        # Configuration
        self.config = config or PositionConfig()
        self.portfolio_value = portfolio_value
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.position_states: Dict[str, PositionState] = {}
        
        # Scaling management
        self.scaling_plans: Dict[str, ScalingPlan] = {}
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Risk management
        self.risk_metrics: Dict[str, Any] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, PositionMetrics] = {}
        self.position_performance: List[Dict[str, Any]] = []
        
        # Market data integration
        self.market_data: Dict[str, Any] = {}
        
        self.logger.info(f"PositionManager initialized with ${portfolio_value:,.2f} portfolio")
    
    def open_position(self, signal: Signal, execution_result: ExecutionResult,
                     microstructure: Optional[MicrostructureSignals] = None) -> Optional[Position]:
        """
        Open new position from signal and execution result.
        
        Args:
            signal: Trading signal
            execution_result: Execution result
            microstructure: Market microstructure data
            
        Returns:
            Created position or None if failed
        """
        try:
            if execution_result.filled_quantity == 0:
                self.logger.warning(f"Cannot open position: no fill for {signal.symbol}")
                return None
            
            # Check position limits
            if not self._check_position_limits(signal.symbol, execution_result.filled_quantity):
                self.logger.warning(f"Position limits exceeded for {signal.symbol}")
                return None
            
            # Determine position side
            side = PositionSide.LONG if signal.direction == SignalDirection.BUY else PositionSide.SHORT
            
            # Calculate initial stop loss and take profit
            entry_price = execution_result.avg_fill_price
            stop_loss, take_profit = self._calculate_initial_targets(
                side, entry_price, signal, microstructure
            )
            
            # Create position
            position_id = f"pos_{signal.symbol}_{int(datetime.now().timestamp())}"
            
            position = Position(
                id=position_id,
                symbol=signal.symbol,
                side=side,
                status=PositionStatus.OPEN,
                entry_price=entry_price,
                current_price=entry_price,
                quantity=execution_result.filled_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=signal.metadata.get("strategy", "unknown"),
                entry_reason=signal.metadata.get("reason", ""),
                metadata={
                    "signal_id": signal.metadata.get("id"),
                    "execution_id": execution_result.request_id,
                    "signal_confidence": signal.confidence,
                    "execution_slippage": execution_result.slippage,
                    "microstructure": microstructure.metadata if microstructure else {}
                }
            )
            
            # Store position
            self.positions[position_id] = position
            self.position_states[position_id] = PositionState.OPEN
            
            # Initialize metrics
            self._initialize_position_metrics(position_id)
            
            # Create scaling plan if enabled
            if self.config.scaling_enabled:
                self._create_initial_scaling_plan(position, signal, microstructure)
            
            self.logger.info(f"Position opened: {position_id} | {side.value} {signal.symbol} | "
                          f"Size: {execution_result.filled_quantity:.6f} @ ${entry_price:.2f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}")
            return None
    
    def close_position(self, position_id: str, reason: str = "manual", 
                      scaling_method: Optional[ScalingMethod] = None) -> Optional[float]:
        """
        Close position with scaling if needed.
        
        Args:
            position_id: Position ID to close
            reason: Reason for closing
            scaling_method: Scaling method for partial closes
            
        Returns:
            Realized P&L
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                self.logger.warning(f"Position not found: {position_id}")
                return None
            
            if position.status == PositionStatus.CLOSED:
                self.logger.warning(f"Position already closed: {position_id}")
                return position.realized_pnl
            
            # Update position state
            self.position_states[position_id] = PositionState.CLOSING
            
            # Close position
            realized_pnl = position.close(reason)
            
            # Move to history
            self.position_history.append(position)
            del self.positions[position_id]
            del self.position_states[position_id]
            
            # Clean up scaling plans
            if position_id in self.scaling_plans:
                del self.scaling_plans[position_id]
            
            # Update performance metrics
            self._update_position_performance(position_id, position)
            
            self.logger.info(f"Position closed: {position_id} | P&L: ${realized_pnl:.2f} | Reason: {reason}")
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            return None
    
    def scale_position(self, position_id: str, quantity: float, direction: str,
                      scaling_method: ScalingMethod, reason: str = "scaling") -> Optional[Position]:
        """
        Scale existing position (add or reduce).
        
        Args:
            position_id: Position ID to scale
            quantity: Quantity to add/reduce
            direction: "add" or "reduce"
            scaling_method: Scaling method
            reason: Reason for scaling
            
        Returns:
            Updated position
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                self.logger.warning(f"Position not found: {position_id}")
                return None
            
            if direction == "add":
                # Add to position
                new_quantity = position.quantity + quantity
                position.quantity = new_quantity
                self.position_states[position_id] = PositionState.SCALING_IN
                
            elif direction == "reduce":
                # Reduce position
                if quantity >= position.quantity:
                    # Close position completely
                    return self.close_position(position_id, reason)
                else:
                    # Partial close
                    partial_pnl = position.close_partial(quantity, reason)
                    self.position_states[position_id] = PositionState.SCALING_OUT
            
            # Recalculate position value
            position._update_financials()
            
            # Update scaling plan
            self._update_scaling_plan(position_id, direction, quantity, reason)
            
            self.logger.info(f"Position scaled: {position_id} | {direction} {quantity:.6f} | "
                          f"New size: {position.quantity:.6f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Failed to scale position {position_id}: {e}")
            return None
    
    def update_position_prices(self, price_updates: Dict[str, float]) -> None:
        """Update position prices and check for automatic actions."""
        try:
            for position_id, position in self.positions.items():
                if position.symbol in price_updates:
                    old_price = position.current_price
                    new_price = price_updates[position.symbol]
                    
                    # Update price
                    position.update_price(new_price)
                    
                    # Check for automatic actions
                    self._check_automatic_actions(position_id, position)
                    
                    # Update metrics
                    self._update_position_metrics(position_id, position)
            
        except Exception as e:
            self.logger.error(f"Failed to update position prices: {e}")
    
    def _check_automatic_actions(self, position_id: str, position: Position) -> None:
        """Check for automatic position actions."""
        try:
            # Check stop loss
            if position.stop_loss and position.current_price <= position.stop_loss:
                self.logger.info(f"Stop loss triggered for {position_id}")
                self.close_position(position_id, "stop_loss")
                return
            
            # Check take profit
            if position.take_profit and position.current_price >= position.take_profit:
                self.logger.info(f"Take profit triggered for {position_id}")
                self.close_position(position_id, "take_profit")
                return
            
            # Check trailing stop
            if position.trailing_stop:
                self._update_trailing_stop(position_id, position)
            
            # Check scaling triggers
            if self.config.auto_reduce_enabled:
                self._check_scaling_triggers(position_id, position)
            
        except Exception as e:
            self.logger.error(f"Failed to check automatic actions for {position_id}: {e}")
    
    def _update_trailing_stop(self, position_id: str, position: Position) -> None:
        """Update trailing stop based on current price."""
        try:
            if position.side == PositionSide.LONG:
                # For long positions, trailing stop moves up
                new_trailing_stop = position.current_price * (1 - self.config.trailing_stop_pct)
                if position.trailing_stop is None or new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    self.logger.debug(f"Trailing stop updated for {position_id}: ${new_trailing_stop:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update trailing stop for {position_id}: {e}")
    
    def _check_scaling_triggers(self, position_id: str, position: Position) -> None:
        """Check if position should be scaled automatically."""
        try:
            # Scale out on profit
            if position.unrealized_pnl > 0 and position.unrealized_pnl > position.entry_value * 0.05:
                # Take partial profits
                reduce_quantity = position.quantity * 0.25  # Reduce by 25%
                self.scale_position(position_id, reduce_quantity, "reduce", ScalingMethod.IMMEDIATE, "profit_taking")
            
            # Scale in on drawdown
            elif position.unrealized_pnl < 0 and abs(position.unrealized_pnl) > position.entry_value * 0.03:
                # Add to position (DCA)
                add_quantity = position.entry_value / position.current_price * 0.1  # 10% of original value
                self.scale_position(position_id, add_quantity, "add", ScalingMethod.DCA, "drawdown_scaling")
            
        except Exception as e:
            self.logger.error(f"Failed to check scaling triggers for {position_id}: {e}")
    
    def _check_position_limits(self, symbol: str, quantity: float) -> bool:
        """Check if position respects limits."""
        try:
            # Check max position size
            position_value = quantity * self._get_current_price(symbol)
            max_position_value = self.portfolio_value * self.config.max_position_size
            
            if position_value > max_position_value:
                return False
            
            # Check max positions
            current_positions = [p for p in self.positions.values() if p.symbol == symbol]
            if len(current_positions) >= self.config.max_positions:
                return False
            
            # Check correlation
            if not self._check_correlation_limits(symbol):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check position limits: {e}")
            return False
    
    def _check_correlation_limits(self, symbol: str) -> bool:
        """Check correlation limits with existing positions."""
        try:
            # Simplified correlation check
            # In production, would use actual correlation matrix
            
            existing_symbols = set(p.symbol for p in self.positions.values())
            
            # Check for highly correlated pairs
            correlated_pairs = {
                "BTC": ["ETH"],
                "ETH": ["BTC"],
                "SPY": ["QQQ", "DIA"],
                "AAPL": ["MSFT", "GOOGL"]
            }
            
            for existing_symbol in existing_symbols:
                if symbol in correlated_pairs.get(existing_symbol, []):
                    # High correlation detected
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to check correlation limits: {e}")
            return True
    
    def _calculate_initial_targets(self, side: PositionSide, entry_price: float,
                                 signal: Signal, microstructure: Optional[MicrostructureSignals]) -> Tuple[float, float]:
        """Calculate initial stop loss and take profit."""
        try:
            # Base targets from config
            stop_loss_pct = self.config.stop_loss_pct
            take_profit_pct = self.config.take_profit_pct
            
            # Adjust based on signal confidence
            confidence = signal.confidence
            if confidence > 0.8:
                stop_loss_pct *= 0.8  # Tighter stop for high confidence
                take_profit_pct *= 1.2  # Wider target for high confidence
            elif confidence < 0.5:
                stop_loss_pct *= 1.2  # Wider stop for low confidence
                take_profit_pct *= 0.8  # Tighter target for low confidence
            
            # Adjust based on market microstructure
            if microstructure:
                if microstructure.liquidity_state == LiquidityState.VERY_LOW:
                    stop_loss_pct *= 1.5  # Wider stop in illiquid markets
                    take_profit_pct *= 0.8  # Tighter target in illiquid markets
                elif microstructure.liquidity_state == LiquidityState.HIGH:
                    stop_loss_pct *= 0.8  # Tighter stop in liquid markets
                    take_profit_pct *= 1.2  # Wider target in liquid markets
            
            # Calculate targets
            if side == PositionSide.LONG:
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SHORT
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Failed to calculate initial targets: {e}")
            # Return default targets
            if side == PositionSide.LONG:
                return entry_price * 0.95, entry_price * 1.10
            else:
                return entry_price * 1.05, entry_price * 0.90
    
    def _initialize_position_metrics(self, position_id: str) -> None:
        """Initialize metrics for new position."""
        self.performance_metrics[position_id] = PositionMetrics(
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            pnl_percentage=0.0,
            max_unrealized=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            time_in_position=0.0,
            risk_reward_ratio=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0
        )
    
    def _update_position_metrics(self, position_id: str, position: Position) -> None:
        """Update metrics for position."""
        try:
            if position_id not in self.performance_metrics:
                self._initialize_position_metrics(position_id)
            
            metrics = self.performance_metrics[position_id]
            
            # Update P&L metrics
            metrics.unrealized_pnl = position.unrealized_pnl
            metrics.realized_pnl = position.realized_pnl
            metrics.total_pnl = metrics.unrealized_pnl + metrics.realized_pnl
            metrics.pnl_percentage = position.pnl_percentage
            
            # Update max unrealized
            if metrics.unrealized_pnl > metrics.max_unrealized:
                metrics.max_unrealized = metrics.unrealized_pnl
            
            # Update drawdown
            if metrics.max_unrealized > 0:
                current_drawdown = (metrics.max_unrealized - metrics.unrealized_pnl) / metrics.max_unrealized
                metrics.current_drawdown = current_drawdown
                metrics.max_drawdown = max(metrics.max_drawdown, current_drawdown)
            
            # Update time in position
            if position.entry_time:
                metrics.time_in_position = (datetime.now() - position.entry_time).total_seconds() / 3600
            
            # Calculate risk/reward ratio
            if position.stop_loss and position.take_profit:
                if position.side == PositionSide.LONG:
                    risk = abs(position.entry_price - position.stop_loss)
                    reward = abs(position.take_profit - position.entry_price)
                else:
                    risk = abs(position.stop_loss - position.entry_price)
                    reward = abs(position.entry_price - position.take_profit)
                
                metrics.risk_reward_ratio = reward / risk if risk > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to update position metrics for {position_id}: {e}")
    
    def _update_position_performance(self, position_id: str, position: Position) -> None:
        """Update performance metrics when position is closed."""
        try:
            if position_id not in self.performance_metrics:
                return
            
            metrics = self.performance_metrics[position_id]
            
            # Add to performance history
            self.position_performance.append({
                "position_id": position_id,
                "symbol": position.symbol,
                "side": position.side.value,
                "entry_price": position.entry_price,
                "exit_price": position.current_price,
                "realized_pnl": position.realized_pnl,
                "pnl_percentage": position.pnl_percentage,
                "time_in_position": metrics.time_in_position,
                "max_drawdown": metrics.max_drawdown,
                "risk_reward_ratio": metrics.risk_reward_ratio,
                "entry_time": position.entry_time,
                "exit_time": position.exit_time,
                "strategy": position.strategy
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update position performance for {position_id}: {e}")
    
    def _create_initial_scaling_plan(self, position: Position, signal: Signal,
                                   microstructure: Optional[MicrostructureSignals]) -> None:
        """Create initial scaling plan for position."""
        try:
            if not self.config.scaling_enabled:
                return
            
            # Determine scaling strategy based on signal and market conditions
            if signal.confidence > 0.7:
                # High confidence - pyramid scaling
                scaling_method = ScalingMethod.PYRAMID
            elif microstructure and microstructure.market_pressure > 0.5:
                # High market pressure - DCA
                scaling_method = ScalingMethod.DCA
            else:
                # Normal conditions - immediate
                scaling_method = ScalingMethod.IMMEDIATE
            
            plan = ScalingPlan(
                action="add",
                quantity=position.quantity * 0.5,  # Plan to add 50% more
                method=scaling_method,
                trigger_price=position.current_price * 1.02 if position.side == PositionSide.LONG else position.current_price * 0.98,
                trigger_time=datetime.now() + timedelta(hours=1),
                reason="Initial scaling plan",
                confidence=signal.confidence
            )
            
            self.scaling_plans[position.id] = plan
            
        except Exception as e:
            self.logger.error(f"Failed to create scaling plan for {position.id}: {e}")
    
    def _update_scaling_plan(self, position_id: str, direction: str, quantity: float, reason: str) -> None:
        """Update scaling plan after position scaling."""
        try:
            if position_id not in self.scaling_plans:
                return
            
            plan = self.scaling_plans[position_id]
            
            # Record scaling action
            self.scaling_history.append({
                "position_id": position_id,
                "timestamp": datetime.now(),
                "action": direction,
                "quantity": quantity,
                "reason": reason,
                "plan": plan.__dict__
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update scaling plan for {position_id}: {e}")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        try:
            summary = {
                "total_positions": len(self.positions),
                "open_positions": len([p for p in self.positions.values() if p.status == PositionStatus.OPEN]),
                "closed_positions": len(self.position_history),
                "total_unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
                "total_realized_pnl": sum(p.realized_pnl for p in self.position_history),
                "position_states": dict(self.position_states),
                "scaling_plans": len(self.scaling_plans),
                "positions": []
            }
            
            # Add position details
            for position in self.positions.values():
                position_data = {
                    "id": position.id,
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "status": position.status.value,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "pnl_percentage": position.pnl_percentage,
                    "time_in_position": (datetime.now() - position.entry_time).total_seconds() / 3600 if position.entry_time else 0
                }
                
                # Add metrics if available
                if position.id in self.performance_metrics:
                    metrics = self.performance_metrics[position.id]
                    position_data.update({
                        "max_drawdown": metrics.max_drawdown,
                        "risk_reward_ratio": metrics.risk_reward_ratio
                    })
                
                summary["positions"].append(position_data)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get position summary: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        try:
            if not self.position_performance:
                return {"error": "No performance data available"}
            
            # Calculate overall metrics
            total_pnl = sum(p["realized_pnl"] for p in self.position_performance)
            winning_trades = [p for p in self.position_performance if p["realized_pnl"] > 0]
            losing_trades = [p for p in self.position_performance if p["realized_pnl"] < 0]
            
            metrics = {
                "total_trades": len(self.position_performance),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": len(winning_trades) / len(self.position_performance) if self.position_performance else 0,
                "total_pnl": total_pnl,
                "avg_win": np.mean([p["realized_pnl"] for p in winning_trades]) if winning_trades else 0,
                "avg_loss": np.mean([p["realized_pnl"] for p in losing_trades]) if losing_trades else 0,
                "profit_factor": abs(np.mean([p["realized_pnl"] for p in winning_trades]) / np.mean([p["realized_pnl"] for p in losing_trades])) if winning_trades and losing_trades else 0,
                "avg_time_in_position": np.mean([p["time_in_position"] for p in self.position_performance]),
                "avg_max_drawdown": np.mean([p["max_drawdown"] for p in self.position_performance]),
                "avg_risk_reward": np.mean([p["risk_reward_ratio"] for p in self.position_performance if p["risk_reward_ratio"] > 0])
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        # In production, would get from market data provider
        # Simplified for now
        price_map = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "AAPL": 150.0,
            "MSFT": 300.0
        }
        return price_map.get(symbol, 100.0)
