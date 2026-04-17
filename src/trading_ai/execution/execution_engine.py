"""
Advanced Execution Engine following Jesse patterns.
Implements sophisticated order execution with timing, scaling, and slippage handling.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import time

from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection
from ..market.market_microstructure import MicrostructureSignals, LiquidityState
from ..portfolio.position import Position, PositionSide, PositionStatus


class ExecutionType(Enum):
    """Execution types following institutional trading."""
    MARKET = "market"
    LIMIT = "limit"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    PARTICIPATE = "participate"


class ScalingMethod(Enum):
    """Position scaling methods."""
    IMMEDIATE = "immediate"
    DCA = "dca"  # Dollar Cost Averaging
    PYRAMID = "pyramid"  # Pyramid scaling
    ANTI_MARTINGALE = "anti_martingale"
    VOLATILITY_BASED = "volatility_based"


class FillType(Enum):
    """Fill types."""
    FULL = "full"
    PARTIAL = "partial"
    SLIPPAGE = "slippage"
    FAILED = "failed"


@dataclass
class ExecutionRequest:
    """Execution request for advanced order handling."""
    symbol: str
    direction: SignalDirection
    quantity: float
    order_type: ExecutionType
    price: Optional[float] = None
    scaling_method: ScalingMethod = ScalingMethod.IMMEDIATE
    time_limit: Optional[int] = None  # Seconds
    slippage_tolerance: float = 0.001  # 0.1%
    min_fill_size: float = 0.0
    max_fill_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    request_id: str
    symbol: str
    direction: SignalDirection
    requested_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_cost: float
    fees: float
    slippage: float
    execution_time: float
    fill_type: FillType
    partial_fills: List[Dict[str, Any]]
    status: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingAnalysis:
    """Timing analysis for optimal execution."""
    optimal_entry_time: datetime
    market_pressure: float
    liquidity_score: float
    volatility_adjustment: float
    timing_confidence: float
    reasoning: str


class ExecutionEngine:
    """
    Advanced execution engine following Jesse patterns.
    
    Key features:
    - Sophisticated order execution algorithms
    - Entry timing optimization
    - Position scaling (DCA, Pyramid, etc.)
    - Slippage handling and prediction
    - Partial fill management
    - Market microstructure integration
    """
    
    def __init__(self, exchange_interface=None):
        """Initialize execution engine."""
        self.logger = get_logger("execution_engine")
        
        # Exchange interface
        self.exchange = exchange_interface
        
        # Execution algorithms
        self._initialize_execution_algorithms()
        
        # Timing models
        self._initialize_timing_models()
        
        # Scaling strategies
        self._initialize_scaling_strategies()
        
        # Slippage models
        self._initialize_slippage_models()
        
        # Execution tracking
        self.active_orders: Dict[str, ExecutionRequest] = {}
        self.execution_history: List[ExecutionResult] = []
        self.timing_cache: Dict[str, TimingAnalysis] = {}
        
        # Performance metrics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "avg_slippage": 0.0,
            "avg_execution_time": 0.0,
            "fill_rate": 0.0,
            "algorithm_performance": defaultdict(lambda: {"count": 0, "slippage": []})
        }
        
        self.logger.info("ExecutionEngine initialized with Jesse-style execution algorithms")
    
    def _initialize_execution_algorithms(self) -> None:
        """Initialize execution algorithms."""
        self.execution_algorithms = {
            ExecutionType.MARKET: self._execute_market_order,
            ExecutionType.LIMIT: self._execute_limit_order,
            ExecutionType.ICEBERG: self._execute_iceberg_order,
            ExecutionType.TWAP: self._execute_twap_order,
            ExecutionType.VWAP: self._execute_vwap_order,
            ExecutionType.PARTICIPATE: self._execute_participate_order
        }
    
    def _initialize_timing_models(self) -> None:
        """Initialize timing optimization models."""
        self.timing_models = {
            "market_pressure_weight": 0.4,
            "liquidity_weight": 0.3,
            "volatility_weight": 0.2,
            "time_decay_weight": 0.1,
            
            "optimal_time_windows": {
                "high_volatility": 300,  # 5 minutes
                "normal_volatility": 600,  # 10 minutes
                "low_volatility": 1800,  # 30 minutes
            },
            
            "pressure_thresholds": {
                "extreme": 0.8,
                "high": 0.6,
                "normal": 0.4,
                "low": 0.2
            }
        }
    
    def _initialize_scaling_strategies(self) -> None:
        """Initialize position scaling strategies."""
        self.scaling_strategies = {
            ScalingMethod.IMMEDIATE: self._scale_immediate,
            ScalingMethod.DCA: self._scale_dca,
            ScalingMethod.PYRAMID: self._scale_pyramid,
            ScalingMethod.ANTI_MARTINGALE: self._scale_anti_martingale,
            ScalingMethod.VOLATILITY_BASED: self._scale_volatility_based
        }
    
    def _initialize_slippage_models(self) -> None:
        """Initialize slippage prediction models."""
        self.slippage_models = {
            "base_slippage": {
                "high_liquidity": 0.0001,  # 0.01%
                "medium_liquidity": 0.0005,  # 0.05%
                "low_liquidity": 0.002,     # 0.2%
                "very_low_liquidity": 0.01   # 1.0%
            },
            
            "volume_impact": {
                "small": 0.0001,    # < 0.1% of daily volume
                "medium": 0.0005,   # 0.1-1% of daily volume
                "large": 0.002,     # 1-5% of daily volume
                "very_large": 0.01  # > 5% of daily volume
            },
            
            "timing_adjustment": {
                "optimal": -0.0002,  # Better timing reduces slippage
                "normal": 0.0,
                "poor": 0.0005       # Poor timing increases slippage
            }
        }
    
    def execute_order(self, request: ExecutionRequest, 
                     microstructure: Optional[MicrostructureSignals] = None) -> ExecutionResult:
        """
        Execute order with advanced algorithms.
        
        Args:
            request: Execution request
            microstructure: Market microstructure data
            
        Returns:
            Execution result
        """
        try:
            request_id = f"exec_{int(time.time())}_{request.symbol}"
            
            # Optimize timing
            timing = self._optimize_execution_timing(request, microstructure)
            
            # Apply scaling strategy
            scaled_requests = self._apply_scaling_strategy(request)
            
            # Execute scaled orders
            results = []
            for scaled_request in scaled_requests:
                result = self._execute_single_order(scaled_request, microstructure, timing)
                results.append(result)
            
            # Aggregate results
            final_result = self._aggregate_execution_results(request_id, request, results)
            
            # Store execution
            self.execution_history.append(final_result)
            
            # Update stats
            self._update_execution_stats(final_result)
            
            self.logger.info(f"Order executed: {request.symbol} {request.direction.value} | "
                          f"Filled: {final_result.filled_quantity}/{final_result.requested_quantity} | "
                          f"Slippage: {final_result.slippage:.4%}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return self._create_failed_result(request, str(e))
    
    def _optimize_execution_timing(self, request: ExecutionRequest, 
                                microstructure: Optional[MicrostructureSignals]) -> TimingAnalysis:
        """Optimize execution timing based on market conditions."""
        if not microstructure:
            # Default timing analysis
            return TimingAnalysis(
                optimal_entry_time=datetime.now(),
                market_pressure=0.0,
                liquidity_score=0.5,
                volatility_adjustment=0.0,
                timing_confidence=0.3,
                reasoning="No microstructure data available"
            )
        
        # Calculate timing score
        timing_score = 0.0
        
        # Market pressure analysis
        pressure_weight = self.timing_models["market_pressure_weight"]
        pressure_score = abs(microstructure.market_pressure)
        timing_score += pressure_score * pressure_weight
        
        # Liquidity analysis
        liquidity_weight = self.timing_models["liquidity_weight"]
        liquidity_scores = {
            LiquidityState.HIGH: 1.0,
            LiquidityState.MEDIUM: 0.7,
            LiquidityState.LOW: 0.4,
            LiquidityState.VERY_LOW: 0.1
        }
        liquidity_score = liquidity_scores.get(microstructure.liquidity_state, 0.5)
        timing_score += liquidity_score * liquidity_weight
        
        # Execution quality analysis
        execution_quality = microstructure.execution_quality
        timing_score += execution_quality * 0.3
        
        # Determine optimal time window
        if microstructure.market_pressure > 0.6:
            time_window = self.timing_models["optimal_time_windows"]["high_volatility"]
        elif microstructure.market_pressure > 0.3:
            time_window = self.timing_models["optimal_time_windows"]["normal_volatility"]
        else:
            time_window = self.timing_models["optimal_time_windows"]["low_volatility"]
        
        # Calculate optimal entry time
        optimal_time = datetime.now() + timedelta(seconds=time_window // 2)
        
        # Generate reasoning
        reasoning_parts = [
            f"Market pressure: {microstructure.market_pressure:.2f}",
            f"Liquidity: {microstructure.liquidity_state.value}",
            f"Execution quality: {microstructure.execution_quality:.2f}",
            f"Time window: {time_window}s"
        ]
        
        return TimingAnalysis(
            optimal_entry_time=optimal_time,
            market_pressure=microstructure.market_pressure,
            liquidity_score=liquidity_score,
            volatility_adjustment=0.0,  # Could be calculated from volatility data
            timing_confidence=min(1.0, timing_score),
            reasoning=" | ".join(reasoning_parts)
        )
    
    def _apply_scaling_strategy(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Apply position scaling strategy."""
        if request.quantity <= request.min_fill_size:
            return [request]
        
        scaling_function = self.scaling_strategies.get(request.scaling_method, self._scale_immediate)
        return scaling_function(request)
    
    def _scale_immediate(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Immediate scaling - no splitting."""
        return [request]
    
    def _scale_dca(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Dollar Cost Averaging scaling."""
        num_splits = min(5, max(2, int(request.quantity / 1000)))  # Split into 2-5 parts
        split_quantity = request.quantity / num_splits
        
        scaled_requests = []
        for i in range(num_splits):
            scaled_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=split_quantity,
                order_type=request.order_type,
                price=request.price,
                scaling_method=request.scaling_method,
                time_limit=request.time_limit,
                slippage_tolerance=request.slippage_tolerance,
                metadata={**request.metadata, "split_index": i, "total_splits": num_splits}
            )
            scaled_requests.append(scaled_request)
        
        return scaled_requests
    
    def _scale_pyramid(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Pyramid scaling - decreasing position sizes."""
        num_levels = min(4, max(2, int(request.quantity / 2000)))  # 2-4 levels
        
        scaled_requests = []
        remaining_quantity = request.quantity
        
        for i in range(num_levels):
            # Calculate size for this level (pyramid: larger first, smaller later)
            if i == 0:
                level_size = remaining_quantity * 0.4  # 40% for first level
            elif i == num_levels - 1:
                level_size = remaining_quantity  # Remaining for last level
            else:
                level_size = remaining_quantity * 0.3  # 30% for middle levels
            
            scaled_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=level_size,
                order_type=request.order_type,
                price=request.price,
                scaling_method=request.scaling_method,
                time_limit=request.time_limit,
                slippage_tolerance=request.slippage_tolerance,
                metadata={**request.metadata, "pyramid_level": i, "total_levels": num_levels}
            )
            scaled_requests.append(scaled_request)
            remaining_quantity -= level_size
        
        return scaled_requests
    
    def _scale_anti_martingale(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Anti-Martingale scaling - increasing position sizes."""
        num_levels = min(4, max(2, int(request.quantity / 2000)))
        
        # Calculate base size (smallest level)
        total_weight = sum(2**i for i in range(num_levels))  # 1 + 2 + 4 + 8 = 15
        base_size = request.quantity / total_weight
        
        scaled_requests = []
        for i in range(num_levels):
            level_size = base_size * (2**i)  # 1x, 2x, 4x, 8x
            
            scaled_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=level_size,
                order_type=request.order_type,
                price=request.price,
                scaling_method=request.scaling_method,
                time_limit=request.time_limit,
                slippage_tolerance=request.slippage_tolerance,
                metadata={**request.metadata, "anti_martingale_level": i, "total_levels": num_levels}
            )
            scaled_requests.append(scaled_request)
        
        return scaled_requests
    
    def _scale_volatility_based(self, request: ExecutionRequest) -> List[ExecutionRequest]:
        """Volatility-based scaling - adjust based on market volatility."""
        # This would use actual volatility data - simplified for now
        volatility_adjustment = 1.0  # Would be calculated from market data
        
        if volatility_adjustment > 2.0:  # High volatility - smaller, more frequent trades
            num_splits = 4
        elif volatility_adjustment > 1.5:  # Medium volatility
            num_splits = 3
        else:  # Low volatility - larger, less frequent trades
            num_splits = 2
        
        split_quantity = request.quantity / num_splits
        
        scaled_requests = []
        for i in range(num_splits):
            scaled_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=split_quantity,
                order_type=request.order_type,
                price=request.price,
                scaling_method=request.scaling_method,
                time_limit=request.time_limit,
                slippage_tolerance=request.slippage_tolerance,
                metadata={**request.metadata, "volatility_split": i, "total_splits": num_splits}
            )
            scaled_requests.append(scaled_request)
        
        return scaled_requests
    
    def _execute_single_order(self, request: ExecutionRequest, 
                            microstructure: Optional[MicrostructureSignals],
                            timing: TimingAnalysis) -> ExecutionResult:
        """Execute a single order."""
        start_time = time.time()
        request_id = f"single_{int(start_time)}"
        
        try:
            # Get execution algorithm
            algorithm = self.execution_algorithms.get(request.order_type, self._execute_market_order)
            
            # Predict slippage
            predicted_slippage = self._predict_slippage(request, microstructure)
            
            # Execute order
            execution_data = algorithm(request, microstructure, timing)
            
            # Calculate actual slippage
            actual_slippage = self._calculate_actual_slippage(request, execution_data)
            
            # Determine fill type
            fill_type = self._determine_fill_type(request, execution_data)
            
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                request_id=request_id,
                symbol=request.symbol,
                direction=request.direction,
                requested_quantity=request.quantity,
                filled_quantity=execution_data.get("filled_quantity", 0.0),
                avg_fill_price=execution_data.get("avg_price", 0.0),
                total_cost=execution_data.get("total_cost", 0.0),
                fees=execution_data.get("fees", 0.0),
                slippage=actual_slippage,
                execution_time=execution_time,
                fill_type=fill_type,
                partial_fills=execution_data.get("partial_fills", []),
                status=execution_data.get("status", "completed"),
                timestamp=datetime.now(),
                metadata={
                    "predicted_slippage": predicted_slippage,
                    "timing_analysis": timing.reasoning,
                    "algorithm": request.order_type.value
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Single order execution failed: {e}")
            return ExecutionResult(
                request_id=request_id,
                symbol=request.symbol,
                direction=request.direction,
                requested_quantity=request.quantity,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                total_cost=0.0,
                fees=0.0,
                slippage=0.0,
                execution_time=execution_time,
                fill_type=FillType.FAILED,
                partial_fills=[],
                status="failed",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _execute_market_order(self, request: ExecutionRequest, 
                             microstructure: Optional[MicrostructureSignals],
                             timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute market order."""
        # Simulate market order execution
        # In production, this would call the actual exchange API
        
        # Simulate immediate fill with some slippage
        base_price = request.price or 50000.0  # Default price for simulation
        
        # Calculate slippage based on market conditions
        slippage = self._predict_slippage(request, microstructure)
        
        if request.direction == SignalDirection.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)
        
        # Simulate partial fills for large orders
        fill_ratio = min(1.0, 1000000 / request.quantity)  # Simulate $1M liquidity
        
        filled_quantity = request.quantity * fill_ratio
        total_cost = filled_quantity * fill_price
        fees = total_cost * 0.001  # 0.1% fee
        
        return {
            "filled_quantity": filled_quantity,
            "avg_price": fill_price,
            "total_cost": total_cost,
            "fees": fees,
            "status": "completed" if fill_ratio >= 0.95 else "partial",
            "partial_fills": []
        }
    
    def _execute_limit_order(self, request: ExecutionRequest, 
                            microstructure: Optional[MicrostructureSignals],
                            timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute limit order."""
        # Simulate limit order execution
        base_price = request.price or 50000.0
        
        # Check if limit price would be hit
        market_price = base_price  # Simplified - would use real market data
        
        if request.direction == SignalDirection.BUY and market_price <= request.price:
            # Limit order filled
            return self._execute_market_order(request, microstructure, timing)
        elif request.direction == SignalDirection.SELL and market_price >= request.price:
            # Limit order filled
            return self._execute_market_order(request, microstructure, timing)
        else:
            # Limit order not filled
            return {
                "filled_quantity": 0.0,
                "avg_price": 0.0,
                "total_cost": 0.0,
                "fees": 0.0,
                "status": "unfilled",
                "partial_fills": []
            }
    
    def _execute_iceberg_order(self, request: ExecutionRequest, 
                              microstructure: Optional[MicrostructureSignals],
                              timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute iceberg order (hidden large order)."""
        # Split into smaller visible orders
        visible_size = min(request.quantity * 0.1, 1000)  # 10% visible, max 1000
        num_slices = max(1, int(request.quantity / visible_size))
        
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        partial_fills = []
        
        for i in range(num_slices):
            slice_size = min(visible_size, request.quantity - total_filled)
            if slice_size <= 0:
                break
            
            # Execute slice
            slice_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=slice_size,
                order_type=ExecutionType.MARKET,
                price=request.price,
                metadata={"iceberg_slice": i, "total_slices": num_slices}
            )
            
            slice_result = self._execute_market_order(slice_request, microstructure, timing)
            
            total_filled += slice_result["filled_quantity"]
            total_cost += slice_result["total_cost"]
            total_fees += slice_result["fees"]
            
            partial_fills.append({
                "slice": i,
                "quantity": slice_result["filled_quantity"],
                "price": slice_result["avg_price"],
                "cost": slice_result["total_cost"]
            })
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return {
            "filled_quantity": total_filled,
            "avg_price": avg_price,
            "total_cost": total_cost,
            "fees": total_fees,
            "status": "completed" if total_filled >= request.quantity * 0.95 else "partial",
            "partial_fills": partial_fills
        }
    
    def _execute_twap_order(self, request: ExecutionRequest, 
                           microstructure: Optional[MicrostructureSignals],
                           timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute TWAP (Time-Weighted Average Price) order."""
        if not request.time_limit:
            request.time_limit = 300  # 5 minutes default
        
        num_intervals = min(10, max(2, request.time_limit // 30))  # Every 30 seconds
        interval_quantity = request.quantity / num_intervals
        
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        partial_fills = []
        
        for i in range(num_intervals):
            slice_request = ExecutionRequest(
                symbol=request.symbol,
                direction=request.direction,
                quantity=interval_quantity,
                order_type=ExecutionType.MARKET,
                price=request.price,
                metadata={"twap_interval": i, "total_intervals": num_intervals}
            )
            
            slice_result = self._execute_market_order(slice_request, microstructure, timing)
            
            total_filled += slice_result["filled_quantity"]
            total_cost += slice_result["total_cost"]
            total_fees += slice_result["fees"]
            
            partial_fills.append({
                "interval": i,
                "quantity": slice_result["filled_quantity"],
                "price": slice_result["avg_price"],
                "cost": slice_result["total_cost"]
            })
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return {
            "filled_quantity": total_filled,
            "avg_price": avg_price,
            "total_cost": total_cost,
            "fees": total_fees,
            "status": "completed" if total_filled >= request.quantity * 0.95 else "partial",
            "partial_fills": partial_fills
        }
    
    def _execute_vwap_order(self, request: ExecutionRequest, 
                           microstructure: Optional[MicrostructureSignals],
                           timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute VWAP (Volume-Weighted Average Price) order."""
        # Simplified VWAP - would use real volume profile data
        return self._execute_twap_order(request, microstructure, timing)
    
    def _execute_participate_order(self, request: ExecutionRequest, 
                                 microstructure: Optional[MicrostructureSignals],
                                 timing: TimingAnalysis) -> Dict[str, Any]:
        """Execute participate order (participate in market volume)."""
        # Participate in 20% of market volume over time
        participation_rate = 0.2
        
        # Estimate market volume (simplified)
        estimated_market_volume = request.quantity / participation_rate
        
        # Execute as TWAP with volume-based sizing
        return self._execute_twap_order(request, microstructure, timing)
    
    def _predict_slippage(self, request: ExecutionRequest, 
                          microstructure: Optional[MicrostructureSignals]) -> float:
        """Predict slippage for the order."""
        base_slippage = 0.0005  # 0.05% base
        
        # Adjust based on liquidity
        if microstructure:
            liquidity_multipliers = {
                LiquidityState.HIGH: 0.5,
                LiquidityState.MEDIUM: 1.0,
                LiquidityState.LOW: 2.0,
                LiquidityState.VERY_LOW: 4.0
            }
            base_slippage *= liquidity_multipliers.get(microstructure.liquidity_state, 1.0)
        
        # Adjust based on order size (simplified)
        size_multiplier = min(2.0, request.quantity / 10000)  # Scale with order size
        base_slippage *= size_multiplier
        
        # Adjust based on market pressure
        if microstructure:
            pressure_adjustment = abs(microstructure.market_pressure) * 0.0002
            base_slippage += pressure_adjustment
        
        return min(request.slippage_tolerance, base_slippage)
    
    def _calculate_actual_slippage(self, request: ExecutionRequest, execution_data: Dict[str, Any]) -> float:
        """Calculate actual slippage from execution."""
        if execution_data["filled_quantity"] == 0:
            return 0.0
        
        expected_price = request.price or 50000.0  # Simplified
        actual_price = execution_data["avg_price"]
        
        if request.direction == SignalDirection.BUY:
            slippage = (actual_price - expected_price) / expected_price
        else:
            slippage = (expected_price - actual_price) / expected_price
        
        return slippage
    
    def _determine_fill_type(self, request: ExecutionRequest, execution_data: Dict[str, Any]) -> FillType:
        """Determine fill type from execution data."""
        fill_ratio = execution_data["filled_quantity"] / request.quantity
        
        if fill_ratio >= 0.95:
            return FillType.FULL
        elif fill_ratio > 0.0:
            return FillType.PARTIAL
        else:
            return FillType.FAILED
    
    def _aggregate_execution_results(self, request_id: str, request: ExecutionRequest, 
                                   results: List[ExecutionResult]) -> ExecutionResult:
        """Aggregate results from multiple scaled executions."""
        total_filled = sum(r.filled_quantity for r in results)
        total_cost = sum(r.total_cost for r in results)
        total_fees = sum(r.fees for r in results)
        total_time = sum(r.execution_time for r in results)
        
        # Calculate weighted average price
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        # Calculate weighted slippage
        weighted_slippage = sum(r.slippage * r.filled_quantity for r in results) / total_filled if total_filled > 0 else 0.0
        
        # Aggregate partial fills
        all_partial_fills = []
        for r in results:
            all_partial_fills.extend(r.partial_fills)
        
        # Determine overall status
        if total_filled >= request.quantity * 0.95:
            status = "completed"
        elif total_filled > 0:
            status = "partial"
        else:
            status = "failed"
        
        return ExecutionResult(
            request_id=request_id,
            symbol=request.symbol,
            direction=request.direction,
            requested_quantity=request.quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            total_cost=total_cost,
            fees=total_fees,
            slippage=weighted_slippage,
            execution_time=total_time,
            fill_type=FillType.FULL if total_filled >= request.quantity * 0.95 else FillType.PARTIAL,
            partial_fills=all_partial_fills,
            status=status,
            timestamp=datetime.now(),
            metadata={
                "scaling_method": request.scaling_method.value,
                "num_orders": len(results),
                "individual_results": [r.request_id for r in results]
            }
        )
    
    def _create_failed_result(self, request: ExecutionRequest, error: str) -> ExecutionResult:
        """Create failed execution result."""
        return ExecutionResult(
            request_id=f"failed_{int(time.time())}",
            symbol=request.symbol,
            direction=request.direction,
            requested_quantity=request.quantity,
            filled_quantity=0.0,
            avg_fill_price=0.0,
            total_cost=0.0,
            fees=0.0,
            slippage=0.0,
            execution_time=0.0,
            fill_type=FillType.FAILED,
            partial_fills=[],
            status="failed",
            timestamp=datetime.now(),
            metadata={"error": error}
        )
    
    def _update_execution_stats(self, result: ExecutionResult) -> None:
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        
        if result.status == "completed":
            self.execution_stats["successful_executions"] += 1
        
        # Update slippage tracking
        if result.slippage != 0:
            slippages = self.execution_stats["algorithm_performance"][result.metadata.get("algorithm", "unknown")]["slippage"]
            slippages.append(abs(result.slippage))
        
        # Update execution time tracking
        if result.execution_time > 0:
            times = [r.execution_time for r in self.execution_history[-100:]]
            self.execution_stats["avg_execution_time"] = np.mean(times) if times else 0.0
        
        # Update fill rate
        total_requested = sum(r.requested_quantity for r in self.execution_history)
        total_filled = sum(r.filled_quantity for r in self.execution_history)
        self.execution_stats["fill_rate"] = total_filled / total_requested if total_requested > 0 else 0.0
        
        # Update average slippage
        if self.execution_history:
            slippages = [abs(r.slippage) for r in self.execution_history if r.slippage != 0]
            self.execution_stats["avg_slippage"] = np.mean(slippages) if slippages else 0.0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution performance summary."""
        return {
            "total_executions": self.execution_stats["total_executions"],
            "success_rate": self.execution_stats["successful_executions"] / max(1, self.execution_stats["total_executions"]),
            "avg_slippage": self.execution_stats["avg_slippage"],
            "avg_execution_time": self.execution_stats["avg_execution_time"],
            "fill_rate": self.execution_stats["fill_rate"],
            "algorithm_performance": dict(self.execution_stats["algorithm_performance"]),
            "recent_executions": [
                {
                    "symbol": r.symbol,
                    "direction": r.direction.value,
                    "filled_ratio": r.filled_quantity / r.requested_quantity,
                    "slippage": r.slippage,
                    "execution_time": r.execution_time,
                    "status": r.status
                }
                for r in self.execution_history[-10:]
            ]
        }
