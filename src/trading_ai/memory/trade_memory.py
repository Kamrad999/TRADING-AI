"""
Trade memory system for storing and analyzing trade history.
Following patterns from ai-hedge-fund-crypto and AgentQuant repositories.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from ..infrastructure.logging import get_logger
from ..infrastructure.config import config


@dataclass
class TradeRecord:
    """Trade record structure."""
    trade_id: str
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    status: str
    fees: float
    confidence: float
    reasoning: str
    market_conditions: Dict[str, Any]
    agent_decisions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class Pattern:
    """Trading pattern identified from history."""
    pattern_type: str
    symbol: str
    success_rate: float
    avg_return: float
    frequency: int
    conditions: Dict[str, Any]
    discovered_at: datetime


class TradeMemory:
    """
    Trade memory system for storing and analyzing trade history.
    
    Following patterns from:
    - ai-hedge-fund-crypto: Memory system for agent learning
    - AgentQuant: Strategy research with historical data
    - ai-trade: Trade logging and analysis
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize trade memory."""
        self.logger = get_logger("trade_memory")
        
        # Storage configuration
        self.storage_path = storage_path or config.get("TRADE_MEMORY_PATH", "data/trade_memory.json")
        self.max_memory_size = 10000  # Maximum trades to store
        
        # Memory storage
        self.trades: List[TradeRecord] = []
        self.patterns: List[Pattern] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Analysis configuration
        self.pattern_discovery_enabled = True
        self.learning_enabled = True
        
        # Load existing memory
        self._load_memory()
        
        self.logger.info(f"Trade memory initialized with {len(self.trades)} trades")
    
    def add_trade(self, trade_record: TradeRecord) -> bool:
        """
        Add trade record to memory.
        
        Args:
            trade_record: Trade record to add
            
        Returns:
            True if added successfully
        """
        try:
            # Validate trade record
            if not self._validate_trade_record(trade_record):
                return False
            
            # Add to memory
            self.trades.append(trade_record)
            
            # Maintain memory size
            if len(self.trades) > self.max_memory_size:
                self.trades = self.trades[-self.max_memory_size:]
            
            # Invalidate cache
            self.performance_cache.clear()
            
            # Discover patterns if enabled
            if self.pattern_discovery_enabled:
                self._discover_patterns()
            
            # Save memory
            self._save_memory()
            
            self.logger.info(f"Trade added to memory: {trade_record.trade_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add trade to memory: {e}")
            return False
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         limit: int = 100, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[TradeRecord]:
        """
        Get trade history with filtering.
        
        Args:
            symbol: Filter by symbol
            limit: Maximum number of trades
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Filtered trade history
        """
        filtered_trades = self.trades
        
        # Filter by symbol
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
        
        # Filter by date range
        if start_date:
            filtered_trades = [t for t in filtered_trades if t.entry_time >= start_date]
        
        if end_date:
            filtered_trades = [t for t in filtered_trades if t.entry_time <= end_date]
        
        # Sort by entry time (most recent first)
        filtered_trades.sort(key=lambda t: t.entry_time, reverse=True)
        
        # Limit results
        return filtered_trades[:limit]
    
    def get_performance_metrics(self, symbol: Optional[str] = None,
                              period_days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            symbol: Filter by symbol
            period_days: Analysis period in days
            
        Returns:
            Performance metrics
        """
        cache_key = f"perf_{symbol}_{period_days}"
        
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # Get recent trades
        start_date = datetime.now() - timedelta(days=period_days)
        recent_trades = self.get_trade_history(symbol, start_date=start_date)
        
        if not recent_trades:
            return {}
        
        # Calculate metrics
        closed_trades = [t for t in recent_trades if t.status == "closed"]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        # Basic metrics
        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in closed_trades)
        gross_wins = sum(t.pnl for t in winning_trades)
        gross_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Average metrics
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Risk metrics
        max_win = max(t.pnl for t in winning_trades) if winning_trades else 0.0
        max_loss = min(t.pnl for t in losing_trades) if losing_trades else 0.0
        
        # Time metrics
        trade_durations = []
        for trade in closed_trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                trade_durations.append(duration)
        
        avg_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0.0
        
        metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "max_win": max_win,
            "max_loss": max_loss,
            "avg_duration": avg_duration,
            "period_days": period_days,
            "symbol": symbol
        }
        
        # Cache results
        self.performance_cache[cache_key] = metrics
        
        return metrics
    
    def get_market_conditions_performance(self) -> Dict[str, Any]:
        """Analyze performance by market conditions."""
        condition_performance = {}
        
        for trade in self.trades:
            if trade.status != "closed":
                continue
            
            # Group by market regime
            regime = trade.market_conditions.get("regime", "unknown")
            if regime not in condition_performance:
                condition_performance[regime] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0
                }
            
            condition_performance[regime]["trades"] += 1
            condition_performance[regime]["total_pnl"] += trade.pnl
            
            if trade.pnl > 0:
                condition_performance[regime]["wins"] += 1
        
        # Calculate win rates
        for regime, data in condition_performance.items():
            if data["trades"] > 0:
                data["win_rate"] = data["wins"] / data["trades"]
                data["avg_pnl"] = data["total_pnl"] / data["trades"]
        
        return condition_performance
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Analyze performance by agent decisions."""
        agent_performance = {}
        
        for trade in self.trades:
            if trade.status != "closed":
                continue
            
            for agent_decision in trade.agent_decisions:
                agent_name = agent_decision.get("agent", "unknown")
                
                if agent_name not in agent_performance:
                    agent_performance[agent_name] = {
                        "decisions": 0,
                        "correct_decisions": 0,
                        "avg_confidence": 0.0
                    }
                
                agent_performance[agent_name]["decisions"] += 1
                agent_performance[agent_name]["avg_confidence"] += agent_decision.get("confidence", 0.0)
                
                # Check if agent decision was correct
                agent_action = agent_decision.get("action", "HOLD")
                if (agent_action == "BUY" and trade.pnl > 0) or (agent_action == "SELL" and trade.pnl > 0):
                    agent_performance[agent_name]["correct_decisions"] += 1
        
        # Calculate accuracy
        for agent, data in agent_performance.items():
            if data["decisions"] > 0:
                data["accuracy"] = data["correct_decisions"] / data["decisions"]
                data["avg_confidence"] = data["avg_confidence"] / data["decisions"]
        
        return agent_performance
    
    def get_mistakes_analysis(self) -> Dict[str, Any]:
        """Analyze common mistakes from losing trades."""
        losing_trades = [t for t in self.trades if t.status == "closed" and t.pnl < 0]
        
        if not losing_trades:
            return {}
        
        # Analyze common patterns in losing trades
        mistakes = {
            "entry_timing": 0,
            "exit_timing": 0,
            "position_size": 0,
            "market_conditions": 0,
            "low_confidence": 0
        }
        
        for trade in losing_trades:
            # Check for entry timing issues
            if trade.confidence < 0.6:
                mistakes["low_confidence"] += 1
            
            # Check for position size issues
            if trade.quantity > 0.2:  # Large position
                mistakes["position_size"] += 1
            
            # Check for market condition issues
            regime = trade.market_conditions.get("regime", "unknown")
            if regime in ["volatile", "crisis"]:
                mistakes["market_conditions"] += 1
        
        # Calculate percentages
        total_losing = len(losing_trades)
        for mistake_type, count in mistakes.items():
            mistakes[mistake_type] = count / total_losing
        
        return mistakes
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights from trade history."""
        insights = {
            "best_conditions": self._get_best_market_conditions(),
            "worst_conditions": self._get_worst_market_conditions(),
            "optimal_confidence": self._get_optimal_confidence(),
            "position_size_analysis": self._get_position_size_analysis(),
            "patterns": self.patterns[:5]  # Top 5 patterns
        }
        
        return insights
    
    def _get_best_market_conditions(self) -> List[Dict[str, Any]]:
        """Get best performing market conditions."""
        condition_performance = self.get_market_conditions_performance()
        
        # Sort by win rate
        best_conditions = sorted(
            condition_performance.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )
        
        return [
            {"condition": cond, **metrics}
            for cond, metrics in best_conditions[:3]
        ]
    
    def _get_worst_market_conditions(self) -> List[Dict[str, Any]]:
        """Get worst performing market conditions."""
        condition_performance = self.get_market_conditions_performance()
        
        # Sort by win rate
        worst_conditions = sorted(
            condition_performance.items(),
            key=lambda x: x[1]["win_rate"]
        )
        
        return [
            {"condition": cond, **metrics}
            for cond, metrics in worst_conditions[:3]
        ]
    
    def _get_optimal_confidence(self) -> Dict[str, Any]:
        """Analyze optimal confidence levels."""
        confidence_buckets = {}
        
        for trade in self.trades:
            if trade.status != "closed":
                continue
            
            # Bucket by confidence
            confidence_bucket = int(trade.confidence * 10) / 10  # Round to 0.1
            if confidence_bucket not in confidence_buckets:
                confidence_buckets[confidence_bucket] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0
                }
            
            confidence_buckets[confidence_bucket]["trades"] += 1
            confidence_buckets[confidence_bucket]["total_pnl"] += trade.pnl
            
            if trade.pnl > 0:
                confidence_buckets[confidence_bucket]["wins"] += 1
        
        # Calculate metrics for each bucket
        for conf_bucket, data in confidence_buckets.items():
            if data["trades"] > 0:
                data["win_rate"] = data["wins"] / data["trades"]
                data["avg_pnl"] = data["total_pnl"] / data["trades"]
        
        return confidence_buckets
    
    def _get_position_size_analysis(self) -> Dict[str, Any]:
        """Analyze performance by position size."""
        size_buckets = {
            "small": {"trades": 0, "wins": 0, "total_pnl": 0.0},  # < 5%
            "medium": {"trades": 0, "wins": 0, "total_pnl": 0.0},  # 5-15%
            "large": {"trades": 0, "wins": 0, "total_pnl": 0.0}   # > 15%
        }
        
        for trade in self.trades:
            if trade.status != "closed":
                continue
            
            # Determine size bucket
            if trade.quantity < 0.05:
                bucket = "small"
            elif trade.quantity < 0.15:
                bucket = "medium"
            else:
                bucket = "large"
            
            size_buckets[bucket]["trades"] += 1
            size_buckets[bucket]["total_pnl"] += trade.pnl
            
            if trade.pnl > 0:
                size_buckets[bucket]["wins"] += 1
        
        # Calculate metrics
        for bucket, data in size_buckets.items():
            if data["trades"] > 0:
                data["win_rate"] = data["wins"] / data["trades"]
                data["avg_pnl"] = data["total_pnl"] / data["trades"]
        
        return size_buckets
    
    def _discover_patterns(self):
        """Discover trading patterns from history."""
        if not self.learning_enabled:
            return
        
        # Simple pattern discovery - in production, use ML algorithms
        self._discover_win_patterns()
        self._discover_loss_patterns()
    
    def _discover_win_patterns(self):
        """Discover patterns in winning trades."""
        winning_trades = [t for t in self.trades if t.status == "closed" and t.pnl > 0]
        
        if len(winning_trades) < 5:
            return
        
        # Analyze common conditions in winning trades
        common_conditions = {
            "high_confidence": 0,
            "bullish_regime": 0,
            "low_volatility": 0,
            "technical_alignment": 0
        }
        
        for trade in winning_trades:
            if trade.confidence > 0.7:
                common_conditions["high_confidence"] += 1
            
            if trade.market_conditions.get("regime") == "bullish":
                common_conditions["bullish_regime"] += 1
            
            if trade.market_conditions.get("volatility", 1.0) < 0.05:
                common_conditions["low_volatility"] += 1
        
        # Create pattern if significant
        total_wins = len(winning_trades)
        for condition, count in common_conditions.items():
            if count / total_wins > 0.6:  # 60% threshold
                pattern = Pattern(
                    pattern_type=f"winning_{condition}",
                    symbol="all",
                    success_rate=count / total_wins,
                    avg_return=sum(t.pnl for t in winning_trades) / total_wins,
                    frequency=count,
                    conditions={condition: True},
                    discovered_at=datetime.now()
                )
                self.patterns.append(pattern)
    
    def _discover_loss_patterns(self):
        """Discover patterns in losing trades."""
        losing_trades = [t for t in self.trades if t.status == "closed" and t.pnl < 0]
        
        if len(losing_trades) < 5:
            return
        
        # Analyze common conditions in losing trades
        common_conditions = {
            "low_confidence": 0,
            "bearish_regime": 0,
            "high_volatility": 0,
            "large_position": 0
        }
        
        for trade in losing_trades:
            if trade.confidence < 0.5:
                common_conditions["low_confidence"] += 1
            
            if trade.market_conditions.get("regime") == "bearish":
                common_conditions["bearish_regime"] += 1
            
            if trade.market_conditions.get("volatility", 0.0) > 0.08:
                common_conditions["high_volatility"] += 1
            
            if trade.quantity > 0.15:
                common_conditions["large_position"] += 1
        
        # Create pattern if significant
        total_losses = len(losing_trades)
        for condition, count in common_conditions.items():
            if count / total_losses > 0.6:  # 60% threshold
                pattern = Pattern(
                    pattern_type=f"losing_{condition}",
                    symbol="all",
                    success_rate=count / total_losses,
                    avg_return=sum(t.pnl for t in losing_trades) / total_losses,
                    frequency=count,
                    conditions={condition: True},
                    discovered_at=datetime.now()
                )
                self.patterns.append(pattern)
    
    def _validate_trade_record(self, trade: TradeRecord) -> bool:
        """Validate trade record."""
        # Check required fields
        if not trade.trade_id or not trade.symbol:
            return False
        
        if trade.entry_time is None:
            return False
        
        if trade.quantity <= 0 or trade.entry_price <= 0:
            return False
        
        return True
    
    def _load_memory(self):
        """Load memory from storage."""
        try:
            storage_file = Path(self.storage_path)
            
            if storage_file.exists():
                with open(storage_file, 'r') as f:
                    data = json.load(f)
                
                # Load trades
                for trade_data in data.get("trades", []):
                    # Convert datetime strings back to datetime objects
                    trade_data["entry_time"] = datetime.fromisoformat(trade_data["entry_time"])
                    if trade_data.get("exit_time"):
                        trade_data["exit_time"] = datetime.fromisoformat(trade_data["exit_time"])
                    
                    trade = TradeRecord(**trade_data)
                    self.trades.append(trade)
                
                # Load patterns
                for pattern_data in data.get("patterns", []):
                    pattern_data["discovered_at"] = datetime.fromisoformat(pattern_data["discovered_at"])
                    pattern = Pattern(**pattern_data)
                    self.patterns.append(pattern)
                
                self.logger.info(f"Loaded {len(self.trades)} trades and {len(self.patterns)} patterns from memory")
            
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
    
    def _save_memory(self):
        """Save memory to storage."""
        try:
            storage_file = Path(self.storage_path)
            storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "trades": [asdict(trade) for trade in self.trades],
                "patterns": [asdict(pattern) for pattern in self.patterns]
            }
            
            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    def cleanup(self):
        """Cleanup memory system."""
        self._save_memory()
        self.trades.clear()
        self.patterns.clear()
        self.performance_cache.clear()
        
        self.logger.info("Trade memory cleaned up")
