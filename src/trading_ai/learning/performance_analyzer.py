"""
Performance Analyzer following FinRL patterns.
Analyzes trading performance and provides insights for learning.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass

from ..portfolio.position import Position
from ..infrastructure.logging import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    volatility: float
    calmar_ratio: float


class PerformanceAnalyzer:
    """
    Performance analyzer following FinRL patterns.
    
    Key features:
    - Comprehensive performance metrics calculation
    - Strategy and agent performance comparison
    - Market regime performance analysis
    - Trend and pattern detection
    - Performance attribution analysis
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.logger = get_logger("performance_analyzer")
        
        # Performance history
        self.performance_history = deque(maxlen=1000)
        self.strategy_performance = defaultdict(list)
        self.agent_performance = defaultdict(list)
        self.regime_performance = defaultdict(list)
        
        # Analysis cache
        self.analysis_cache = {}
        self.last_analysis_time = None
        
        self.logger.info("PerformanceAnalyzer initialized")
    
    def analyze_positions(self, positions: List[Position], 
                          benchmark_return: float = 0.0) -> PerformanceMetrics:
        """
        Analyze trading positions and calculate performance metrics.
        
        Args:
            positions: List of closed positions
            benchmark_return: Benchmark return for comparison
            
        Returns:
            PerformanceMetrics object
        """
        try:
            if not positions:
                return self._empty_metrics()
            
            # Basic metrics
            total_trades = len(positions)
            winning_trades = len([p for p in positions if p.realized_pnl > 0])
            losing_trades = total_trades - winning_trades
            
            # Return metrics
            total_pnl = sum(p.realized_pnl for p in positions)
            total_invested = sum(p.entry_value for p in positions)
            total_return = total_pnl / total_invested if total_invested > 0 else 0.0
            
            # Time-based metrics
            if positions:
                start_time = min(p.entry_time for p in positions)
                end_time = max(p.exit_time for p in positions if p.exit_time)
                days = (end_time - start_time).days if end_time and start_time else 1
                annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
            else:
                annual_return = 0.0
            
            # Trade statistics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            winning_pnls = [p.realized_pnl for p in positions if p.realized_pnl > 0]
            losing_pnls = [p.realized_pnl for p in positions if p.realized_pnl < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
            
            largest_win = max(winning_pnls) if winning_pnls else 0.0
            largest_loss = min(losing_pnls) if losing_pnls else 0.0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
            
            # Duration metrics
            durations = [(p.exit_time - p.entry_time).total_seconds() / 3600 
                        for p in positions if p.exit_time and p.entry_time]
            avg_trade_duration = np.mean(durations) if durations else 0.0
            
            # Risk metrics
            returns = [p.realized_pnl / p.entry_value for p in positions if p.entry_value > 0]
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            
            # Drawdown calculation
            cumulative_returns = np.cumsum(returns)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
            # Risk-adjusted returns
            risk_free_rate = 0.02
            excess_returns = [r - risk_free_rate/252 for r in returns]
            
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(excess_returns) > 1 and np.std(excess_returns) > 0 else 0.0
            
            downside_returns = [r for r in excess_returns if r < 0]
            sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 1 and np.std(downside_returns) > 0 else 0.0
            
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                volatility=volatility,
                calmar_ratio=calmar_ratio
            )
            
            # Store in history
            self.performance_history.append((datetime.now(), metrics))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze positions: {e}")
            return self._empty_metrics()
    
    def analyze_strategy_performance(self, positions: List[Position]) -> Dict[str, PerformanceMetrics]:
        """Analyze performance by strategy."""
        strategy_positions = defaultdict(list)
        
        for position in positions:
            strategy = position.strategy
            if strategy:
                strategy_positions[strategy].append(position)
        
        strategy_metrics = {}
        for strategy, strat_positions in strategy_positions.items():
            metrics = self.analyze_positions(strat_positions)
            strategy_metrics[strategy] = metrics
            
            # Store in strategy performance history
            self.strategy_performance[strategy].append((datetime.now(), metrics))
        
        return strategy_metrics
    
    def analyze_agent_performance(self, positions: List[Position]) -> Dict[str, PerformanceMetrics]:
        """Analyze performance by agent (from position metadata)."""
        agent_positions = defaultdict(list)
        
        for position in positions:
            # Try to extract agent from metadata
            agent = position.metadata.get("agent", "Unknown")
            if agent != "Unknown":
                agent_positions[agent].append(position)
        
        agent_metrics = {}
        for agent, agent_positions_list in agent_positions.items():
            metrics = self.analyze_positions(agent_positions_list)
            agent_metrics[agent] = metrics
            
            # Store in agent performance history
            self.agent_performance[agent].append((datetime.now(), metrics))
        
        return agent_metrics
    
    def analyze_regime_performance(self, positions: List[Position]) -> Dict[str, PerformanceMetrics]:
        """Analyze performance by market regime."""
        regime_positions = defaultdict(list)
        
        for position in positions:
            # Try to extract regime from metadata
            regime = position.metadata.get("market_regime", "neutral")
            regime_positions[regime].append(position)
        
        regime_metrics = {}
        for regime, regime_positions_list in regime_positions.items():
            metrics = self.analyze_positions(regime_positions_list)
            regime_metrics[regime] = metrics
            
            # Store in regime performance history
            self.regime_performance[regime].append((datetime.now(), metrics))
        
        return regime_metrics
    
    def analyze_performance_trends(self, window_days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "return_trend": [],
            "sharpe_trend": [],
            "win_rate_trend": [],
            "drawdown_trend": [],
            "volatility_trend": []
        }
        
        try:
            cutoff_time = datetime.now() - timedelta(days=window_days)
            recent_performance = [(time, metrics) for time, metrics in self.performance_history if time >= cutoff_time]
            
            if len(recent_performance) > 1:
                for time, metrics in recent_performance:
                    trends["return_trend"].append((time, metrics.total_return))
                    trends["sharpe_trend"].append((time, metrics.sharpe_ratio))
                    trends["win_rate_trend"].append((time, metrics.win_rate))
                    trends["drawdown_trend"].append((time, metrics.max_drawdown))
                    trends["volatility_trend"].append((time, metrics.volatility))
                
                # Calculate trend directions
                for metric_name, trend_data in trends.items():
                    if len(trend_data) > 1:
                        values = [value for _, value in trend_data]
                        recent_avg = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
                        older_avg = np.mean(values[:5]) if len(values) >= 10 else np.mean(values[:len(values)//2])
                        
                        trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                        trends[f"{metric_name}_direction"] = trend_direction
            
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
        
        return trends
    
    def generate_performance_report(self, positions: List[Position]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "summary": {},
            "strategy_analysis": {},
            "agent_analysis": {},
            "regime_analysis": {},
            "trends": {},
            "recommendations": []
        }
        
        try:
            # Overall performance
            overall_metrics = self.analyze_positions(positions)
            report["summary"] = {
                "total_return": overall_metrics.total_return,
                "annual_return": overall_metrics.annual_return,
                "sharpe_ratio": overall_metrics.sharpe_ratio,
                "max_drawdown": overall_metrics.max_drawdown,
                "win_rate": overall_metrics.win_rate,
                "total_trades": overall_metrics.total_trades,
                "profit_factor": overall_metrics.profit_factor
            }
            
            # Strategy analysis
            strategy_metrics = self.analyze_strategy_performance(positions)
            for strategy, metrics in strategy_metrics.items():
                report["strategy_analysis"][strategy] = {
                    "total_return": metrics.total_return,
                    "win_rate": metrics.win_rate,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "total_trades": metrics.total_trades,
                    "profit_factor": metrics.profit_factor
                }
            
            # Agent analysis
            agent_metrics = self.analyze_agent_performance(positions)
            for agent, metrics in agent_metrics.items():
                report["agent_analysis"][agent] = {
                    "total_return": metrics.total_return,
                    "win_rate": metrics.win_rate,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "total_trades": metrics.total_trades,
                    "profit_factor": metrics.profit_factor
                }
            
            # Regime analysis
            regime_metrics = self.analyze_regime_performance(positions)
            for regime, metrics in regime_metrics.items():
                report["regime_analysis"][regime] = {
                    "total_return": metrics.total_return,
                    "win_rate": metrics.win_rate,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "total_trades": metrics.total_trades,
                    "profit_factor": metrics.profit_factor
                }
            
            # Trends
            report["trends"] = self.analyze_performance_trends()
            
            # Recommendations
            report["recommendations"] = self._generate_performance_recommendations(report)
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
        
        return report
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance-based recommendations."""
        recommendations = []
        
        try:
            summary = report.get("summary", {})
            
            # Overall performance recommendations
            if summary.get("sharpe_ratio", 0) < 1.0:
                recommendations.append("Low Sharpe ratio - improve risk-adjusted returns")
            
            if summary.get("max_drawdown", 0) > 0.2:
                recommendations.append("High drawdown - improve risk management")
            
            if summary.get("win_rate", 0) < 0.4:
                recommendations.append("Low win rate - review entry criteria")
            
            if summary.get("profit_factor", 0) < 1.5:
                recommendations.append("Low profit factor - improve risk/reward ratio")
            
            # Strategy recommendations
            strategy_analysis = report.get("strategy_analysis", {})
            if strategy_analysis:
                best_strategy = max(strategy_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                worst_strategy = min(strategy_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                
                recommendations.append(f"Best performing strategy: {best_strategy[0]}")
                recommendations.append(f"Review {worst_strategy[0]} strategy")
            
            # Agent recommendations
            agent_analysis = report.get("agent_analysis", {})
            if agent_analysis:
                best_agent = max(agent_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                worst_agent = min(agent_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                
                recommendations.append(f"Best performing agent: {best_agent[0]}")
                recommendations.append(f"Review {worst_agent[0]} agent")
            
            # Regime recommendations
            regime_analysis = report.get("regime_analysis", {})
            if regime_analysis:
                best_regime = max(regime_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                worst_regime = min(regime_analysis.items(), key=lambda x: x[1].get("sharpe_ratio", 0))
                
                recommendations.append(f"Best performance in {best_regime[0]} regime")
                recommendations.append(f"Caution in {worst_regime[0]} regime")
            
            # Trend recommendations
            trends = report.get("trends", {})
            if trends.get("return_trend_direction") == "declining":
                recommendations.append("Returns declining - review strategy")
            
            if trends.get("volatility_trend_direction") == "increasing":
                recommendations.append("Volatility increasing - adjust risk parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate performance recommendations")
        
        return recommendations
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_trade_duration=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            volatility=0.0,
            calmar_ratio=0.0
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            "total_analyses": len(self.performance_history),
            "recent_performance": None,
            "strategy_comparison": {},
            "agent_comparison": {},
            "regime_comparison": {}
        }
        
        try:
            # Recent performance
            if self.performance_history:
                summary["recent_performance"] = self.performance_history[-1][1]
            
            # Strategy comparison
            for strategy, history in self.strategy_performance.items():
                if history:
                    latest_metrics = history[-1][1]
                    summary["strategy_comparison"][strategy] = {
                        "total_return": latest_metrics.total_return,
                        "sharpe_ratio": latest_metrics.sharpe_ratio,
                        "win_rate": latest_metrics.win_rate,
                        "total_trades": latest_metrics.total_trades
                    }
            
            # Agent comparison
            for agent, history in self.agent_performance.items():
                if history:
                    latest_metrics = history[-1][1]
                    summary["agent_comparison"][agent] = {
                        "total_return": latest_metrics.total_return,
                        "sharpe_ratio": latest_metrics.sharpe_ratio,
                        "win_rate": latest_metrics.win_rate,
                        "total_trades": latest_metrics.total_trades
                    }
            
            # Regime comparison
            for regime, history in self.regime_performance.items():
                if history:
                    latest_metrics = history[-1][1]
                    summary["regime_comparison"][regime] = {
                        "total_return": latest_metrics.total_return,
                        "sharpe_ratio": latest_metrics.sharpe_ratio,
                        "win_rate": latest_metrics.win_rate,
                        "total_trades": latest_metrics.total_trades
                    }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
        
        return summary
    
    def clear_history(self) -> None:
        """Clear performance history."""
        self.performance_history.clear()
        self.strategy_performance.clear()
        self.agent_performance.clear()
        self.regime_performance.clear()
        self.analysis_cache.clear()
        
        self.logger.info("Performance history cleared")
