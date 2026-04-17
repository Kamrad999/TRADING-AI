"""
Performance analyzer for backtesting results.
Following patterns from VectorBT and AgentQuant repositories.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

from ..infrastructure.logging import get_logger


class PerformanceAnalyzer:
    """
    Performance analyzer for backtesting results.
    
    Following patterns from:
    - VectorBT: Performance metrics calculation
    - AgentQuant: Walk-forward validation metrics
    - Backtrader: Performance analysis
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.logger = get_logger("performance_analyzer")
        
        # Risk-free rate for Sharpe ratio calculation
        self.risk_free_rate = 0.02  # 2% annual
        
        self.logger.info("Performance analyzer initialized")
    
    def calculate_performance_metrics(self, equity_curve: List[Dict[str, Any]], 
                                    trade_history: List[Dict[str, Any]], 
                                    initial_cash: float) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Equity curve data
            trade_history: Trade history
            initial_cash: Initial cash amount
            
        Returns:
            Performance metrics dictionary
        """
        try:
            if not equity_curve:
                return {}
            
            # Calculate returns
            returns = self._calculate_returns(equity_curve)
            
            # Calculate basic metrics
            total_return = (equity_curve[-1]["portfolio_value"] - initial_cash) / initial_cash
            
            # Calculate risk metrics
            volatility = self._calculate_volatility(returns)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # Calculate risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(trade_history)
            
            # Calculate benchmark metrics
            benchmark_metrics = self._calculate_benchmark_metrics(equity_curve)
            
            return {
                "total_return": total_return,
                "annualized_return": self._annualize_return(total_return, len(equity_curve)),
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "trade_metrics": trade_metrics,
                "benchmark_metrics": benchmark_metrics,
                "win_rate": trade_metrics.get("win_rate", 0.0),
                "profit_factor": trade_metrics.get("profit_factor", 0.0),
                "avg_trade_duration": trade_metrics.get("avg_trade_duration", 0.0),
                "total_trades": trade_metrics.get("total_trades", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Performance calculation failed: {e}")
            return {}
    
    def _calculate_returns(self, equity_curve: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns from equity curve."""
        returns = []
        
        for i in range(1, len(equity_curve)):
            prev_value = equity_curve[i-1]["portfolio_value"]
            curr_value = equity_curve[i]["portfolio_value"]
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        return returns
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return math.sqrt(variance)
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]["portfolio_value"]
        max_drawdown = 0.0
        
        for point in equity_curve:
            current_value = point["portfolio_value"]
            
            if current_value > peak:
                peak = current_value
            
            drawdown = (peak - current_value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: List[float], volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or volatility == 0:
            return 0.0
        
        # Annualize returns and volatility
        avg_return = sum(returns) / len(returns)
        annual_return = avg_return * 252  # 252 trading days
        annual_volatility = volatility * math.sqrt(252)
        
        # Calculate Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        
        return excess_return / annual_volatility
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        avg_return = sum(returns) / len(returns)
        annual_return = avg_return * 252
        
        # Calculate downside deviation
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if annual_return > self.risk_free_rate else 0.0
        
        mean_negative = sum(negative_returns) / len(negative_returns)
        downside_variance = sum((r - mean_negative) ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = math.sqrt(downside_variance) * math.sqrt(252)
        
        excess_return = annual_return - self.risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Annualize return."""
        if num_periods <= 1:
            return total_return
        
        # Assuming daily periods
        years = num_periods / 252
        
        if years <= 0:
            return total_return
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_trade_metrics(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        if not trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade_duration": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trade_history if t["pnl"] > 0]
        losing_trades = [t for t in trade_history if t["pnl"] < 0]
        
        # Calculate basic metrics
        total_trades = len(trade_history)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Calculate P&L metrics
        gross_wins = sum(t["pnl"] for t in winning_trades)
        gross_losses = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        # Calculate average metrics
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Calculate duration metrics
        durations = [t["duration"] for t in trade_history if t["duration"] is not None]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate largest win/loss
        largest_win = max(t["pnl"] for t in winning_trades) if winning_trades else 0.0
        largest_loss = min(t["pnl"] for t in losing_trades) if losing_trades else 0.0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade_duration": avg_trade_duration,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "gross_wins": gross_wins,
            "gross_losses": gross_losses
        }
    
    def _calculate_benchmark_metrics(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        if not equity_curve:
            return {}
        
        # Mock benchmark (e.g., S&P 500) - in production, use real benchmark data
        benchmark_return = 0.08  # 8% annual return
        benchmark_volatility = 0.15  # 15% annual volatility
        
        # Calculate strategy metrics
        strategy_return = equity_curve[-1]["portfolio_value"] / equity_curve[0]["portfolio_value"] - 1
        strategy_volatility = self._calculate_volatility(self._calculate_returns(equity_curve))
        
        # Calculate beta (simplified)
        beta = 1.0  # In production, calculate from correlation with benchmark
        
        # Calculate alpha
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Calculate information ratio
        tracking_error = abs(strategy_volatility - benchmark_volatility)
        information_ratio = (strategy_return - benchmark_return) / tracking_error if tracking_error > 0 else 0.0
        
        return {
            "benchmark_return": benchmark_return,
            "strategy_return": strategy_return,
            "excess_return": strategy_return - benchmark_return,
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error
        }
    
    def calculate_rolling_metrics(self, equity_curve: List[Dict[str, Any]], window_size: int = 30) -> List[Dict[str, Any]]:
        """Calculate rolling performance metrics."""
        if len(equity_curve) < window_size:
            return []
        
        rolling_metrics = []
        
        for i in range(window_size, len(equity_curve)):
            window_data = equity_curve[i-window_size:i+1]
            
            # Calculate window metrics
            window_returns = self._calculate_returns(window_data)
            window_volatility = self._calculate_volatility(window_returns)
            window_drawdown = self._calculate_max_drawdown(window_data)
            
            rolling_metrics.append({
                "date": window_data[-1]["date"],
                "rolling_return": (window_data[-1]["portfolio_value"] - window_data[0]["portfolio_value"]) / window_data[0]["portfolio_value"],
                "rolling_volatility": window_volatility,
                "rolling_drawdown": window_drawdown,
                "rolling_sharpe": self._calculate_sharpe_ratio(window_returns, window_volatility)
            })
        
        return rolling_metrics
    
    def calculate_sector_analysis(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sector-specific performance metrics."""
        if not trade_history:
            return {}
        
        # Group trades by symbol
        symbol_trades = {}
        for trade in trade_history:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        # Calculate metrics for each symbol
        symbol_metrics = {}
        
        for symbol, trades in symbol_trades.items():
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] < 0]
            
            total_pnl = sum(t["pnl"] for t in trades)
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
            
            symbol_metrics[symbol] = {
                "total_trades": len(trades),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_pnl": total_pnl / len(trades) if trades else 0.0,
                "largest_win": max(t["pnl"] for t in winning_trades) if winning_trades else 0.0,
                "largest_loss": min(t["pnl"] for t in losing_trades) if losing_trades else 0.0
            }
        
        return symbol_metrics
    
    def generate_performance_report(self, equity_curve: List[Dict[str, Any]], 
                                  trade_history: List[Dict[str, Any]], 
                                  initial_cash: float) -> str:
        """Generate comprehensive performance report."""
        metrics = self.calculate_performance_metrics(equity_curve, trade_history, initial_cash)
        
        if not metrics:
            return "No performance data available."
        
        report = f"""
PERFORMANCE ANALYSIS REPORT
========================

Portfolio Performance:
- Total Return: {metrics['total_return']:.2%}
- Annualized Return: {metrics['annualized_return']:.2%}
- Volatility: {metrics['volatility']:.2%}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}

Risk-Adjusted Metrics:
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Sortino Ratio: {metrics['sortino_ratio']:.2f}
- Calmar Ratio: {metrics['calmar_ratio']:.2f}

Trading Performance:
- Total Trades: {metrics['total_trades']}
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Average Win: ${metrics['avg_win']:.2f}
- Average Loss: ${metrics['avg_loss']:.2f}
- Average Trade Duration: {metrics['avg_trade_duration']:.1f} hours
- Largest Win: ${metrics['largest_win']:.2f}
- Largest Loss: ${metrics['largest_loss']:.2f}

Benchmark Comparison:
- Strategy Return: {metrics['benchmark_metrics']['strategy_return']:.2%}
- Benchmark Return: {metrics['benchmark_metrics']['benchmark_return']:.2%}
- Excess Return: {metrics['benchmark_metrics']['excess_return']:.2%}
- Alpha: {metrics['benchmark_metrics']['alpha']:.2%}
- Information Ratio: {metrics['benchmark_metrics']['information_ratio']:.2f}

Risk Assessment:
- {'LOW' if metrics['volatility'] < 0.1 else 'MEDIUM' if metrics['volatility'] < 0.2 else 'HIGH'} Volatility
- {'EXCELLENT' if metrics['sharpe_ratio'] > 2.0 else 'GOOD' if metrics['sharpe_ratio'] > 1.0 else 'POOR' if metrics['sharpe_ratio'] > 0.5 else 'VERY POOR'} Risk-Adjusted Returns
- {'LOW' if metrics['max_drawdown'] < 0.1 else 'MEDIUM' if metrics['max_drawdown'] < 0.2 else 'HIGH'} Drawdown Risk
"""
        
        return report
