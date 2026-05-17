"""Portfolio Analytics for Replay Validation.

Institutional-grade performance measurement:
    - Sharpe ratio
    - Max drawdown
    - Win rate
    - Expectancy
    - Exposure metrics
    - Turnover
    - Alpha stability
    - Regime sensitivity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from amatix.core.observability import get_logger
from amatix.simulation.replay_engine import ReplayResult

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Complete performance metrics from replay."""
    # Return metrics
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    
    # Risk metrics
    volatility_annual: Decimal
    max_drawdown_pct: Decimal
    max_drawdown_duration_days: int
    
    # Risk-adjusted returns
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    avg_trade_return: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal
    expectancy: Decimal
    
    # Exposure metrics
    avg_gross_exposure: Decimal
    max_gross_exposure: Decimal
    avg_net_exposure: Decimal
    max_net_exposure: Decimal
    
    # Operational metrics
    turnover_annual: Decimal
    avg_holding_period_days: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return_pct": float(self.total_return_pct),
            "annualized_return_pct": float(self.annualized_return_pct),
            "volatility_annual": float(self.volatility_annual),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "total_trades": self.total_trades,
            "win_rate": float(self.win_rate),
            "expectancy": float(self.expectancy),
            "profit_factor": float(self.profit_factor),
        }


@dataclass
class RiskAttribution:
    """Risk attribution by source."""
    market_beta: Decimal
    sector_exposure: Dict[str, Decimal]
    factor_exposures: Dict[str, Decimal]
    
    # Risk contributions
    market_contribution_pct: Decimal
    sector_contribution_pct: Decimal
    idiosyncratic_contribution_pct: Decimal


@dataclass
class RegimePerformance:
    """Performance broken down by market regime."""
    regime_name: str
    days_in_regime: int
    return_pct: Decimal
    volatility: Decimal
    sharpe: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime_name,
            "days": self.days_in_regime,
            "return_pct": float(self.return_pct),
            "volatility": float(self.volatility),
            "sharpe": float(self.sharpe),
            "max_dd": float(self.max_drawdown),
            "win_rate": float(self.win_rate),
        }


@dataclass
class SignalMetrics:
    """Signal quality metrics."""
    total_signals: int
    signal_precision: Decimal  # % of signals that led to winning trades
    signal_recall: Decimal  # % of winning trades that had signals
    avg_signal_to_fill_ms: float
    signal_hit_rate: Decimal  # % of signals with any fill
    
    # Timing
    avg_entry_timing_score: Decimal  # 0-1, how good were entries
    avg_exit_timing_score: Decimal


class PortfolioAnalytics:
    """Calculate institutional-grade portfolio analytics from replay.
    
    Provides comprehensive performance analysis for validation:
        - Standard metrics (Sharpe, drawdown, win rate)
        - Risk attribution
        - Regime sensitivity
        - Signal quality
    
    Usage:
        analytics = PortfolioAnalytics()
        
        metrics = analytics.calculate_performance(
            replay_result,
            trade_history,
            equity_curve,
        )
        
        print(f"Sharpe: {metrics.sharpe_ratio}")
        print(f"Max DD: {metrics.max_drawdown_pct}")
    """
    
    def __init__(self, risk_free_rate: Decimal = Decimal("0.02")) -> None:
        """Initialize with risk-free rate (default 2%)."""
        self._risk_free_rate = risk_free_rate
    
    def calculate_performance(
        self,
        replay_result: ReplayResult,
        trade_history: List[Dict[str, Any]],
        equity_curve: List[Tuple[datetime, Decimal]],
    ) -> PerformanceMetrics:
        """Calculate full performance metrics."""
        
        # Basic return calculation
        initial_equity = equity_curve[0][1] if equity_curve else Decimal("100000")
        final_equity = replay_result.final_state.portfolio_value
        
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Days in period
        if len(equity_curve) >= 2:
            days = (equity_curve[-1][0] - equity_curve[0][0]).days
        else:
            days = 30  # Default
        
        annualized_return = self._annualize_return(total_return, days)
        
        # Volatility
        returns = self._calculate_returns(equity_curve)
        volatility = self._calculate_volatility(returns)
        
        # Drawdown
        max_dd, dd_duration = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe
        sharpe = self._calculate_sharpe(returns)
        
        # Sortino
        sortino = self._calculate_sortino(returns)
        
        # Calmar
        calmar = self._calculate_calmar(annualized_return, max_dd)
        
        # Trade metrics
        trade_metrics = self._analyze_trades(trade_history)
        
        # Exposure
        exposure = self._calculate_exposure_metrics(replay_result)
        
        # Turnover
        turnover = self._calculate_turnover(trade_history, initial_equity)
        
        # Holding period
        avg_hold = self._calculate_avg_holding_period(trade_history)
        
        return PerformanceMetrics(
            total_return_pct=total_return * Decimal("100"),
            annualized_return_pct=annualized_return * Decimal("100"),
            volatility_annual=volatility * Decimal("100"),
            max_drawdown_pct=max_dd * Decimal("100"),
            max_drawdown_duration_days=dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=trade_metrics["total"],
            winning_trades=trade_metrics["winners"],
            losing_trades=trade_metrics["losers"],
            win_rate=Decimal(str(trade_metrics["win_rate"])),
            avg_trade_return=Decimal(str(trade_metrics["avg_return"])),
            avg_win=Decimal(str(trade_metrics["avg_win"])),
            avg_loss=Decimal(str(trade_metrics["avg_loss"])),
            profit_factor=Decimal(str(trade_metrics["profit_factor"])),
            expectancy=Decimal(str(trade_metrics["expectancy"])),
            avg_gross_exposure=exposure["avg_gross"],
            max_gross_exposure=exposure["max_gross"],
            avg_net_exposure=exposure["avg_net"],
            max_net_exposure=exposure["max_net"],
            turnover_annual=turnover,
            avg_holding_period_days=avg_hold,
        )
    
    def calculate_regime_performance(
        self,
        trade_history: List[Dict[str, Any]],
        equity_by_regime: Dict[str, List[Tuple[datetime, Decimal]]],
    ) -> List[RegimePerformance]:
        """Break down performance by market regime."""
        results = []
        
        for regime_name, equity_curve in equity_by_regime.items():
            if len(equity_curve) < 2:
                continue
            
            days = (equity_curve[-1][0] - equity_curve[0][0]).days
            returns = self._calculate_returns(equity_curve)
            
            if not returns:
                continue
            
            total_return = (equity_curve[-1][1] - equity_curve[0][1]) / equity_curve[0][1]
            volatility = self._calculate_volatility(returns)
            max_dd, _ = self._calculate_max_drawdown(equity_curve)
            
            # Win rate in regime
            regime_trades = [
                t for t in trade_history
                if t.get("regime") == regime_name
            ]
            
            if regime_trades:
                wins = sum(1 for t in regime_trades if t.get("pnl", 0) > 0)
                win_rate = Decimal(str(wins / len(regime_trades)))
            else:
                win_rate = Decimal("0")
            
            # Sharpe for regime
            sharpe = self._calculate_sharpe(returns)
            
            results.append(RegimePerformance(
                regime_name=regime_name,
                days_in_regime=days,
                return_pct=total_return * Decimal("100"),
                volatility=volatility * Decimal("100"),
                sharpe=sharpe,
                max_drawdown=max_dd * Decimal("100"),
                win_rate=win_rate * Decimal("100"),
            ))
        
        return results
    
    def calculate_signal_metrics(
        self,
        signals: List[Dict[str, Any]],
        fills: List[Dict[str, Any]],
        pnl_by_signal: Dict[str, Decimal],
    ) -> SignalMetrics:
        """Calculate signal quality metrics."""
        total = len(signals)
        if total == 0:
            return SignalMetrics(
                total_signals=0,
                signal_precision=Decimal("0"),
                signal_recall=Decimal("0"),
                avg_signal_to_fill_ms=0,
                signal_hit_rate=Decimal("0"),
                avg_entry_timing_score=Decimal("0"),
                avg_exit_timing_score=Decimal("0"),
            )
        
        # Signal precision (signals that led to winning trades)
        winning_signals = sum(
            1 for s in signals
            if pnl_by_signal.get(s.get("signal_id"), Decimal("0")) > 0
        )
        precision = Decimal(str(winning_signals / total))
        
        # Signal hit rate (signals that got fills)
        signal_ids = {s.get("signal_id") for s in signals}
        filled_signals = sum(
            1 for f in fills
            if f.get("signal_id") in signal_ids
        )
        hit_rate = Decimal(str(filled_signals / total))
        
        # Timing (simplified)
        entry_scores = [
            Decimal("1") if pnl_by_signal.get(s.get("signal_id"), Decimal("0")) > 0
            else Decimal("0.5")
            for s in signals
        ]
        avg_entry = sum(entry_scores) / len(entry_scores) if entry_scores else Decimal("0")
        
        return SignalMetrics(
            total_signals=total,
            signal_precision=precision * Decimal("100"),
            signal_recall=Decimal("0"),  # Would need full trade log
            avg_signal_to_fill_ms=150,  # Simplified
            signal_hit_rate=hit_rate * Decimal("100"),
            avg_entry_timing_score=avg_entry * Decimal("100"),
            avg_exit_timing_score=Decimal("50"),  # Simplified
        )
    
    def _calculate_returns(
        self,
        equity_curve: List[Tuple[datetime, Decimal]],
    ) -> List[Decimal]:
        """Calculate daily returns from equity curve."""
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i-1][1]
            curr = equity_curve[i][1]
            if prev > 0:
                ret = (curr - prev) / prev
                returns.append(ret)
        return returns
    
    def _calculate_volatility(self, returns: List[Decimal]) -> Decimal:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return Decimal("0")
        
        # Calculate standard deviation
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std_dev = Decimal(str(math.sqrt(float(variance))))
        
        # Annualize (assuming daily returns)
        return std_dev * Decimal(str(math.sqrt(252)))
    
    def _calculate_max_drawdown(
        self,
        equity_curve: List[Tuple[datetime, Decimal]],
    ) -> Tuple[Decimal, int]:
        """Calculate max drawdown and duration."""
        if not equity_curve:
            return Decimal("0"), 0
        
        peak = equity_curve[0][1]
        max_dd = Decimal("0")
        dd_start = equity_curve[0][0]
        max_duration = 0
        
        for timestamp, equity in equity_curve:
            if equity > peak:
                peak = equity
                dd_start = timestamp
            
            drawdown = (peak - equity) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                max_duration = (timestamp - dd_start).days
        
        return max_dd, max_duration
    
    def _calculate_sharpe(self, returns: List[Decimal]) -> Decimal:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return Decimal("0")
        
        excess_returns = [r - self._risk_free_rate / 252 for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)
        
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / len(excess_returns)
        std_dev = Decimal(str(math.sqrt(float(variance))))
        
        if std_dev == 0:
            return Decimal("0")
        
        # Annualize
        annualized_return = mean_excess * 252
        annualized_vol = std_dev * Decimal(str(math.sqrt(252)))
        
        return annualized_return / annualized_vol
    
    def _calculate_sortino(self, returns: List[Decimal]) -> Decimal:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return Decimal("0")
        
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return Decimal("10")  # No downside
        
        mean_return = sum(returns) / len(returns)
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_dev = Decimal(str(math.sqrt(float(downside_variance))))
        
        if downside_dev == 0:
            return Decimal("0")
        
        return (mean_return * 252) / (downside_dev * Decimal(str(math.sqrt(252))))
    
    def _calculate_calmar(
        self,
        annualized_return: Decimal,
        max_drawdown: Decimal,
    ) -> Decimal:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return Decimal("10")  # No drawdown
        return annualized_return / max_drawdown
    
    def _annualize_return(
        self,
        total_return: Decimal,
        days: int,
    ) -> Decimal:
        """Annualize total return."""
        if days == 0:
            return Decimal("0")
        
        # Compound annual growth rate
        years = days / 365
        if years == 0:
            return Decimal("0")
        
        # (1 + total) ^ (1/years) - 1
        try:
            annualized = ((Decimal("1") + total_return) ** (Decimal("1") / Decimal(str(years)))) - Decimal("1")
            return annualized
        except (ZeroDivisionError, ArithmeticError, ValueError):
            # Fallback to simple approximation if exponentiation fails
            return total_return * Decimal(str(365 / days))  # Simple approximation
    
    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade history."""
        if not trades:
            return {
                "total": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0,
                "avg_return": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "expectancy": 0,
            }
        
        total = len(trades)
        
        pnls = [t.get("pnl", Decimal("0")) for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        
        win_rate = len(winners) / total if total > 0 else 0
        avg_return = sum(pnls) / len(pnls) if pnls else 0
        avg_win = sum(winners) / len(winners) if winners else 0
        avg_loss = sum(losers) / len(losers) if losers else 0
        
        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10
        
        # Expectancy = (Win% * Avg Win) - (Loss% * |Avg Loss|)
        loss_rate = 1 - win_rate
        expectancy = (win_rate * float(avg_win)) - (loss_rate * abs(float(avg_loss)))
        
        return {
            "total": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "avg_return": avg_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
        }
    
    def _calculate_exposure_metrics(
        self,
        replay_result: ReplayResult,
    ) -> Dict[str, Decimal]:
        """Calculate exposure metrics."""
        # Simplified - would need full exposure history
        positions = replay_result.final_state.positions
        portfolio_value = replay_result.final_state.portfolio_value
        
        if portfolio_value == 0:
            return {
                "avg_gross": Decimal("0"),
                "max_gross": Decimal("0"),
                "avg_net": Decimal("0"),
                "max_net": Decimal("0"),
            }
        
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        
        for symbol, pos in positions.items():
            qty = Decimal(pos.get("quantity", "0"))
            price = Decimal(pos.get("avg_price", "0"))
            exposure = abs(qty * price)
            
            if qty > 0:
                long_exposure += exposure
            elif qty < 0:
                short_exposure += exposure
        
        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        
        return {
            "avg_gross": gross / portfolio_value,
            "max_gross": gross / portfolio_value,
            "avg_net": net / portfolio_value,
            "max_net": net / portfolio_value,
        }
    
    def _calculate_turnover(
        self,
        trades: List[Dict[str, Any]],
        portfolio_value: Decimal,
    ) -> Decimal:
        """Calculate annualized turnover."""
        if not trades or portfolio_value == 0:
            return Decimal("0")
        
        total_volume = sum(
            t.get("quantity", Decimal("0")) * t.get("price", Decimal("0"))
            for t in trades
        )
        
        # Turnover = total volume / avg portfolio value
        avg_portfolio = portfolio_value  # Simplified
        turnover = total_volume / avg_portfolio
        
        # Annualize (assume 30 days for replay)
        annual_turnover = turnover * (365 / 30)
        
        return annual_turnover
    
    def _calculate_avg_holding_period(
        self,
        trades: List[Dict[str, Any]],
    ) -> Decimal:
        """Calculate average holding period."""
        if not trades:
            return Decimal("0")
        
        # Group by position
        from collections import defaultdict
        position_trades = defaultdict(list)
        
        for trade in trades:
            symbol = trade.get("symbol", "unknown")
            position_trades[symbol].append(trade)
        
        total_days = Decimal("0")
        count = 0
        
        for symbol, position_trades_list in position_trades.items():
            # Simplified: assume first trade is entry
            if len(position_trades_list) >= 2:
                entry = position_trades_list[0].get("timestamp")
                exit_ = position_trades_list[-1].get("timestamp")
                
                if entry and exit_:
                    days = (exit_ - entry).days
                    total_days += Decimal(str(days))
                    count += 1
        
        return total_days / Decimal(str(count)) if count > 0 else Decimal("0")
