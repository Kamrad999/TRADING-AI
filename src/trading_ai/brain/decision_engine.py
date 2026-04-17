"""
Production-grade Decision Engine for TRADING-AI system.
Following patterns from Freqtrade, Jesse, VectorBT, and FinRL repositories.

LLM is the FINAL decision maker - no hard thresholds, always produces a decision.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .llm_client import LLMClient, LLMDecision
from .market_context import MarketContext
from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType


@dataclass
class TradingDecision:
    """Final trading decision from LLM."""
    action: str  # BUY | SELL | HOLD
    symbol: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime
    agent_contributions: Dict[str, Any]
    market_context: Dict[str, Any]


class DecisionEngine:
    """
    Production-grade decision engine with LLM as final decision maker.
    
    Following patterns from:
    - Freqtrade: Strategy system and risk management
    - Jesse: Position lifecycle and execution
    - VectorBT: Signal performance pipeline
    - FinRL: Data-driven decision making
    - ai-trade: LLM-based decision engine
    """
    
    def __init__(self):
        """Initialize decision engine."""
        self.logger = get_logger("decision_engine")
        
        # Components
        self.llm_client = LLMClient()
        self.market_context = MarketContext()
        
        # Learning weights (will be updated by learning loop)
        self.agent_weights = {
            "news_weight": 0.3,
            "technical_weight": 0.4,
            "risk_weight": 0.3
        }
        
        # Past trades for learning
        self.past_trades = []
        
        self.logger.info("Decision engine initialized - LLM as final decision maker")
    
    def make_decision(self, symbol: str, market_data: Dict[str, Any], 
                      news_data: List[Dict[str, Any]], positions: Dict[str, float],
                      past_trades: Optional[List[Dict[str, Any]]] = None) -> Optional[TradingDecision]:
        """
        Make trading decision using LLM as final decision maker.
        
        Args:
            symbol: Trading symbol
            market_data: Market data and indicators
            news_data: News articles and sentiment
            positions: Current positions
            past_trades: Historical trades for learning
            
        Returns:
            Final trading decision (ALWAYS produces a decision)
        """
        try:
            # Update past trades for learning
            if past_trades:
                self.past_trades = past_trades[-20:]  # Keep last 20 trades
            
            # Build comprehensive market context
            context = self.market_context.build_context(
                symbol, market_data, news_data, positions
            )
            
            # Analyze agent contributions (for LLM context, not decision making)
            agent_analysis = self._analyze_agent_contributions(context)
            
            # Build LLM prompt with all data
            llm_context = self._build_llm_context(context, agent_analysis, positions)
            
            # LLM makes FINAL decision
            decision = self.llm_client.make_trading_decision(llm_context)
            
            if decision:
                # Create final trading decision
                trading_decision = TradingDecision(
                    action=decision.action,
                    symbol=decision.symbol,
                    confidence=decision.confidence,
                    entry=decision.entry,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    reasoning=decision.reasoning,
                    timestamp=datetime.now(),
                    agent_contributions=agent_analysis,
                    market_context=context
                )
                
                self.logger.info(f"LLM FINAL DECISION: {decision.action} {decision.symbol} (conf: {decision.confidence:.2f})")
                self.logger.info(f"Entry: ${decision.entry:.2f} | SL: ${decision.stop_loss:.2f} | TP: ${decision.take_profit:.2f}")
                self.logger.info(f"Reasoning: {decision.reasoning}")
                
                return trading_decision
            else:
                # Fallback decision - LLM should always return something
                self.logger.warning("LLM returned None, creating fallback decision")
                return self._create_fallback_decision(symbol, context, agent_analysis)
            
        except Exception as e:
            self.logger.error(f"Decision engine failed: {e}")
            # Always return a decision, even on error
            return self._create_fallback_decision(symbol, {}, {})
    
    def _analyze_agent_contributions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual agent contributions for LLM context."""
        contributions = {}
        
        # News analysis
        news_summary = context.get("news_summary", "")
        sentiment_score = context.get("sentiment_score", 0.0)
        news_count = context.get("news_count", 0)
        
        contributions["news"] = {
            "summary": news_summary,
            "sentiment": sentiment_score,
            "count": news_count,
            "signal": "bullish" if sentiment_score > 0.3 else "bearish" if sentiment_score < -0.3 else "neutral"
        }
        
        # Technical analysis
        indicators = context.get("technical_indicators", {})
        rsi = indicators.get("rsi", 50.0)
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        current_price = context.get("current_price", 0.0)
        
        # Determine technical signals
        rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        macd_signal = "bullish" if macd > macd_signal else "bearish" if macd < macd_signal else "neutral"
        
        # SMA trend
        sma_trend = "bullish" if current_price > sma_20 > sma_50 else "bearish" if current_price < sma_20 < sma_50 else "neutral"
        
        contributions["technical"] = {
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "macd": macd,
            "macd_signal": macd_signal,
            "sma_trend": sma_trend,
            "current_price": current_price,
            "signal": "bullish" if (rsi_signal == "oversold" or macd_signal == "bullish") else "bearish" if (rsi_signal == "overbought" or macd_signal == "bearish") else "neutral"
        }
        
        # Risk analysis
        positions = context.get("positions", {})
        current_position = positions.get(context.get("symbol", ""), 0.0)
        volatility = context.get("volatility", 0.0)
        portfolio_value = context.get("portfolio_value", 0.0)
        
        risk_level = "high" if volatility > 0.08 else "medium" if volatility > 0.04 else "low"
        position_risk = "high" if abs(current_position) > 0.5 else "medium" if abs(current_position) > 0.2 else "low"
        
        contributions["risk"] = {
            "current_position": current_position,
            "volatility": volatility,
            "portfolio_value": portfolio_value,
            "risk_level": risk_level,
            "position_risk": position_risk,
            "signal": "cautious" if risk_level == "high" else "normal"
        }
        
        return contributions
    
    def _build_llm_context(self, market_context: Dict[str, Any], 
                          agent_contributions: Dict[str, Any], positions: Dict[str, float]) -> Dict[str, Any]:
        """Build comprehensive context for LLM decision making."""
        
        # Extract key indicators
        indicators = market_context.get("technical_indicators", {})
        
        # Build past trades summary
        recent_trades = self.past_trades[-5:] if self.past_trades else []
        good_trades = [t for t in recent_trades if t.get("pnl_pct", 0) > 0]
        bad_trades = [t for t in recent_trades if t.get("pnl_pct", 0) < 0]
        
        llm_context = {
            "symbol": market_context.get("symbol", ""),
            "current_price": market_context.get("current_price", 0.0),
            "volume": market_context.get("volume", 0.0),
            "volatility": market_context.get("volatility", 0.0),
            "market_trend": market_context.get("market_trend", "neutral"),
            
            # Technical indicators
            "rsi": indicators.get("rsi", 50.0),
            "macd": indicators.get("macd", 0.0),
            "macd_signal": indicators.get("macd_signal", 0.0),
            "sma_20": indicators.get("sma_20", 0.0),
            "sma_50": indicators.get("sma_50", 0.0),
            "ema_12": indicators.get("ema_12", 0.0),
            "ema_26": indicators.get("ema_26", 0.0),
            "bollinger_upper": indicators.get("bollinger_upper", 0.0),
            "bollinger_lower": indicators.get("bollinger_lower", 0.0),
            "atr": indicators.get("atr", 0.0),
            
            # News data
            "news_summary": market_context.get("news_summary", ""),
            "sentiment_score": market_context.get("sentiment_score", 0.0),
            "news_count": market_context.get("news_count", 0),
            
            # Position data
            "current_position": positions.get(market_context.get("symbol", ""), 0.0),
            "portfolio_value": market_context.get("portfolio_value", 0.0),
            
            # Agent contributions
            "news_analysis": agent_contributions.get("news", {}),
            "technical_analysis": agent_contributions.get("technical", {}),
            "risk_analysis": agent_contributions.get("risk", {}),
            
            # Learning data
            "recent_performance": {
                "total_recent_trades": len(recent_trades),
                "good_trades": len(good_trades),
                "bad_trades": len(bad_trades),
                "win_rate": len(good_trades) / len(recent_trades) if recent_trades else 0.5
            },
            
            # Agent weights for context
            "agent_weights": self.agent_weights
        }
        
        return llm_context
    
    def _create_fallback_decision(self, symbol: str, context: Dict[str, Any], 
                                 agent_contributions: Dict[str, Any]) -> TradingDecision:
        """Create fallback decision when LLM fails."""
        current_price = context.get("current_price", 50000.0)
        
        # Simple fallback logic based on technical indicators
        indicators = context.get("technical_indicators", {})
        rsi = indicators.get("rsi", 50.0)
        
        if rsi < 30:
            action = "BUY"
            confidence = 0.6
            entry = current_price
            stop_loss = entry * 0.97
            take_profit = entry * 1.06
            reasoning = f"Oversold conditions (RSI: {rsi:.1f}) - fallback buy signal"
        elif rsi > 70:
            action = "SELL"
            confidence = 0.6
            entry = current_price
            stop_loss = entry * 1.03
            take_profit = entry * 0.94
            reasoning = f"Overbought conditions (RSI: {rsi:.1f}) - fallback sell signal"
        else:
            action = "HOLD"
            confidence = 0.4
            entry = current_price
            stop_loss = entry * 0.95
            take_profit = entry * 1.05
            reasoning = f"Neutral conditions (RSI: {rsi:.1f}) - fallback hold signal"
        
        return TradingDecision(
            action=action,
            symbol=symbol,
            confidence=confidence,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=datetime.now(),
            agent_contributions=agent_contributions,
            market_context=context
        )
    
    def convert_to_signal(self, decision: TradingDecision) -> Signal:
        """
        Convert trading decision to Signal object.
        ALWAYS produces a signal, even for HOLD decisions.
        """
        try:
            # Determine direction
            if decision.action == "BUY":
                direction = SignalDirection.BUY
            elif decision.action == "SELL":
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD  # HOLD produces neutral signal
            
            # Determine urgency
            if decision.confidence > 0.8:
                urgency = Urgency.HIGH
            elif decision.confidence > 0.6:
                urgency = Urgency.MEDIUM
            else:
                urgency = Urgency.LOW
            
            # Determine market regime
            market_trend = decision.market_context.get("market_trend", "neutral")
            if market_trend == "bullish":
                market_regime = MarketRegime.RISK_ON
            elif market_trend == "bearish":
                market_regime = MarketRegime.RISK_OFF
            else:
                market_regime = MarketRegime.SIDEWAYS
            
            # Calculate position size based on confidence
            base_position = 0.1  # 10% base
            position_size = base_position * decision.confidence
            
            # Create signal (ALWAYS returns a signal)
            signal = Signal(
                symbol=decision.symbol,
                direction=direction,
                confidence=decision.confidence,
                urgency=urgency,
                market_regime=market_regime,
                position_size=position_size,
                execution_priority=1,
                signal_type=SignalType.NEWS,
                article_id=None,
                generated_at=decision.timestamp,
                metadata={
                    "decision_engine": True,
                    "llm_decision": True,
                    "action": decision.action,
                    "entry": decision.entry,
                    "stop_loss": decision.stop_loss,
                    "take_profit": decision.take_profit,
                    "reasoning": decision.reasoning,
                    "agent_contributions": decision.agent_contributions,
                    "market_context": {
                        "price": decision.market_context.get("current_price", 0.0),
                        "volume": decision.market_context.get("volume", 0.0),
                        "volatility": decision.market_context.get("volatility", 0.0),
                        "trend": decision.market_context.get("market_trend", "neutral")
                    },
                    "technical_indicators": decision.market_context.get("technical_indicators", {}),
                    "sentiment": decision.market_context.get("sentiment_score", 0.0)
                }
            )
            
            self.logger.info(f"Signal generated: {direction.value} {decision.symbol} (conf: {decision.confidence:.2f})")
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to convert decision to signal: {e}")
            # Return neutral signal on error
            return Signal(
                symbol=decision.symbol,
                direction=SignalDirection.HOLD,
                confidence=0.1,
                urgency=Urgency.LOW,
                market_regime=MarketRegime.SIDEWAYS,
                position_size=0.0,
                execution_priority=1,
                signal_type=SignalType.NEWS,
                article_id=None,
                generated_at=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def update_weights(self, performance_feedback: Dict[str, Any]):
        """Update agent weights based on performance feedback (learning loop)."""
        try:
            # Simple weight adjustment based on performance
            win_rate = performance_feedback.get("win_rate", 0.5)
            
            if win_rate > 0.6:
                # Increase technical weight if performing well
                self.agent_weights["technical_weight"] = min(0.6, self.agent_weights["technical_weight"] + 0.05)
                self.agent_weights["news_weight"] = max(0.2, self.agent_weights["news_weight"] - 0.025)
                self.agent_weights["risk_weight"] = max(0.2, self.agent_weights["risk_weight"] - 0.025)
            elif win_rate < 0.4:
                # Increase news weight if technical not working
                self.agent_weights["news_weight"] = min(0.5, self.agent_weights["news_weight"] + 0.05)
                self.agent_weights["technical_weight"] = max(0.2, self.agent_weights["technical_weight"] - 0.025)
                self.agent_weights["risk_weight"] = max(0.2, self.agent_weights["risk_weight"] - 0.025)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.agent_weights.values())
            if total_weight > 0:
                for key in self.agent_weights:
                    self.agent_weights[key] /= total_weight
            
            self.logger.info(f"Updated agent weights: {self.agent_weights}")
            
        except Exception as e:
            self.logger.error(f"Failed to update weights: {e}")
