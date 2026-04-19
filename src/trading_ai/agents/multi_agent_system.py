"""
Multi-agent system for trading decisions.
Following patterns from ai-hedge-fund-crypto repository.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..brain.llm_client import LLMClient, LLMDecision
from ..infrastructure.logging import get_logger


@dataclass
class AgentScore:
    """Score from individual agent."""
    agent_name: str
    score: float
    confidence: float
    reasoning: str
    timestamp: datetime


@dataclass
class AgentDecision:
    """Decision from individual agent."""
    agent_name: str
    action: str
    symbol: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


class BaseAgent:
    """Base class for trading agents."""
    
    def __init__(self, name: str, llm_client: LLMClient):
        """Initialize agent."""
        self.name = name
        self.llm_client = llm_client
        self.logger = get_logger(f"agent.{name}")
    
    def analyze(self, context: Dict[str, Any]) -> Optional[AgentDecision]:
        """Analyze market context and make decision."""
        raise NotImplementedError("Subclasses must implement analyze method")


class NewsAgent(BaseAgent):
    """
    News analysis agent.
    Following patterns from ai-trade: LLM Trading Brain with structured outputs.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize news agent."""
        super().__init__("news_agent", llm_client)
        
        # News analysis configuration
        self.sentiment_threshold = 0.3
        self.confidence_threshold = 0.6
        self.max_news_age_hours = 24
    
    def analyze(self, context: Dict[str, Any]) -> Optional[AgentDecision]:
        """Analyze news sentiment and make decision."""
        try:
            # Extract news data
            news_data = context.get("news_data", [])
            symbol = context.get("symbol", "")
            
            if not news_data:
                return None
            
            # Build news context
            news_context = self._build_news_context(context)
            
            # Get LLM decision
            decision = self.llm_client.make_trading_decision(news_context)
            
            if decision and self._validate_decision(decision, context):
                return AgentDecision(
                    agent_name=self.name,
                    action=decision.action,
                    symbol=decision.symbol,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    metadata={
                        "sentiment_score": context.get("sentiment_score", 0.0),
                        "news_count": len(news_data),
                        "news_summary": context.get("news_summary", ""),
                        "analysis_type": "news_sentiment"
                    },
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"News agent analysis failed: {e}")
            return None
    
    def _build_news_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build news-specific context for LLM."""
        news_data = context.get("news_data", [])
        
        # Analyze news sentiment
        positive_count = 0
        negative_count = 0
        total_sentiment = 0.0
        
        recent_news = []
        for article in news_data:
            sentiment = article.get("sentiment", 0.0)
            total_sentiment += sentiment
            
            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
                negative_count += 1
            
            recent_news.append({
                "title": article.get("title", ""),
                "sentiment": sentiment,
                "timestamp": article.get("timestamp", "")
            })
        
        avg_sentiment = total_sentiment / len(news_data) if news_data else 0.0
        
        return {
            "symbol": context.get("symbol", ""),
            "news_summary": context.get("news_summary", ""),
            "sentiment_score": avg_sentiment,
            "news_count": len(news_data),
            "positive_news": positive_count,
            "negative_news": negative_count,
            "current_price": context.get("current_price", 0.0),
            "market_trend": context.get("market_trend", "neutral"),
            "agent_type": "news",
            "recent_news": recent_news[:5]  # Top 5 recent news
        }
    
    def _validate_decision(self, decision: LLMDecision, context: Dict[str, Any]) -> bool:
        """Validate decision based on news analysis."""
        # Check confidence threshold
        if decision.confidence < self.confidence_threshold:
            return False
        
        # Check if there's significant news sentiment
        sentiment_score = context.get("sentiment_score", 0.0)
        if abs(sentiment_score) < self.sentiment_threshold:
            return False
        
        # Check action alignment with sentiment
        if sentiment_score > 0 and decision.action == "SELL":
            return False
        elif sentiment_score < 0 and decision.action == "BUY":
            return False
        
        return True


class TechnicalAgent(BaseAgent):
    """
    Technical analysis agent.
    Following patterns from AgentQuant: Feature Engine with indicators.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize technical agent."""
        super().__init__("technical_agent", llm_client)
        
        # Technical analysis configuration
        self.indicator_weights = {
            "rsi": 0.3,
            "macd": 0.25,
            "sma": 0.2,
            "bollinger": 0.15,
            "volume": 0.1
        }
        self.confidence_threshold = 0.6
    
    def analyze(self, context: Dict[str, Any]) -> Optional[AgentDecision]:
        """Analyze technical indicators and make decision."""
        try:
            # Extract technical data
            indicators = context.get("technical_indicators", {})
            symbol = context.get("symbol", "")
            
            if not indicators:
                return None
            
            # Debug: Log all indicators and their types
            for key, value in indicators.items():
                self.logger.debug(f"Indicator {key}: {value} (type: {type(value)})")
            
            # Build technical context
            technical_context = self._build_technical_context(context)
            
            # Get LLM decision
            decision = self.llm_client.make_trading_decision(technical_context)
            
            if decision and self._validate_decision(decision, context):
                signal_strength = self._calculate_signal_strength(indicators)
                trend = self._determine_trend(indicators)
                
                return AgentDecision(
                    agent_name=self.name,
                    action=decision.action,
                    symbol=decision.symbol,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    metadata={
                        "indicators": indicators,
                        "signal_strength": signal_strength,
                        "trend": trend,
                        "analysis_type": "technical_analysis"
                    },
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Technical agent analysis failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _build_technical_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build technical-specific context for LLM."""
        indicators = context.get("technical_indicators", {})
        market_data = context.get("market_data", {})
        
        # Analyze indicator signals
        rsi = indicators.get("rsi", 50.0)
        macd = indicators.get("macd", 0.0)
        macd_signal_line = indicators.get("macd_signal", 0.0)
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        current_price = float(context.get("current_price", 0.0))
        
        # Handle None values and string values
        if macd_signal_line is None:
            macd_signal_line = 0.0
        elif isinstance(macd_signal_line, str):
            # If it's a string like "bullish", we can't convert it to float
            # Use a default value for MACD signal line
            self.logger.debug(f"MACD signal is string '{macd_signal_line}', using default 0.0")
            macd_signal_line = 0.0
        else:
            macd_signal_line = float(macd_signal_line)
        
        # Determine indicator signals
        rsi_signal = "neutral"
        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        
        macd_trend = "neutral"
        if macd > macd_signal_line:
            macd_trend = "bullish"
        elif macd < macd_signal_line:
            macd_trend = "bearish"
        
        sma_signal = "neutral"
        if current_price > sma_20 > sma_50:
            sma_signal = "bullish"
        elif current_price < sma_20 < sma_50:
            sma_signal = "bearish"
        
        return {
            "symbol": context.get("symbol", ""),
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "macd": macd,
            "macd_signal": macd_signal_line,
            "macd_state": macd_trend,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_signal": sma_signal,
            "current_price": current_price,
            "volume": market_data.get("volume", 0.0),
            "volatility": market_data.get("volatility", 0.0),
            "bollinger_upper": indicators.get("bollinger_upper", 0.0),
            "bollinger_lower": indicators.get("bollinger_lower", 0.0),
            "atr": indicators.get("atr", 0.0),
            "agent_type": "technical"
        }
    
    def _calculate_signal_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate overall signal strength from indicators."""
        strength = 0.0
        
        # RSI signal
        rsi = indicators.get("rsi", 50.0)
        self.logger.debug(f"RSI: {rsi} (type: {type(rsi)})")
        if rsi < 30:
            strength += self.indicator_weights["rsi"] * 1.0
        elif rsi < 40:
            strength += self.indicator_weights["rsi"] * 0.5
        elif rsi > 70:
            strength += self.indicator_weights["rsi"] * -1.0
        elif rsi > 60:
            strength += self.indicator_weights["rsi"] * -0.5
        
        # MACD signal
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        
        # Handle None values and string values
        if macd_signal is None:
            self.logger.debug("Skipping MACD comparison due to None signal value")
        elif isinstance(macd_signal, str):
            self.logger.debug(f"Skipping MACD comparison due to string signal value: {macd_signal}")
        else:
            macd_signal = float(macd_signal)
            self.logger.debug(f"MACD: {macd} (type: {type(macd)}), MACD Signal: {macd_signal} (type: {type(macd_signal)})")
            if macd > macd_signal:
                strength += self.indicator_weights["macd"] * 0.5
            else:
                strength += self.indicator_weights["macd"] * -0.5
        
        # SMA signal
        current_price = float(indicators.get("current_price", 0.0))
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        
        # Handle None values
        if sma_20 is None or sma_50 is None:
            self.logger.debug(f"Skipping SMA comparison due to None values - SMA20: {sma_20}, SMA50: {sma_50}")
        else:
            # Convert to float to ensure numeric comparison
            sma_20 = float(sma_20)
            sma_50 = float(sma_50)
            
            self.logger.debug(f"Price: {current_price} (type: {type(current_price)}), SMA20: {sma_20} (type: {type(sma_20)}), SMA50: {sma_50} (type: {type(sma_50)})")
            
            # Check types before comparison
            if not all(isinstance(x, (int, float)) for x in [current_price, sma_20, sma_50]):
                self.logger.error(f"Type error in SMA comparison - Price: {type(current_price)}, SMA20: {type(sma_20)}, SMA50: {type(sma_50)}")
                return 0.0
            
            if current_price > sma_20 > sma_50:
                strength += self.indicator_weights["sma"] * 1.0
            elif current_price < sma_20 < sma_50:
                strength += self.indicator_weights["sma"] * -1.0
        
        return strength
    
    def _determine_trend(self, indicators: Dict[str, float]) -> str:
        """Determine overall trend from indicators."""
        strength = self._calculate_signal_strength(indicators)
        
        if strength > 0.3:
            return "bullish"
        elif strength < -0.3:
            return "bearish"
        else:
            return "neutral"
    
    def _validate_decision(self, decision: LLMDecision, context: Dict[str, Any]) -> bool:
        """Validate decision based on technical analysis."""
        # Check confidence threshold
        if decision.confidence < self.confidence_threshold:
            return False
        
        # Check if decision aligns with technical signals
        indicators = context.get("technical_indicators", {})
        
        # Debug: Print indicator types
        for key, value in indicators.items():
            if not isinstance(value, (int, float)):
                self.logger.error(f"Indicator {key} is not numeric: {type(value)} = {value}")
        
        signal_strength = self._calculate_signal_strength(indicators)
        
        # Validate action alignment
        if signal_strength > 0.3 and decision.action == "SELL":
            return False
        elif signal_strength < -0.3 and decision.action == "BUY":
            return False
        elif abs(signal_strength) < 0.1 and decision.action != "HOLD":
            return False
        
        return True


class RiskAgent(BaseAgent):
    """
    Risk management agent.
    Following patterns from ai-hedge-fund-crypto: Risk Management Node.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize risk agent."""
        super().__init__("risk_agent", llm_client)
        
        # Risk management configuration
        self.max_position_size = 0.2
        self.max_portfolio_risk = 0.3
        self.volatility_threshold = 0.05
        self.confidence_threshold = 0.7
    
    def analyze(self, context: Dict[str, Any]) -> Optional[AgentDecision]:
        """Analyze risk factors and make decision."""
        try:
            # Extract risk data
            positions = context.get("positions", {})
            portfolio_value = context.get("portfolio_value", 0.0)
            symbol = context.get("symbol", "")
            
            # Build risk context
            risk_context = self._build_risk_context(context)
            
            # Get LLM decision
            decision = self.llm_client.make_trading_decision(risk_context)
            
            if decision and self._validate_decision(decision, context):
                return AgentDecision(
                    agent_name=self.name,
                    action=decision.action,
                    symbol=decision.symbol,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    metadata={
                        "risk_score": self._calculate_risk_score(context),
                        "position_size": self._calculate_position_size(context),
                        "risk_level": self._assess_risk_level(context),
                        "analysis_type": "risk_management"
                    },
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Risk agent analysis failed: {e}")
            return None
    
    def _build_risk_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk-specific context for LLM."""
        positions = context.get("positions", {})
        portfolio_value = context.get("portfolio_value", 0.0)
        current_position = positions.get(context.get("symbol", ""), 0.0)
        
        # Calculate position metrics
        current_price = context.get("current_price", 0.0)
        position_value = abs(current_position) * current_price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Risk metrics
        volatility = context.get("volatility", 0.0)
        market_trend = context.get("market_trend", "neutral")
        
        return {
            "symbol": context.get("symbol", ""),
            "current_position": current_position,
            "position_value": position_value,
            "position_pct": position_pct,
            "portfolio_value": portfolio_value,
            "volatility": volatility,
            "market_trend": market_trend,
            "max_position_size": self.max_position_size,
            "risk_tolerance": context.get("risk_tolerance", 0.5),
            "current_price": current_price,
            "recent_performance": context.get("recent_performance", 0.0),
            "agent_type": "risk"
        }
    
    def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        """Calculate overall risk score."""
        risk_score = 0.0
        
        # Position size risk
        positions = context.get("positions", {})
        portfolio_value = context.get("portfolio_value", 0.0)
        symbol = context.get("symbol", "")
        current_position = positions.get(symbol, 0.0)
        current_price = context.get("current_price", 0.0)
        
        if current_position != 0 and portfolio_value > 0:
            position_value = abs(current_position) * current_price
            position_pct = position_value / portfolio_value
            risk_score += position_pct * 0.4
        
        # Volatility risk
        volatility = context.get("volatility", 0.0)
        risk_score += volatility * 0.3
        
        # Market trend risk
        market_trend = context.get("market_trend", "neutral")
        if market_trend == "bearish":
            risk_score += 0.2
        elif market_trend == "volatile":
            risk_score += 0.3
        
        # Portfolio concentration risk
        total_positions = sum(abs(pos) for pos in positions.values())
        if portfolio_value > 0:
            concentration = total_positions / portfolio_value
            risk_score += concentration * 0.1
        
        return min(1.0, risk_score)
    
    def _calculate_position_size(self, context: Dict[str, Any]) -> float:
        """Calculate recommended position size."""
        risk_score = self._calculate_risk_score(context)
        volatility = context.get("volatility", 0.0)
        
        # Base position size
        base_size = self.max_position_size
        
        # Adjust for risk
        risk_adjustment = 1.0 - risk_score
        
        # Adjust for volatility
        if volatility > self.volatility_threshold:
            volatility_adjustment = 0.5
        else:
            volatility_adjustment = 1.0
        
        recommended_size = base_size * risk_adjustment * volatility_adjustment
        
        return max(0.01, min(self.max_position_size, recommended_size))
    
    def _assess_risk_level(self, context: Dict[str, Any]) -> str:
        """Assess overall risk level."""
        risk_score = self._calculate_risk_score(context)
        
        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _validate_decision(self, decision: LLMDecision, context: Dict[str, Any]) -> bool:
        """Validate decision based on risk management."""
        # Check confidence threshold (higher for risk agent)
        if decision.confidence < self.confidence_threshold:
            return False
        
        # Check risk level
        risk_score = self._calculate_risk_score(context)
        if risk_score > 0.8:  # Very high risk - no new positions
            return decision.action == "SELL" or decision.action == "HOLD"
        
        # Check position size limits
        positions = context.get("positions", {})
        symbol = context.get("symbol", "")
        current_position = positions.get(symbol, 0.0)
        
        if current_position > 0 and decision.action == "BUY":
            return False  # Don't add to long position
        elif current_position < 0 and decision.action == "SELL":
            return False  # Don't add to short position
        
        return True


class MultiAgentSystem:
    """
    Multi-agent system for trading decisions with dynamic weighting and conflict resolution.
    Following patterns from ai-hedge-fund-crypto: Multi-agent consensus system.
    """
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.logger = get_logger("multi_agent_system")
        
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Initialize agents
        self.agents = {
            "news_agent": NewsAgent(self.llm_client),
            "technical_agent": TechnicalAgent(self.llm_client),
            "risk_agent": RiskAgent(self.llm_client)
        }
        
        # Dynamic agent weights (adjust based on performance)
        self.base_weights = {
            "news_agent": 0.3,
            "technical_agent": 0.4,
            "risk_agent": 0.3
        }
        self.agent_weights = self.base_weights.copy()
        
        # Performance tracking for dynamic weighting
        self.agent_performance: Dict[str, Dict[str, Any]] = {
            name: {"correct": 0, "total": 0, "accuracy": 0.5}
            for name in self.agents.keys()
        }
        
        # Conflict resolution thresholds
        self.disagreement_threshold = 0.5  # High disagreement if score diff > 0.5
        self.min_confidence_for_trade = 0.65  # Minimum confidence to execute
        self.high_disagreement_penalty = 0.3  # Confidence penalty when agents disagree
        
        self.logger.info("Multi-agent system initialized with dynamic weighting")
    
    def update_agent_performance(self, agent_name: str, was_correct: bool) -> None:
        """Update agent performance and adjust weights."""
        perf = self.agent_performance[agent_name]
        perf["total"] += 1
        if was_correct:
            perf["correct"] += 1
        
        # Calculate accuracy with smoothing
        perf["accuracy"] = (perf["correct"] + 1) / (perf["total"] + 2)
        
        # Recalculate all weights based on relative accuracy
        self._recalculate_weights()
        
        self.logger.info(
            f"Agent {agent_name} performance updated: {perf['accuracy']:.2%} accuracy, "
            f"new weight: {self.agent_weights[agent_name]:.2f}"
        )
    
    def _recalculate_weights(self) -> None:
        """Recalculate agent weights based on performance."""
        # Get accuracies
        accuracies = {
            name: self.agent_performance[name]["accuracy"]
            for name in self.agents.keys()
        }
        
        # Normalize to sum to 1
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            for name in self.agents.keys():
                # Blend base weight with performance-based weight
                performance_weight = accuracies[name] / total_accuracy
                self.agent_weights[name] = (
                    0.3 * self.base_weights[name] +  # Keep 30% base weight
                    0.7 * performance_weight          # 70% performance-based
                )
        
        # Normalize final weights
        total = sum(self.agent_weights.values())
        if total > 0:
            for name in self.agent_weights:
                self.agent_weights[name] /= total
    
    def make_consensus_decision(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make consensus decision from all agents.
        
        Args:
            context: Market context data
            
        Returns:
            Consensus decision
        """
        try:
            # Get decisions from all agents
            agent_decisions = []
            
            for agent_name, agent in self.agents.items():
                decision = agent.analyze(context)
                if decision:
                    agent_decisions.append(decision)
                    self.logger.info(f"{agent_name}: {decision.action} {decision.symbol} (conf: {decision.confidence:.2f})")
            
            if not agent_decisions:
                return None
            
            # Build consensus
            consensus = self._build_consensus(agent_decisions, context)
            
            if consensus:
                self.logger.info(f"Consensus: {consensus['action']} {consensus['symbol']} (conf: {consensus['confidence']:.2f})")
            
            return consensus
            
        except Exception as e:
            self.logger.error(f"Consensus decision failed: {e}")
            return None
    
    def _build_consensus(self, agent_decisions: List[AgentDecision], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build consensus from agent decisions with conflict resolution."""
        # Count actions
        action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        weighted_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        
        self.logger.debug(f"Building consensus from {len(agent_decisions)} agent decisions")
        
        for decision in agent_decisions:
            action_counts[decision.action] += 1
            
            # Weight by agent weight and confidence
            agent_weight = self.agent_weights.get(decision.agent_name, 0.33)
            weighted_score = agent_weight * decision.confidence
            weighted_scores[decision.action] += weighted_score
            
            self.logger.debug(f"Agent {decision.agent_name}: {decision.action} (conf: {decision.confidence:.2f}, weight: {agent_weight:.2f}, weighted_score: {weighted_score:.2f})")
        
        self.logger.debug(f"Action counts: {action_counts}")
        self.logger.debug(f"Weighted scores: {weighted_scores}")
        
        # Determine consensus action
        max_score = max(weighted_scores.values())
        consensus_action = max(weighted_scores, key=weighted_scores.get)
        
        self.logger.debug(f"Max score: {max_score}, Consensus action: {consensus_action}")
        
        # CONFLICT RESOLUTION: Check for high disagreement
        sorted_scores = sorted(weighted_scores.values(), reverse=True)
        score_diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        
        # Check if agents disagree significantly
        is_high_disagreement = score_diff < self.disagreement_threshold
        
        # Calculate consensus confidence
        max_possible_weight = sum(self.agent_weights.values())
        consensus_confidence = weighted_scores[consensus_action] / max_possible_weight if max_possible_weight > 0 else 0.0
        
        # CONFLICT RESOLUTION: Apply penalty for high disagreement
        if is_high_disagreement:
            consensus_confidence *= (1 - self.high_disagreement_penalty)
            self.logger.warning(
                f"High agent disagreement detected (diff: {score_diff:.3f}). "
                f"Reducing confidence to {consensus_confidence:.3f}"
            )
        
        # CONFLICT RESOLUTION: If disagreement is very high, force HOLD
        if score_diff < 0.2 and consensus_confidence < 0.4:
            self.logger.warning(f"Severe disagreement - forcing HOLD")
            consensus_action = "HOLD"
            consensus_confidence = 0.3
        
        # Check if confidence is below minimum threshold for trading
        if consensus_confidence < self.min_confidence_for_trade and consensus_action != "HOLD":
            self.logger.warning(
                f"Confidence {consensus_confidence:.3f} below threshold {self.min_confidence_for_trade}. "
                f"Downgrading to HOLD"
            )
            consensus_action = "HOLD"
            consensus_confidence = min(consensus_confidence, 0.5)
        
        self.logger.debug(f"Probabilistic decision: {consensus_action} with confidence {consensus_confidence:.2f}")
        
        # Build reasoning
        reasoning_parts = []
        for decision in agent_decisions:
            reasoning_parts.append(f"{decision.agent_name}: {decision.reasoning}")
        
        combined_reasoning = " | ".join(reasoning_parts)
        
        # Create consensus decision
        consensus = {
            "action": consensus_action,
            "symbol": context.get("symbol", ""),
            "confidence": consensus_confidence,
            "consensus_score": max_score,
            "agent_decisions": [
                {
                    "agent": d.agent_name,
                    "action": d.action,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning
                }
                for d in agent_decisions
            ],
            "reasoning": combined_reasoning,
            "timestamp": datetime.now()
        }
        
        return consensus
