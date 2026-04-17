"""
Enhanced signal generator with multi-agent consensus and LLM reasoning.
Following patterns from ai-hedge-fund-crypto and AgentQuant repositories.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..agents.multi_agent_system import MultiAgentSystem
from ..market.data_provider import DataProvider
from ..brain.market_context import MarketContext
from ..infrastructure.logging import get_logger


@dataclass
class SignalFactors:
    """Signal factors for scoring."""
    news_sentiment: float
    technical_strength: float
    risk_adjustment: float
    consensus_confidence: float
    market_regime: float
    volatility_adjustment: float


class EnhancedSignalGenerator:
    """
    Enhanced signal generator with multi-agent consensus and LLM reasoning.
    
    Following patterns from:
    - ai-hedge-fund-crypto: Multi-agent consensus system
    - AgentQuant: Strategy planning with LLM
    - ai-trade: LLM Trading Brain with structured outputs
    """
    
    def __init__(self):
        """Initialize enhanced signal generator."""
        self.logger = get_logger("enhanced_signal_generator")
        
        # Components
        self.multi_agent_system = MultiAgentSystem()
        self.data_provider = DataProvider()
        self.market_context = MarketContext()
        
        # Signal configuration
        self.min_confidence_threshold = 0.6
        self.max_signals_per_symbol = 3
        self.signal_timeout = 300  # 5 minutes
        
        # Scoring weights
        self.scoring_weights = {
            "news_sentiment": 0.25,
            "technical_strength": 0.35,
            "risk_adjustment": 0.25,
            "consensus_confidence": 0.15
        }
        
        # Signal cache
        self.signal_cache: Dict[str, List[Signal]] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        
        self.logger.info("Enhanced signal generator initialized")
    
    def generate_signals(self, symbols: List[str], news_data: List[Dict[str, Any]], 
                        positions: Dict[str, float]) -> List[Signal]:
        """
        Generate trading signals using multi-agent consensus.
        
        Args:
            symbols: List of symbols to analyze
            news_data: News articles and sentiment
            positions: Current positions
            
        Returns:
            List of generated signals
        """
        try:
            signals = []
            
            for symbol in symbols:
                symbol_signals = self._generate_symbol_signals(symbol, news_data, positions)
                signals.extend(symbol_signals)
            
            # Sort by confidence and limit
            signals.sort(key=lambda s: s.confidence, reverse=True)
            
            # Remove duplicates and limit per symbol
            signals = self._deduplicate_signals(signals)
            
            self.logger.info(f"Generated {len(signals)} signals from {len(symbols)} symbols")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return []
    
    def _generate_symbol_signals(self, symbol: str, news_data: List[Dict[str, Any]], 
                                 positions: Dict[str, float]) -> List[Signal]:
        """Generate signals for a single symbol."""
        try:
            # Get market data
            market_data = self.data_provider.get_market_data(symbol, include_indicators=True)
            if not market_data:
                return []
            
            # Filter relevant news
            symbol_news = self._filter_symbol_news(symbol, news_data)
            
            # Build market context
            context = self.market_context.build_context(symbol, market_data, symbol_news, positions)
            
            # Get multi-agent consensus
            consensus = self.multi_agent_system.make_consensus_decision(context)
            
            if not consensus:
                return []
            
            # Convert to signal
            signal = self._convert_consensus_to_signal(consensus, context)
            
            if signal and self._validate_signal(signal, context):
                return [signal]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}")
            return []
    
    def _filter_symbol_news(self, symbol: str, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter news relevant to symbol."""
        relevant_news = []
        
        # Symbol keywords
        symbol_keywords = {
            "BTC": ["bitcoin", "btc", "bitcoin btc"],
            "ETH": ["ethereum", "eth", "ethereum eth"],
            "AAPL": ["apple", "aapl", "apple aapl"],
            "MSFT": ["microsoft", "msft", "microsoft msft"],
            "GOOGL": ["google", "googl", "google googl", "alphabet"],
            "AMZN": ["amazon", "amzn", "amazon amzn"],
            "TSLA": ["tesla", "tsla", "tesla tsla"],
            "META": ["meta", "facebook", "meta facebook"],
            "NVDA": ["nvidia", "nvda", "nvidia nvda"],
            "NFLX": ["netflix", "nflx", "netflix nflx"]
        }
        
        keywords = symbol_keywords.get(symbol, [symbol.lower()])
        
        for article in news_data:
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            
            # Check if article mentions symbol
            for keyword in keywords:
                if keyword in title or keyword in content:
                    relevant_news.append(article)
                    break
        
        return relevant_news
    
    def _convert_consensus_to_signal(self, consensus: Dict[str, Any], context: Dict[str, Any]) -> Signal:
        """Convert consensus decision to Signal object. ALWAYS produces a signal."""
        try:
            # ALWAYS produce a signal - even HOLD becomes neutral signal
            if consensus["action"] == "BUY":
                direction = SignalDirection.BUY
            elif consensus["action"] == "SELL":
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD  # HOLD produces neutral signal
            
            # Determine urgency
            confidence = consensus["confidence"]
            if confidence > 0.8:
                urgency = Urgency.HIGH
            elif confidence > 0.6:
                urgency = Urgency.MEDIUM
            else:
                urgency = Urgency.LOW
            
            # Determine market regime
            market_trend = context.get("market_trend", "neutral")
            if market_trend == "bullish":
                market_regime = MarketRegime.RISK_ON
            elif market_trend == "bearish":
                market_regime = MarketRegime.RISK_OFF
            else:
                market_regime = MarketRegime.SIDEWAYS
            
            # Calculate position size
            position_size = self._calculate_position_size(consensus, context)
            
            # Create signal
            signal = Signal(
                symbol=consensus["symbol"],
                direction=direction,
                confidence=confidence,
                urgency=urgency,
                market_regime=market_regime,
                position_size=position_size,
                execution_priority=1,
                signal_type=SignalType.NEWS,
                article_id=None,
                generated_at=consensus["timestamp"],
                metadata={
                    "enhanced_generator": True,
                    "consensus_score": consensus["consensus_score"],
                    "agent_decisions": consensus["agent_decisions"],
                    "reasoning": consensus["reasoning"],
                    "signal_factors": self._calculate_signal_factors(consensus, context),
                    "market_context": {
                        "price": context.get("current_price", 0.0),
                        "volume": context.get("volume", 0.0),
                        "volatility": context.get("volatility", 0.0),
                        "trend": context.get("market_trend", "neutral")
                    }
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to convert consensus to signal: {e}")
            # Always return a neutral signal on error
            return Signal(
                symbol=consensus.get("symbol", ""),
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
    
    def _calculate_signal_factors(self, consensus: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate signal factors for scoring."""
        factors = {}
        
        # News sentiment factor
        news_sentiment = context.get("sentiment_score", 0.0)
        factors["news_sentiment"] = abs(news_sentiment)
        
        # Technical strength factor
        indicators = context.get("technical_indicators", {})
        technical_strength = self._calculate_technical_strength(indicators)
        factors["technical_strength"] = technical_strength
        
        # Risk adjustment factor
        risk_metrics = context.get("risk_metrics", {})
        risk_score = risk_metrics.get("risk_score", 0.5)
        factors["risk_adjustment"] = 1.0 - risk_score
        
        # Consensus confidence factor
        factors["consensus_confidence"] = consensus["confidence"]
        
        # Market regime factor
        market_trend = context.get("market_trend", "neutral")
        if market_trend == "bullish":
            factors["market_regime"] = 0.8
        elif market_trend == "bearish":
            factors["market_regime"] = 0.3
        else:
            factors["market_regime"] = 0.5
        
        # Volatility adjustment factor
        volatility = context.get("volatility", 0.0)
        if volatility > 0.05:
            factors["volatility_adjustment"] = 0.5
        elif volatility > 0.03:
            factors["volatility_adjustment"] = 0.7
        else:
            factors["volatility_adjustment"] = 1.0
        
        return factors
    
    def _calculate_technical_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate technical indicator strength."""
        strength = 0.0
        count = 0
        
        # RSI strength
        rsi = indicators.get("rsi", 50.0)
        if rsi < 30:
            strength += 1.0
        elif rsi < 40:
            strength += 0.5
        elif rsi > 70:
            strength += 1.0
        elif rsi > 60:
            strength += 0.5
        count += 1
        
        # MACD strength
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)
        
        # Handle string values for macd_signal
        if isinstance(macd_signal, str):
            # Skip MACD strength calculation if signal is a string
            self.logger.debug(f"Skipping MACD strength calculation due to string signal: {macd_signal}")
        elif macd_signal is not None:
            if abs(macd - macd_signal) > 0.01:
                strength += 0.5
            count += 1
        
        # SMA strength
        current_price = indicators.get("current_price", 0.0)
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        
        if sma_20 and sma_50 and current_price:
            if current_price > sma_20 > sma_50:
                strength += 1.0
            elif current_price < sma_20 < sma_50:
                strength += 1.0
            count += 1
        
        return strength / count if count > 0 else 0.0
    
    def _calculate_position_size(self, consensus: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate position size based on consensus and risk."""
        base_size = 0.1  # 10% base position
        
        # Adjust by confidence
        confidence = consensus["confidence"]
        confidence_adjustment = confidence * 0.5  # 0-50% adjustment
        
        # Adjust by risk
        risk_metrics = context.get("risk_metrics", {})
        risk_score = risk_metrics.get("risk_score", 0.5)
        risk_adjustment = 1.0 - risk_score
        
        # Adjust by volatility
        volatility = context.get("volatility", 0.0)
        if volatility > 0.05:
            volatility_adjustment = 0.5
        else:
            volatility_adjustment = 1.0
        
        # Calculate final position size
        position_size = base_size * (1.0 + confidence_adjustment) * risk_adjustment * volatility_adjustment
        
        return max(0.05, min(0.3, position_size))
    
    def _validate_signal(self, signal: Signal, context: Dict[str, Any]) -> bool:
        """Validate signal before returning."""
        # Check confidence threshold
        if signal.confidence < self.min_confidence_threshold:
            return False
        
        # Check signal timeout
        last_time = self.last_signal_time.get(signal.symbol, datetime.min)
        if (datetime.now() - last_time).seconds < self.signal_timeout:
            return False
        
        # Check market conditions
        volatility = context.get("volatility", 0.0)
        if volatility > 0.1:  # Too volatile
            return False
        
        # Check position limits
        positions = context.get("positions", {})
        current_position = positions.get(signal.symbol, 0.0)
        
        if current_position > 0 and signal.direction == SignalDirection.BUY:
            return False  # Don't add to existing position
        elif current_position < 0 and signal.direction == SignalDirection.SELL:
            return False  # Don't add to existing position
        
        return True
    
    def _deduplicate_signals(self, signals: List[Signal]) -> List[Signal]:
        """Remove duplicate signals and limit per symbol."""
        seen_symbols = set()
        deduplicated = []
        
        for signal in signals:
            if signal.symbol not in seen_symbols:
                seen_symbols.add(signal.symbol)
                deduplicated.append(signal)
                self.last_signal_time[signal.symbol] = datetime.now()
        
        return deduplicated
    
    def get_signal_performance(self, symbol: str) -> Dict[str, Any]:
        """Get signal performance metrics for symbol."""
        cached_signals = self.signal_cache.get(symbol, [])
        
        if not cached_signals:
            return {
                "total_signals": 0,
                "avg_confidence": 0.0,
                "success_rate": 0.0,
                "last_signal": None
            }
        
        # Calculate metrics
        total_signals = len(cached_signals)
        avg_confidence = sum(s.confidence for s in cached_signals) / total_signals
        
        # In production, calculate actual success rate from trade results
        success_rate = 0.0  # Placeholder
        
        last_signal = cached_signals[-1] if cached_signals else None
        
        return {
            "total_signals": total_signals,
            "avg_confidence": avg_confidence,
            "success_rate": success_rate,
            "last_signal": last_signal
        }
    
    def cleanup(self) -> None:
        """Cleanup signal generator resources."""
        self.signal_cache.clear()
        self.last_signal_time.clear()
        self.data_provider.cleanup()
        self.logger.info("Enhanced signal generator cleaned up")
