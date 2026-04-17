"""
News-based trading strategy following Freqtrade patterns.
Analyzes news sentiment and generates trading signals based on news data.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy
from .strategy_interface import StrategyContext, StrategyOutput
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..infrastructure.logging import get_logger


class NewsStrategy(BaseStrategy):
    """
    News-based strategy that analyzes sentiment and news flow.
    
    Following patterns from:
    - Freqtrade: Strategy system and sentiment analysis
    - ai-trade: LLM-based news analysis
    - VectorBT: Signal generation pipeline
    """
    
    def __init__(self, **kwargs):
        """Initialize news strategy."""
        super().__init__("NewsStrategy", **kwargs)
        
        # News-specific configuration
        self.sentiment_threshold = kwargs.get("sentiment_threshold", 0.3)
        self.news_decay_hours = kwargs.get("news_decay_hours", 24)
        self.min_news_count = kwargs.get("min_news_count", 1)
        self.max_news_age_hours = kwargs.get("max_news_age_hours", 48)
        
        # Sentiment weights
        self.strong_sentiment_weight = kwargs.get("strong_sentiment_weight", 1.5)
        self.moderate_sentiment_weight = kwargs.get("moderate_sentiment_weight", 1.0)
        self.weak_sentiment_weight = kwargs.get("weak_sentiment_weight", 0.5)
        
        # Strategy-specific risk
        self.news_stop_loss = kwargs.get("news_stop_loss", 0.04)  # 4% stop loss for news trades
        self.news_take_profit = kwargs.get("news_take_profit", 0.08)  # 8% take profit
        
        self.logger.info(f"NewsStrategy initialized with sentiment_threshold={self.sentiment_threshold}")
    
    def execute(self, context: StrategyContext) -> StrategyOutput:
        """
        Execute news strategy.
        
        Args:
            context: Current market context
            
        Returns:
            Strategy output with signals
        """
        signals = []
        
        try:
            # Get news data
            news_data = context.metadata.get("news_data", [])
            if not news_data:
                self.logger.debug("No news data available")
                return StrategyOutput(signals=signals, metadata={"strategy": "NewsStrategy", "reason": "no_news"})
            
            # Process each symbol
            for symbol in context.symbols:
                symbol_news = self._get_symbol_news(symbol, news_data)
                if not symbol_news:
                    continue
                
                # Analyze sentiment
                sentiment_analysis = self._analyze_sentiment(symbol_news)
                if not sentiment_analysis["valid"]:
                    continue
                
                # Generate signal based on sentiment
                signal = self._generate_news_signal(symbol, sentiment_analysis, context)
                if signal and self.validate_signal(signal, context):
                    signals.append(signal)
            
            self.logger.info(f"NewsStrategy generated {len(signals)} signals")
            
            return StrategyOutput(
                signals=signals,
                metadata={
                    "strategy": "NewsStrategy",
                    "processed_symbols": len(context.symbols),
                    "news_count": len(news_data),
                    "sentiment_analysis": sentiment_analysis
                }
            )
            
        except Exception as e:
            self.logger.error(f"NewsStrategy execution failed: {e}")
            return StrategyOutput(signals=signals, metadata={"strategy": "NewsStrategy", "error": str(e)})
    
    def _get_symbol_news(self, symbol: str, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get relevant news for a specific symbol."""
        symbol_news = []
        current_time = datetime.now()
        
        for article in news_data:
            # Check if article mentions symbol
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()
            
            symbol_keywords = [symbol.lower(), symbol.replace("-", "").lower()]
            if symbol == "BTC":
                symbol_keywords.extend(["bitcoin", "btc"])
            elif symbol == "ETH":
                symbol_keywords.extend(["ethereum", "eth"])
            
            # Check if article mentions symbol
            mentions_symbol = any(keyword in title or keyword in content for keyword in symbol_keywords)
            
            if mentions_symbol:
                # Check article age
                article_time = article.get("timestamp", current_time)
                if isinstance(article_time, str):
                    try:
                        article_time = datetime.fromisoformat(article_time.replace('Z', '+00:00'))
                    except:
                        article_time = current_time
                
                age_hours = (current_time - article_time).total_seconds() / 3600
                
                if age_hours <= self.max_news_age_hours:
                    symbol_news.append(article)
        
        return symbol_news
    
    def _analyze_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not news_articles:
            return {"valid": False, "reason": "no_articles"}
        
        # Calculate weighted sentiment
        total_sentiment = 0.0
        total_weight = 0.0
        sentiment_scores = []
        
        current_time = datetime.now()
        
        for article in news_articles:
            sentiment = article.get("sentiment", 0.0)
            
            # Calculate time-based weight (newer news = higher weight)
            article_time = article.get("timestamp", current_time)
            if isinstance(article_time, str):
                try:
                    article_time = datetime.fromisoformat(article_time.replace('Z', '+00:00'))
                except:
                    article_time = current_time
            
            age_hours = (current_time - article_time).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (age_hours / self.news_decay_hours))
            
            # Calculate sentiment strength weight
            abs_sentiment = abs(sentiment)
            if abs_sentiment >= 0.7:
                strength_weight = self.strong_sentiment_weight
            elif abs_sentiment >= 0.4:
                strength_weight = self.moderate_sentiment_weight
            else:
                strength_weight = self.weak_sentiment_weight
            
            # Combined weight
            combined_weight = time_weight * strength_weight
            
            total_sentiment += sentiment * combined_weight
            total_weight += combined_weight
            sentiment_scores.append(sentiment)
        
        if total_weight == 0:
            return {"valid": False, "reason": "no_weight"}
        
        # Calculate final sentiment
        avg_sentiment = total_sentiment / total_weight
        
        # Calculate sentiment consistency
        if len(sentiment_scores) > 1:
            sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
            consistency = max(0.0, 1.0 - sentiment_variance)
        else:
            consistency = 1.0
        
        # Determine if sentiment is strong enough
        is_strong = abs(avg_sentiment) >= self.sentiment_threshold
        
        return {
            "valid": True,
            "sentiment": avg_sentiment,
            "strength": abs(avg_sentiment),
            "consistency": consistency,
            "article_count": len(news_articles),
            "is_strong": is_strong,
            "direction": "bullish" if avg_sentiment > 0 else "bearish" if avg_sentiment < 0 else "neutral"
        }
    
    def _generate_news_signal(self, symbol: str, sentiment_analysis: Dict[str, Any], 
                           context: StrategyContext) -> Optional[Signal]:
        """Generate trading signal based on sentiment analysis."""
        sentiment = sentiment_analysis["sentiment"]
        strength = sentiment_analysis["strength"]
        consistency = sentiment_analysis["consistency"]
        
        # Determine signal direction
        if sentiment > self.sentiment_threshold:
            direction = SignalDirection.BUY
            confidence = min(0.9, strength * consistency)
            urgency = Urgency.HIGH if strength > 0.7 else Urgency.MEDIUM
            reason = f"Strong bullish sentiment ({sentiment:.2f}) from {sentiment_analysis['article_count']} news articles"
        elif sentiment < -self.sentiment_threshold:
            direction = SignalDirection.SELL
            confidence = min(0.9, strength * consistency)
            urgency = Urgency.HIGH if strength > 0.7 else Urgency.MEDIUM
            reason = f"Strong bearish sentiment ({sentiment:.2f}) from {sentiment_analysis['article_count']} news articles"
        else:
            # Weak sentiment - generate HOLD signal
            direction = SignalDirection.HOLD
            confidence = 0.3
            urgency = Urgency.LOW
            reason = f"Weak neutral sentiment ({sentiment:.2f}) - no clear signal"
        
        # Adjust confidence based on market regime
        if context.market_regime == MarketRegime.VOLATILE:
            confidence *= 0.8  # Reduce confidence in volatile markets
        elif context.market_regime == MarketRegime.RISK_ON and direction == SignalDirection.BUY:
            confidence *= 1.1  # Boost confidence for buys in risk-on markets
        elif context.market_regime == MarketRegime.RISK_OFF and direction == SignalDirection.SELL:
            confidence *= 1.1  # Boost confidence for sells in risk-off markets
        
        # Create signal
        signal = self.create_signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            reason=reason,
            urgency=urgency,
            signal_type=SignalType.NEWS,
            sentiment_score=sentiment,
            article_count=sentiment_analysis["article_count"],
            sentiment_strength=strength,
            sentiment_consistency=consistency,
            strategy_specific={
                "stop_loss": self.news_stop_loss,
                "take_profit": self.news_take_profit
            }
        )
        
        return signal
    
    def get_risk_parameters(self, context: StrategyContext) -> Dict[str, Any]:
        """Get risk parameters specific to news strategy."""
        base_params = super().get_risk_parameters(context)
        
        # News-specific risk adjustments
        news_volatility = context.metadata.get("news_volatility", 0.02)
        
        # Adjust risk based on news flow
        if news_volatility > 0.05:  # High news volatility
            base_params["stop_loss"] *= 1.2  # Wider stops
            base_params["take_profit"] *= 0.8  # Tighter targets
        elif news_volatility < 0.01:  # Low news volatility
            base_params["stop_loss"] *= 0.8  # Tighter stops
            base_params["take_profit"] *= 1.2  # Wider targets
        
        return base_params
