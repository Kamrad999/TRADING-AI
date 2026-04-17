"""
Market context builder for decision engine.
Following patterns from AgentQuant and ai-trade repositories.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..infrastructure.logging import get_logger


class MarketContext:
    """
    Market context builder for decision engine.
    
    Following patterns from:
    - AgentQuant: Market Context Analysis
    - ai-trade: Market Data Pipeline
    - ai-hedge-fund-crypto: Multi-timeframe data
    """
    
    def __init__(self):
        """Initialize market context builder."""
        self.logger = get_logger("market_context")
        
        # Technical indicator calculations
        self.indicator_periods = {
            "rsi": 14,
            "macd": 12,
            "sma_short": 20,
            "sma_long": 50,
            "ema_short": 12,
            "ema_long": 26
        }
        
        self.logger.info("Market context builder initialized")
    
    def build_context(self, symbol: str, market_data: Dict[str, Any], 
                      news_data: List[Dict[str, Any]], positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Build comprehensive market context.
        
        Args:
            symbol: Trading symbol
            market_data: Market data and indicators
            news_data: News articles and sentiment
            positions: Current positions
            
        Returns:
            Comprehensive market context
        """
        try:
            # Base context
            context = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": float(market_data.get("price", 0.0)),
                "volume": float(market_data.get("volume", 0.0)),
                "volatility": float(market_data.get("volatility", 0.0)),
                "market_trend": self._determine_market_trend(market_data),
                "positions": positions,
                "portfolio_value": float(market_data.get("portfolio_value", 0.0))
            }
            
            # Technical indicators
            context["technical_indicators"] = self._calculate_technical_indicators(market_data)
            
            # Market analysis
            context["market_analysis"] = self._analyze_market_conditions(market_data)
            
            # News sentiment
            context["news_sentiment"] = self._analyze_news_sentiment(news_data)
            
            # Risk metrics
            context["risk_metrics"] = self._calculate_risk_metrics(market_data, positions)
            
            # Position analysis
            context["position_analysis"] = self._analyze_positions(symbol, positions, market_data)
            
            # Summary fields for LLM
            context["news_summary"] = self._build_news_summary(news_data)
            context["sentiment_score"] = context["news_sentiment"].get("overall_sentiment", 0.0)
            context["news_count"] = len(news_data)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to build market context: {e}")
            return self._get_fallback_context(symbol)
    
    def _determine_market_trend(self, market_data: Dict[str, Any]) -> str:
        """Determine market trend from technical indicators."""
        try:
            indicators = market_data.get("indicators", {})
            
            if not indicators:
                return "neutral"
            
            # Check SMA trend
            sma_short = indicators.get("sma_20")
            sma_long = indicators.get("sma_50")
            current_price = market_data.get("price", 0.0)
            
            if sma_short and sma_long and current_price:
                if current_price > sma_short > sma_long:
                    return "bullish"
                elif current_price < sma_short < sma_long:
                    return "bearish"
                else:
                    return "neutral"
            
            # Check MACD
            macd = indicators.get("macd", 0.0)
            if macd > 0.1:
                return "bullish"
            elif macd < -0.1:
                return "bearish"
            
            return "neutral"
            
        except Exception as e:
            self.logger.error(f"Error determining market trend: {e}")
            return "neutral"
    
    def _calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            indicators = market_data.get("indicators", {})
            
            # RSI analysis
            rsi = indicators.get("rsi", 50.0)
            rsi_signal = "neutral"
            if rsi < 30:
                rsi_signal = "oversold"
            elif rsi > 70:
                rsi_signal = "overbought"
            
            # MACD analysis
            macd = indicators.get("macd", 0.0)
            macd_signal = "neutral"
            if macd > 0:
                macd_signal = "bullish"
            elif macd < 0:
                macd_signal = "bearish"
            
            # SMA analysis
            sma_short = indicators.get("sma_20", 0.0)
            sma_long = indicators.get("sma_50", 0.0)
            current_price = market_data.get("price", 0.0)
            
            sma_signal = "neutral"
            if sma_short and sma_long and current_price:
                if current_price > sma_short > sma_long:
                    sma_signal = "bullish"
                elif current_price < sma_short < sma_long:
                    sma_signal = "bearish"
            
            return {
                "rsi": rsi,
                "rsi_signal": rsi_signal,
                "macd": macd,
                "macd_signal": macd_signal,
                "sma_20": sma_short,
                "sma_50": sma_long,
                "sma_trend": sma_signal,
                "ema_12": indicators.get("ema_12", 0.0),
                "ema_26": indicators.get("ema_26", 0.0),
                "bollinger_upper": indicators.get("bollinger_upper", 0.0),
                "bollinger_lower": indicators.get("bollinger_lower", 0.0),
                "atr": indicators.get("atr", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions."""
        try:
            volatility = market_data.get("volatility", 0.0)
            volume = market_data.get("volume", 0.0)
            
            # Volatility analysis
            volatility_level = "low"
            if volatility > 0.05:
                volatility_level = "high"
            elif volatility > 0.03:
                volatility_level = "medium"
            
            # Volume analysis
            volume_level = "low"
            if volume > 1000000:
                volume_level = "high"
            elif volume > 500000:
                volume_level = "medium"
            
            # Market regime
            market_regime = "normal"
            if volatility > 0.08:
                market_regime = "volatile"
            elif volatility < 0.01:
                market_regime = "quiet"
            
            return {
                "volatility": volatility,
                "volatility_level": volatility_level,
                "volume": volume,
                "volume_level": volume_level,
                "market_regime": market_regime,
                "liquidity": self._assess_liquidity(market_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    def _analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze news sentiment."""
        try:
            if not news_data:
                return {
                    "overall_sentiment": 0.0,
                    "sentiment_label": "neutral",
                    "article_count": 0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
                }
            
            # Analyze sentiment distribution
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_sentiment = 0.0
            
            for article in news_data:
                sentiment = article.get("sentiment", 0.0)
                total_sentiment += sentiment
                
                if sentiment > 0.1:
                    positive_count += 1
                elif sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate overall sentiment
            overall_sentiment = total_sentiment / len(news_data) if news_data else 0.0
            
            # Determine sentiment label
            if overall_sentiment > 0.2:
                sentiment_label = "positive"
            elif overall_sentiment < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_label": sentiment_label,
                "article_count": len(news_data),
                "sentiment_distribution": {
                    "positive": positive_count,
                    "negative": negative_count,
                    "neutral": neutral_count
                },
                "sentiment_strength": abs(overall_sentiment)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {}
    
    def _calculate_risk_metrics(self, market_data: Dict[str, Any], positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate risk metrics."""
        try:
            volatility = market_data.get("volatility", 0.0)
            portfolio_value = market_data.get("portfolio_value", 0.0)
            
            # Position concentration
            total_position_value = 0.0
            for symbol, quantity in positions.items():
                if symbol in market_data:
                    price = market_data.get("price", 0.0)
                    total_position_value += abs(quantity) * price
            
            concentration = total_position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Risk score
            risk_score = 0.0
            risk_score += volatility * 0.4  # Volatility risk
            risk_score += concentration * 0.3  # Concentration risk
            risk_score += (1.0 - portfolio_value / 1000000) * 0.3  # Size risk (assuming $1M is optimal)
            
            return {
                "volatility": volatility,
                "concentration": concentration,
                "risk_score": min(1.0, risk_score),
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
                "max_position_size": 0.2 if volatility > 0.05 else 0.3,
                "stop_loss_pct": 0.05 if volatility > 0.05 else 0.03
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_positions(self, symbol: str, positions: Dict[str, float], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current positions."""
        try:
            current_position = positions.get(symbol, 0.0)
            current_price = market_data.get("price", 0.0)
            
            position_value = abs(current_position) * current_price
            portfolio_value = market_data.get("portfolio_value", 0.0)
            
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # Position status
            if current_position > 0:
                position_type = "long"
            elif current_position < 0:
                position_type = "short"
            else:
                position_type = "flat"
            
            # P&L estimation (simplified)
            unrealized_pnl = 0.0
            if current_position != 0:
                entry_price = market_data.get("entry_price", current_price)
                unrealized_pnl = (current_price - entry_price) * current_position
            
            return {
                "current_position": current_position,
                "position_type": position_type,
                "position_value": position_value,
                "position_pct": position_pct,
                "unrealized_pnl": unrealized_pnl,
                "can_add_position": position_pct < 0.2,
                "should_reduce_position": position_pct > 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing positions: {e}")
            return {}
    
    def _build_news_summary(self, news_data: List[Dict[str, Any]]) -> str:
        """Build news summary for LLM."""
        try:
            if not news_data:
                return "No recent news available."
            
            # Get top 3 most recent articles
            recent_news = sorted(news_data, key=lambda x: x.get("timestamp", ""), reverse=True)[:3]
            
            summary_parts = []
            for article in recent_news:
                title = article.get("title", "No title")
                sentiment = article.get("sentiment", 0.0)
                sentiment_label = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
                
                summary_parts.append(f"{title} ({sentiment_label})")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error building news summary: {e}")
            return "News summary unavailable."
    
    def _assess_liquidity(self, market_data: Dict[str, Any]) -> str:
        """Assess market liquidity."""
        try:
            volume = market_data.get("volume", 0.0)
            spread = market_data.get("spread", 0.0)
            
            if volume > 1000000 and spread < 0.01:
                return "high"
            elif volume > 500000 and spread < 0.02:
                return "medium"
            else:
                return "low"
            
        except Exception as e:
            self.logger.error(f"Error assessing liquidity: {e}")
            return "unknown"
    
    def _get_fallback_context(self, symbol: str) -> Dict[str, Any]:
        """Get fallback context when data is unavailable."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": 0.0,
            "volume": 0.0,
            "volatility": 0.0,
            "market_trend": "neutral",
            "positions": {},
            "portfolio_value": 0.0,
            "technical_indicators": {},
            "market_analysis": {},
            "news_sentiment": {"overall_sentiment": 0.0, "sentiment_label": "neutral"},
            "risk_metrics": {"risk_score": 0.5, "risk_level": "medium"},
            "position_analysis": {"current_position": 0.0, "position_type": "flat"},
            "news_summary": "No news available",
            "sentiment_score": 0.0,
            "news_count": 0
        }
