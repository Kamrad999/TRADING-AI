"""
LLM-based sentiment and event analysis.
Replaces ALL keyword-based logic with contextual LLM reasoning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from ..infrastructure.logging import get_logger


@dataclass
class LLMSentimentResult:
    """Structured output from LLM sentiment analysis."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0-1.0
    event_type: str  # macro, earnings, crypto, regulation, geopolitical
    impact: str  # low, medium, high
    reasoning: str
    time_horizon: str  # short, medium, long
    sentiment_score: float  # -1.0 to +1.0
    key_factors: List[str]
    risk_factors: List[str]


class LLMSentimentAnalyzer:
    """
    LLM-based sentiment analyzer.
    
    NO KEYWORD COUNTING - uses contextual reasoning only.
    """
    
    def __init__(self, llm_client=None):
        """Initialize LLM sentiment analyzer."""
        self.logger = get_logger("llm_sentiment_analyzer")
        self.llm_client = llm_client
        
        # Prompt template for structured analysis
        self.analysis_prompt = """You are an expert financial analyst. Analyze the following market news/article and provide a structured assessment.

ARTICLE TEXT:
{article_text}

MARKET CONTEXT:
- Symbol: {symbol}
- Current Price: ${current_price}
- 24h Change: {price_change_pct}%
- Trend: {trend}
- Volatility: {volatility}

INSTRUCTIONS:
1. Read the article carefully and understand the context
2. Consider market conditions and price action
3. Analyze the potential impact on the specified symbol
4. Handle ambiguous or unclear information appropriately
5. If information is insufficient, indicate low confidence

Provide your analysis in this EXACT JSON format:
{{
    "symbol": "{symbol}",
    "action": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "event_type": "macro|earnings|crypto|regulation|geopolitical|technical|adoption",
    "impact": "low|medium|high",
    "reasoning": "Clear explanation of your analysis",
    "time_horizon": "short|medium|long",
    "sentiment_score": -1.0 to 1.0,
    "key_factors": ["factor1", "factor2", ...],
    "risk_factors": ["risk1", "risk2", ...]
}}

IMPORTANT:
- Be objective and consider both bullish and bearish aspects
- Confidence should reflect certainty level (0.3 for ambiguous, 0.9 for clear signals)
- If the article is unrelated to the symbol, set action to HOLD with low confidence
- Consider historical context and market sentiment
"""
        
        self.logger.info("LLMSentimentAnalyzer initialized (NO keyword logic)")
    
    def analyze_article(
        self,
        article_text: str,
        symbol: str,
        market_context: Dict[str, Any]
    ) -> Optional[LLMSentimentResult]:
        """
        Analyze article using LLM reasoning.
        
        Args:
            article_text: Full article text
            symbol: Trading symbol
            market_context: Market data (price, trend, volatility)
            
        Returns:
            Structured sentiment result or None
        """
        try:
            if not self.llm_client:
                self.logger.error("No LLM client available")
                return None
            
            # Build prompt with context
            prompt = self.analysis_prompt.format(
                article_text=article_text[:2000],  # Limit to avoid token issues
                symbol=symbol,
                current_price=market_context.get("current_price", 0),
                price_change_pct=market_context.get("price_change_24h", 0),
                trend=market_context.get("trend", "neutral"),
                volatility=market_context.get("volatility", "medium")
            )
            
            # Get LLM response
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,  # Lower for more consistent outputs
                max_tokens=1000
            )
            
            if not response:
                return None
            
            # Parse JSON response
            result = self._parse_llm_response(response)
            
            if result:
                self.logger.info(
                    f"LLM analysis: {symbol} → {result.action} "
                    f"(conf: {result.confidence:.2f}, impact: {result.impact})"
                )
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[LLMSentimentResult]:
        """Parse LLM JSON response into structured result."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            
            if not json_str:
                self.logger.error("No JSON found in LLM response")
                return None
            
            data = json.loads(json_str)
            
            # Validate required fields
            required = ["symbol", "action", "confidence", "event_type", 
                       "impact", "reasoning", "time_horizon"]
            
            for field in required:
                if field not in data:
                    self.logger.error(f"Missing field: {field}")
                    return None
            
            # Validate action
            action = data["action"].upper()
            if action not in ["BUY", "SELL", "HOLD"]:
                self.logger.warning(f"Invalid action: {action}, defaulting to HOLD")
                action = "HOLD"
            
            # Validate and clamp confidence
            confidence = float(data["confidence"])
            confidence = max(0.0, min(1.0, confidence))
            
            # Validate sentiment score
            sentiment = float(data.get("sentiment_score", 0))
            sentiment = max(-1.0, min(1.0, sentiment))
            
            return LLMSentimentResult(
                symbol=data["symbol"],
                action=action,
                confidence=confidence,
                event_type=data["event_type"].lower(),
                impact=data["impact"].lower(),
                reasoning=data["reasoning"],
                time_horizon=data["time_horizon"].lower(),
                sentiment_score=sentiment,
                key_factors=data.get("key_factors", []),
                risk_factors=data.get("risk_factors", [])
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Response parsing failed: {e}")
            return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response text."""
        try:
            # Find JSON between braces
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
            
            # If no braces, return None
            return None
            
        except Exception:
            return None
    
    def batch_analyze(
        self,
        articles: List[Dict[str, Any]],
        symbol: str,
        market_context: Dict[str, Any]
    ) -> List[LLMSentimentResult]:
        """Analyze multiple articles."""
        results = []
        
        for article in articles:
            text = article.get("title", "") + "\n" + article.get("content", "")
            
            result = self.analyze_article(text, symbol, market_context)
            if result:
                results.append(result)
        
        return results
