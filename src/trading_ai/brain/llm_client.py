"""
LLM client for trading decisions.
Following patterns from ai-trade and AgentQuant repositories.
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..infrastructure.logging import get_logger
from ..infrastructure.config import config


@dataclass
class LLMDecision:
    """Structured LLM decision output."""
    action: str  # BUY | SELL | HOLD
    symbol: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime


class LLMClient:
    """
    LLM client for trading decisions.
    
    Following patterns from:
    - ai-trade: LLM Trading Brain with structured JSON outputs
    - AgentQuant: Gemini 2.5 Flash for strategy planning
    - ai-hedge-fund-crypto: Multi-agent consensus system
    """
    
    def __init__(self):
        """Initialize LLM client."""
        self.logger = get_logger("llm_client")
        
        # Configuration
        self.model_name = config.get("LLM_MODEL", "gpt-4")
        self.api_key = config.get("OPENAI_API_KEY", "")
        self.max_retries = config.get("LLM_MAX_RETRIES", 3)
        self.timeout = config.get("LLM_TIMEOUT", 30)
        
        # Decision templates
        self.decision_prompt = self._build_decision_prompt()
        
        self.logger.info(f"LLM client initialized with model: {self.model_name}")
    
    def make_trading_decision(self, market_context: Dict[str, Any]) -> Optional[LLMDecision]:
        """
        Make trading decision using LLM.
        
        Args:
            market_context: Market data and indicators
            
        Returns:
            Structured trading decision
        """
        try:
            # Build prompt
            prompt = self._build_prompt(market_context)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            decision = self._parse_response(response, market_context)
            
            if decision:
                self.logger.info(f"LLM decision: {decision.action} {decision.symbol} (conf: {decision.confidence:.2f})")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")
            return None
    
    def _build_prompt(self, market_context: Dict[str, Any]) -> str:
        """Build structured prompt for LLM."""
        return f"""
You are an expert cryptocurrency trader. Analyze the following market data and make a trading decision.

MARKET DATA:
{json.dumps(market_context, indent=2)}

ANALYSIS REQUIREMENTS:
1. Evaluate technical indicators (RSI, MACD, SMA trends)
2. Consider market sentiment from news
3. Assess current positions and risk
4. Determine optimal entry/exit points

DECISION FORMAT (STRICT JSON):
{{
    "action": "BUY|SELL|HOLD",
    "symbol": "{market_context.get('symbol', 'BTC')}",
    "confidence": 0.0-1.0,
    "entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "reasoning": "Detailed explanation of your decision"
}}

RULES:
- Confidence must be > 0.7 for BUY/SELL actions
- Stop loss should be 2-5% from entry
- Take profit should be 3-8% from entry
- Reasoning must be specific and data-driven
- If uncertain, choose HOLD with low confidence

RESPOND WITH ONLY JSON:
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        # Mock implementation - replace with actual LLM call
        # Following ai-trade pattern with fallback
        
        for attempt in range(self.max_retries):
            try:
                # Simulate LLM call
                time.sleep(0.1)  # Simulate API delay
                
                # Mock response for testing
                mock_response = self._generate_mock_response(prompt)
                return mock_response
                
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        raise Exception("LLM call failed after all retries")
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock LLM response for testing."""
        # Extract symbol from prompt
        symbol = "BTC"
        if "symbol" in prompt:
            import re
            symbol_match = re.search(r'"symbol":\s*"([^"]+)"', prompt)
            if symbol_match:
                symbol = symbol_match.group(1)
        
        # Generate realistic mock decision
        import random
        
        if random.random() > 0.3:  # 70% chance of trading signal
            action = random.choice(["BUY", "SELL"])
            confidence = random.uniform(0.7, 0.95)
            entry = random.uniform(45000, 55000) if symbol == "BTC" else random.uniform(2000, 4000)
            
            if action == "BUY":
                stop_loss = entry * (1 - random.uniform(0.02, 0.05))
                take_profit = entry * (1 + random.uniform(0.03, 0.08))
            else:
                stop_loss = entry * (1 + random.uniform(0.02, 0.05))
                take_profit = entry * (1 - random.uniform(0.03, 0.08))
            
            reasoning = f"Technical indicators show {'bullish' if action == 'BUY' else 'bearish'} momentum with RSI at {random.uniform(30, 70):.1f} and MACD confirming the trend."
        else:
            action = "HOLD"
            confidence = random.uniform(0.3, 0.6)
            entry = random.uniform(45000, 55000) if symbol == "BTC" else random.uniform(2000, 4000)
            stop_loss = entry * 0.95
            take_profit = entry * 1.05
            reasoning = "Market conditions are uncertain, waiting for clearer signals before taking position."
        
        return json.dumps({
            "action": action,
            "symbol": symbol,
            "confidence": confidence,
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": reasoning
        })
    
    def _parse_response(self, response: str, market_context: Dict[str, Any]) -> Optional[LLMDecision]:
        """Parse LLM response into structured decision."""
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ["action", "symbol", "confidence", "entry", "stop_loss", "take_profit", "reasoning"]
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field in LLM response: {field}")
                    return None
            
            # Validate action
            if data["action"] not in ["BUY", "SELL", "HOLD"]:
                self.logger.error(f"Invalid action in LLM response: {data['action']}")
                return None
            
            # Validate confidence
            if not (0.0 <= data["confidence"] <= 1.0):
                self.logger.error(f"Invalid confidence in LLM response: {data['confidence']}")
                return None
            
            # Validate prices
            if data["entry"] <= 0 or data["stop_loss"] <= 0 or data["take_profit"] <= 0:
                self.logger.error("Invalid prices in LLM response")
                return None
            
            # Create decision
            decision = LLMDecision(
                action=data["action"],
                symbol=data["symbol"],
                confidence=data["confidence"],
                entry=data["entry"],
                stop_loss=data["stop_loss"],
                take_profit=data["take_profit"],
                reasoning=data["reasoning"],
                timestamp=datetime.now()
            )
            
            return decision
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None
    
    def _build_decision_prompt(self) -> str:
        """Build base decision prompt template."""
        return """
You are an expert cryptocurrency trader with deep understanding of technical analysis, market sentiment, and risk management.

Your task is to analyze market data and provide structured trading decisions.

KEY PRINCIPLES:
1. Risk Management: Always use stop-losses (2-5%)
2. Profit Targets: Take profit at 3-8% gains
3. Confidence: Only trade with >70% confidence
4. Analysis: Consider multiple indicators and timeframes
5. Reasoning: Provide clear, data-driven explanations

ANALYSIS FRAMEWORK:
- Technical Indicators: RSI, MACD, Moving Averages
- Market Context: Volatility, Trend, Volume
- Risk Assessment: Position size, stop-loss, take-profit
- Sentiment Analysis: News sentiment, market psychology

DECISION CRITERIA:
- BUY: Bullish indicators, uptrend, positive sentiment
- SELL: Bearish indicators, downtrend, negative sentiment  
- HOLD: Uncertain conditions, mixed signals, high volatility

Always respond with structured JSON following the exact format provided.
"""
