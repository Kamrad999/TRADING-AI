"""
Multi-Factor Signal Model following Qlib patterns.
Implements sophisticated factor-based signal generation with dynamic weights.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from ..infrastructure.logging import get_logger
from ..core.models import Signal, SignalDirection, Urgency, MarketRegime, SignalType
from ..events.event_classifier import EventClassification, EventType, ImpactLevel
from ..market.market_microstructure import MicrostructureSignals, LiquidityState
from ..portfolio.position import Position


class FactorCategory(Enum):
    """Factor categories following Qlib classification."""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    RISK = "risk"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"
    REVERSION = "reversion"


class SignalStrength(Enum):
    """Signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class Factor:
    """Individual factor in the multi-factor model."""
    name: str
    category: FactorCategory
    value: float
    weight: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorSignal:
    """Signal generated from factor analysis."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    factors: List[Factor]
    factor_score: float
    risk_adjusted_score: float
    expected_return: float
    risk_level: float
    timestamp: datetime
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiFactorModel:
    """
    Multi-factor signal model following Qlib patterns.
    
    Key features:
    - Factor-based signal generation
    - Dynamic weight optimization
    - Risk-adjusted scoring
    - Factor correlation analysis
    - Performance attribution
    - Adaptive factor selection
    """
    
    def __init__(self):
        """Initialize multi-factor model."""
        self.logger = get_logger("multi_factor_model")
        
        # Factor definitions
        self._initialize_factors()
        
        # Factor weights
        self.factor_weights: Dict[str, float] = {}
        self.category_weights: Dict[FactorCategory, float] = {}
        
        # Factor history
        self.factor_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.signal_history: List[FactorSignal] = []
        
        # Performance tracking
        self.factor_performance: Dict[str, Dict[str, float]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Model parameters
        self._initialize_model_parameters()
        
        # Adaptive learning
        self._initialize_adaptive_learning()
        
        self.logger.info("MultiFactorModel initialized with Qlib-style factor analysis")
    
    def _initialize_factors(self) -> None:
        """Initialize factor definitions."""
        self.factor_definitions = {
            # Technical factors
            "rsi": {
                "category": FactorCategory.TECHNICAL,
                "default_weight": 0.15,
                "normalization": "rsi_normalize",
                "description": "Relative Strength Index"
            },
            "macd": {
                "category": FactorCategory.TECHNICAL,
                "default_weight": 0.12,
                "normalization": "macd_normalize",
                "description": "MACD indicator"
            },
            "bollinger_position": {
                "category": FactorCategory.TECHNICAL,
                "default_weight": 0.10,
                "normalization": "bollinger_normalize",
                "description": "Bollinger Band position"
            },
            "volume_trend": {
                "category": FactorCategory.TECHNICAL,
                "default_weight": 0.08,
                "normalization": "volume_normalize",
                "description": "Volume trend indicator"
            },
            
            # Momentum factors
            "price_momentum": {
                "category": FactorCategory.MOMENTUM,
                "default_weight": 0.12,
                "normalization": "momentum_normalize",
                "description": "Price momentum over lookback period"
            },
            "earnings_momentum": {
                "category": FactorCategory.MOMENTUM,
                "default_weight": 0.08,
                "normalization": "earnings_normalize",
                "description": "Earnings momentum"
            },
            
            # Sentiment factors
            "news_sentiment": {
                "category": FactorCategory.SENTIMENT,
                "default_weight": 0.10,
                "normalization": "sentiment_normalize",
                "description": "News sentiment score"
            },
            "social_sentiment": {
                "category": FactorCategory.SENTIMENT,
                "default_weight": 0.06,
                "normalization": "sentiment_normalize",
                "description": "Social media sentiment"
            },
            
            # Macro factors
            "market_regime": {
                "category": FactorCategory.MACRO,
                "default_weight": 0.08,
                "normalization": "regime_normalize",
                "description": "Market regime indicator"
            },
            "volatility_regime": {
                "category": FactorCategory.MACRO,
                "default_weight": 0.06,
                "normalization": "volatility_normalize",
                "description": "Volatility regime"
            },
            
            # Risk factors
            "value_at_risk": {
                "category": FactorCategory.RISK,
                "default_weight": 0.08,
                "normalization": "risk_normalize",
                "description": "Value at Risk measure"
            },
            "max_drawdown": {
                "category": FactorCategory.RISK,
                "default_weight": 0.06,
                "normalization": "risk_normalize",
                "description": "Maximum drawdown risk"
            },
            
            # Liquidity factors
            "bid_ask_spread": {
                "category": FactorCategory.LIQUIDITY,
                "default_weight": 0.05,
                "normalization": "liquidity_normalize",
                "description": "Bid-ask spread"
            },
            "order_book_depth": {
                "category": FactorCategory.LIQUIDITY,
                "default_weight": 0.05,
                "normalization": "liquidity_normalize",
                "description": "Order book depth"
            }
        }
        
        # Initialize weights
        for factor_name, definition in self.factor_definitions.items():
            self.factor_weights[factor_name] = definition["default_weight"]
        
        # Initialize category weights
        category_totals = defaultdict(float)
        for factor_name, definition in self.factor_definitions.items():
            category_totals[definition["category"]] += definition["default_weight"]
        
        # Normalize category weights
        total_weight = sum(category_totals.values())
        for category, weight in category_totals.items():
            self.category_weights[category] = weight / total_weight
    
    def _initialize_model_parameters(self) -> None:
        """Initialize model parameters."""
        self.model_params = {
            # Signal generation
            "min_factor_count": 3,
            "max_factor_count": 8,
            "min_confidence": 0.6,
            "signal_threshold": 0.3,
            
            # Risk adjustment
            "risk_adjustment_factor": 0.8,
            "volatility_penalty": 0.1,
            "correlation_penalty": 0.05,
            
            # Factor selection
            "factor_selection_method": "top_n",
            "factor_correlation_threshold": 0.7,
            "factor_performance_threshold": 0.02,
            
            # Normalization
            "normalization_method": "z_score",
            "outlier_detection": True,
            "outlier_threshold": 3.0
        }
    
    def _initialize_adaptive_learning(self) -> None:
        """Initialize adaptive learning parameters."""
        self.adaptive_params = {
            "learning_rate": 0.01,
            "decay_rate": 0.99,
            "adaptation_frequency": timedelta(hours=6),
            "performance_window": timedelta(days=30),
            "min_samples_for_adaptation": 50
        }
        
        self.last_adaptation = datetime.min
        self.performance_history: deque = deque(maxlen=1000)
    
    def generate_signal(self, symbol: str, market_data: Dict[str, Any],
                       event_classifications: List[EventClassification] = None,
                       microstructure: Optional[MicrostructureSignals] = None,
                       positions: List[Position] = None) -> Optional[FactorSignal]:
        """
        Generate multi-factor signal for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Market data including technical indicators
            event_classifications: Event classifications
            microstructure: Market microstructure data
            positions: Current positions
            
        Returns:
            Factor signal or None
        """
        try:
            timestamp = datetime.now()
            
            # Extract factors
            factors = self._extract_factors(symbol, market_data, event_classifications, microstructure, positions)
            
            if len(factors) < self.model_params["min_factor_count"]:
                self.logger.warning(f"Insufficient factors for {symbol}: {len(factors)}")
                return None
            
            # Select best factors
            selected_factors = self._select_factors(factors)
            
            # Calculate factor scores
            factor_score = self._calculate_factor_score(selected_factors)
            
            # Risk adjustment
            risk_adjusted_score = self._apply_risk_adjustment(factor_score, selected_factors, microstructure)
            
            # Determine signal direction and strength
            direction, strength = self._determine_signal_characteristics(risk_adjusted_score)
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(selected_factors, risk_adjusted_score)
            
            # Calculate expected return and risk
            expected_return = self._calculate_expected_return(selected_factors, risk_adjusted_score)
            risk_level = self._calculate_risk_level(selected_factors, microstructure)
            
            # Generate reasoning
            reasoning = self._generate_signal_reasoning(selected_factors, direction, strength, confidence)
            
            # Create signal
            signal = FactorSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                factors=selected_factors,
                factor_score=factor_score,
                risk_adjusted_score=risk_adjusted_score,
                expected_return=expected_return,
                risk_level=risk_level,
                timestamp=timestamp,
                reasoning=reasoning,
                metadata={
                    "market_data_summary": self._summarize_market_data(market_data),
                    "event_count": len(event_classifications) if event_classifications else 0,
                    "microstructure_liquidity": microstructure.liquidity_state.value if microstructure else "unknown"
                }
            )
            
            # Store signal
            self.signal_history.append(signal)
            
            # Update factor performance
            self._update_factor_performance(selected_factors)
            
            # Trigger adaptation if needed
            self._trigger_adaptation()
            
            self.logger.info(f"Multi-factor signal generated: {symbol} {direction.value} | "
                          f"Strength: {strength.name} | Confidence: {confidence:.2f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to generate multi-factor signal for {symbol}: {e}")
            return None
    
    def _extract_factors(self, symbol: str, market_data: Dict[str, Any],
                        event_classifications: List[EventClassification],
                        microstructure: Optional[MicrostructureSignals],
                        positions: List[Position]) -> List[Factor]:
        """Extract all relevant factors."""
        factors = []
        timestamp = datetime.now()
        
        # Technical factors
        technical_factors = self._extract_technical_factors(symbol, market_data, timestamp)
        factors.extend(technical_factors)
        
        # Momentum factors
        momentum_factors = self._extract_momentum_factors(symbol, market_data, timestamp)
        factors.extend(momentum_factors)
        
        # Sentiment factors
        sentiment_factors = self._extract_sentiment_factors(symbol, event_classifications, timestamp)
        factors.extend(sentiment_factors)
        
        # Macro factors
        macro_factors = self._extract_macro_factors(symbol, market_data, microstructure, timestamp)
        factors.extend(macro_factors)
        
        # Risk factors
        risk_factors = self._extract_risk_factors(symbol, market_data, positions, timestamp)
        factors.extend(risk_factors)
        
        # Liquidity factors
        liquidity_factors = self._extract_liquidity_factors(symbol, microstructure, timestamp)
        factors.extend(liquidity_factors)
        
        return factors
    
    def _extract_technical_factors(self, symbol: str, market_data: Dict[str, Any], timestamp: datetime) -> List[Factor]:
        """Extract technical analysis factors."""
        factors = []
        
        try:
            # RSI factor
            rsi = market_data.get("rsi", 50)
            if rsi is not None:
                rsi_normalized = self._normalize_rsi(rsi)
                factors.append(Factor(
                    name="rsi",
                    category=FactorCategory.TECHNICAL,
                    value=rsi_normalized,
                    weight=self.factor_weights["rsi"],
                    confidence=0.8,
                    timestamp=timestamp,
                    metadata={"raw_rsi": rsi}
                ))
            
            # MACD factor
            macd = market_data.get("macd", 0)
            if macd is not None:
                macd_normalized = self._normalize_macd(macd)
                factors.append(Factor(
                    name="macd",
                    category=FactorCategory.TECHNICAL,
                    value=macd_normalized,
                    weight=self.factor_weights["macd"],
                    confidence=0.7,
                    timestamp=timestamp,
                    metadata={"raw_macd": macd}
                ))
            
            # Bollinger Band position
            bb_position = market_data.get("bb_position", 0.5)
            if bb_position is not None:
                bb_normalized = self._normalize_bollinger(bb_position)
                factors.append(Factor(
                    name="bollinger_position",
                    category=FactorCategory.TECHNICAL,
                    value=bb_normalized,
                    weight=self.factor_weights["bollinger_position"],
                    confidence=0.6,
                    timestamp=timestamp,
                    metadata={"raw_bb_position": bb_position}
                ))
            
            # Volume trend
            volume_trend = market_data.get("volume_trend", 0)
            if volume_trend is not None:
                volume_normalized = self._normalize_volume(volume_trend)
                factors.append(Factor(
                    name="volume_trend",
                    category=FactorCategory.TECHNICAL,
                    value=volume_normalized,
                    weight=self.factor_weights["volume_trend"],
                    confidence=0.5,
                    timestamp=timestamp,
                    metadata={"raw_volume_trend": volume_trend}
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract technical factors: {e}")
        
        return factors
    
    def _extract_momentum_factors(self, symbol: str, market_data: Dict[str, Any], timestamp: datetime) -> List[Factor]:
        """Extract momentum factors."""
        factors = []
        
        try:
            # Price momentum
            price_momentum = market_data.get("price_momentum", 0)
            if price_momentum is not None:
                momentum_normalized = self._normalize_momentum(price_momentum)
                factors.append(Factor(
                    name="price_momentum",
                    category=FactorCategory.MOMENTUM,
                    value=momentum_normalized,
                    weight=self.factor_weights["price_momentum"],
                    confidence=0.7,
                    timestamp=timestamp,
                    metadata={"raw_price_momentum": price_momentum}
                ))
            
            # Earnings momentum (if available)
            earnings_momentum = market_data.get("earnings_momentum", 0)
            if earnings_momentum is not None:
                earnings_normalized = self._normalize_earnings(earnings_momentum)
                factors.append(Factor(
                    name="earnings_momentum",
                    category=FactorCategory.MOMENTUM,
                    value=earnings_normalized,
                    weight=self.factor_weights["earnings_momentum"],
                    confidence=0.6,
                    timestamp=timestamp,
                    metadata={"raw_earnings_momentum": earnings_momentum}
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract momentum factors: {e}")
        
        return factors
    
    def _extract_sentiment_factors(self, symbol: str, event_classifications: List[EventClassification], timestamp: datetime) -> List[Factor]:
        """Extract sentiment factors."""
        factors = []
        
        try:
            # News sentiment from events
            if event_classifications:
                # Calculate average sentiment from events
                sentiments = []
                confidences = []
                
                for event in event_classifications:
                    if event.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
                        # Positive events contribute positively, negative negatively
                        if "positive" in event.reasoning.lower():
                            sentiments.append(1.0)
                        elif "negative" in event.reasoning.lower():
                            sentiments.append(-1.0)
                        else:
                            sentiments.append(0.0)
                        confidences.append(event.confidence)
                
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    avg_confidence = np.mean(confidences)
                    
                    sentiment_normalized = self._normalize_sentiment(avg_sentiment)
                    factors.append(Factor(
                        name="news_sentiment",
                        category=FactorCategory.SENTIMENT,
                        value=sentiment_normalized,
                        weight=self.factor_weights["news_sentiment"],
                        confidence=avg_confidence,
                        timestamp=timestamp,
                        metadata={
                            "raw_sentiment": avg_sentiment,
                            "event_count": len(event_classifications)
                        }
                    ))
            
            # Social sentiment (if available)
            # This would be implemented with actual social media data
            social_sentiment = 0.1  # Placeholder
            social_normalized = self._normalize_sentiment(social_sentiment)
            factors.append(Factor(
                name="social_sentiment",
                category=FactorCategory.SENTIMENT,
                value=social_normalized,
                weight=self.factor_weights["social_sentiment"],
                confidence=0.4,
                timestamp=timestamp,
                metadata={"raw_social_sentiment": social_sentiment}
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract sentiment factors: {e}")
        
        return factors
    
    def _extract_macro_factors(self, symbol: str, market_data: Dict[str, Any],
                             microstructure: Optional[MicrostructureSignals], timestamp: datetime) -> List[Factor]:
        """Extract macro factors."""
        factors = []
        
        try:
            # Market regime
            market_regime = market_data.get("market_regime", "neutral")
            regime_normalized = self._normalize_regime(market_regime)
            factors.append(Factor(
                name="market_regime",
                category=FactorCategory.MACRO,
                value=regime_normalized,
                weight=self.factor_weights["market_regime"],
                confidence=0.6,
                timestamp=timestamp,
                metadata={"raw_regime": market_regime}
            ))
            
            # Volatility regime
            volatility = market_data.get("volatility", 0.02)
            volatility_normalized = self._normalize_volatility(volatility)
            factors.append(Factor(
                name="volatility_regime",
                category=FactorCategory.MACRO,
                value=volatility_normalized,
                weight=self.factor_weights["volatility_regime"],
                confidence=0.7,
                timestamp=timestamp,
                metadata={"raw_volatility": volatility}
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract macro factors: {e}")
        
        return factors
    
    def _extract_risk_factors(self, symbol: str, market_data: Dict[str, Any], positions: List[Position], timestamp: datetime) -> List[Factor]:
        """Extract risk factors."""
        factors = []
        
        try:
            # Value at Risk (simplified)
            current_price = market_data.get("price", 100)
            volatility = market_data.get("volatility", 0.02)
            var = -current_price * volatility * 2.33  # 99% VaR
            var_normalized = self._normalize_risk(var)
            factors.append(Factor(
                name="value_at_risk",
                category=FactorCategory.RISK,
                value=var_normalized,
                weight=self.factor_weights["value_at_risk"],
                confidence=0.6,
                timestamp=timestamp,
                metadata={"raw_var": var}
            ))
            
            # Maximum drawdown from positions
            if positions:
                symbol_positions = [p for p in positions if p.symbol == symbol]
                if symbol_positions:
                    max_dd = max([p.max_drawdown for p in symbol_positions])
                    dd_normalized = self._normalize_risk(max_dd)
                    factors.append(Factor(
                        name="max_drawdown",
                        category=FactorCategory.RISK,
                        value=dd_normalized,
                        weight=self.factor_weights["max_drawdown"],
                        confidence=0.5,
                        timestamp=timestamp,
                        metadata={"raw_max_drawdown": max_dd}
                    ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract risk factors: {e}")
        
        return factors
    
    def _extract_liquidity_factors(self, symbol: str, microstructure: Optional[MicrostructureSignals], timestamp: datetime) -> List[Factor]:
        """Extract liquidity factors."""
        factors = []
        
        try:
            if microstructure:
                # Bid-ask spread
                spread = microstructure.metadata.get("order_book_spread", 0.001)
                spread_normalized = self._normalize_liquidity(spread)
                factors.append(Factor(
                    name="bid_ask_spread",
                    category=FactorCategory.LIQUIDITY,
                    value=spread_normalized,
                    weight=self.factor_weights["bid_ask_spread"],
                    confidence=0.7,
                    timestamp=timestamp,
                    metadata={"raw_spread": spread}
                ))
                
                # Order book depth
                volume_ratio = microstructure.metadata.get("volume_ratio", 1.0)
                depth_normalized = self._normalize_liquidity(volume_ratio)
                factors.append(Factor(
                    name="order_book_depth",
                    category=FactorCategory.LIQUIDITY,
                    value=depth_normalized,
                    weight=self.factor_weights["order_book_depth"],
                    confidence=0.6,
                    timestamp=timestamp,
                    metadata={"raw_volume_ratio": volume_ratio}
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to extract liquidity factors: {e}")
        
        return factors
    
    def _select_factors(self, factors: List[Factor]) -> List[Factor]:
        """Select best factors for signal generation."""
        try:
            # Sort factors by weighted score
            scored_factors = []
            for factor in factors:
                score = factor.value * factor.weight * factor.confidence
                scored_factors.append((factor, score))
            
            scored_factors.sort(key=lambda x: x[1], reverse=True)
            
            # Select top factors
            max_factors = min(self.model_params["max_factor_count"], len(scored_factors))
            selected_factors = [factor for factor, score in scored_factors[:max_factors]]
            
            # Filter out highly correlated factors
            filtered_factors = self._filter_correlated_factors(selected_factors)
            
            return filtered_factors
            
        except Exception as e:
            self.logger.error(f"Failed to select factors: {e}")
            return factors[:self.model_params["min_factor_count"]]
    
    def _filter_correlated_factors(self, factors: List[Factor]) -> List[Factor]:
        """Filter out highly correlated factors."""
        try:
            if len(factors) <= 3:
                return factors
            
            filtered = [factors[0]]  # Always keep the highest scoring factor
            
            for factor in factors[1:]:
                # Check correlation with already selected factors
                is_correlated = False
                
                for selected_factor in filtered:
                    correlation = self._calculate_factor_correlation(factor, selected_factor)
                    if correlation > self.model_params["factor_correlation_threshold"]:
                        is_correlated = True
                        break
                
                if not is_correlated:
                    filtered.append(factor)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Failed to filter correlated factors: {e}")
            return factors
    
    def _calculate_factor_correlation(self, factor1: Factor, factor2: Factor) -> float:
        """Calculate correlation between two factors."""
        try:
            # Simple correlation based on category and recent values
            if factor1.category == factor2.category:
                return 0.8  # High correlation for same category
            
            # Check historical correlation
            factor1_history = list(self.factor_history[factor1.name])
            factor2_history = list(self.factor_history[factor2.name])
            
            if len(factor1_history) > 10 and len(factor2_history) > 10:
                values1 = [f.value for f in factor1_history[-10:]]
                values2 = [f.value for f in factor2_history[-10:]]
                
                correlation = np.corrcoef(values1, values2)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            
            return 0.3  # Default low correlation
            
        except Exception as e:
            self.logger.error(f"Failed to calculate factor correlation: {e}")
            return 0.0
    
    def _calculate_factor_score(self, factors: List[Factor]) -> float:
        """Calculate weighted factor score."""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for factor in factors:
                weighted_score = factor.value * factor.weight * factor.confidence
                total_score += weighted_score
                total_weight += factor.weight * factor.confidence
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate factor score: {e}")
            return 0.0
    
    def _apply_risk_adjustment(self, factor_score: float, factors: List[Factor],
                             microstructure: Optional[MicrostructureSignals]) -> float:
        """Apply risk adjustment to factor score."""
        try:
            adjusted_score = factor_score
            
            # Volatility penalty
            volatility_factors = [f for f in factors if f.category == FactorCategory.RISK]
            if volatility_factors:
                avg_risk = np.mean([f.value for f in volatility_factors])
                volatility_penalty = avg_risk * self.model_params["volatility_penalty"]
                adjusted_score -= volatility_penalty
            
            # Liquidity adjustment
            if microstructure:
                liquidity_multiplier = 1.0
                if microstructure.liquidity_state == LiquidityState.VERY_LOW:
                    liquidity_multiplier = 0.8
                elif microstructure.liquidity_state == LiquidityState.LOW:
                    liquidity_multiplier = 0.9
                elif microstructure.liquidity_state == LiquidityState.HIGH:
                    liquidity_multiplier = 1.1
                
                adjusted_score *= liquidity_multiplier
            
            # Correlation penalty
            correlation_penalty = len(factors) * self.model_params["correlation_penalty"]
            adjusted_score -= correlation_penalty
            
            return adjusted_score * self.model_params["risk_adjustment_factor"]
            
        except Exception as e:
            self.logger.error(f"Failed to apply risk adjustment: {e}")
            return factor_score
    
    def _determine_signal_characteristics(self, adjusted_score: float) -> Tuple[SignalDirection, SignalStrength]:
        """Determine signal direction and strength from adjusted score."""
        try:
            # Determine direction
            if adjusted_score > self.model_params["signal_threshold"]:
                direction = SignalDirection.BUY
            elif adjusted_score < -self.model_params["signal_threshold"]:
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD
            
            # Determine strength
            abs_score = abs(adjusted_score)
            if abs_score > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif abs_score > 0.6:
                strength = SignalStrength.STRONG
            elif abs_score > 0.4:
                strength = SignalStrength.MODERATE
            elif abs_score > 0.2:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return direction, strength
            
        except Exception as e:
            self.logger.error(f"Failed to determine signal characteristics: {e}")
            return SignalDirection.HOLD, SignalStrength.WEAK
    
    def _calculate_signal_confidence(self, factors: List[Factor], adjusted_score: float) -> float:
        """Calculate signal confidence."""
        try:
            # Base confidence from factor confidences
            factor_confidences = [f.confidence for f in factors]
            base_confidence = np.mean(factor_confidences) if factor_confidences else 0.5
            
            # Adjust based on score magnitude
            score_magnitude = abs(adjusted_score)
            confidence_boost = score_magnitude * 0.2
            
            # Adjust based on number of factors
            factor_count_boost = min(0.2, len(factors) * 0.03)
            
            total_confidence = base_confidence + confidence_boost + factor_count_boost
            
            return min(1.0, max(0.0, total_confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate signal confidence: {e}")
            return 0.5
    
    def _calculate_expected_return(self, factors: List[Factor], adjusted_score: float) -> float:
        """Calculate expected return."""
        try:
            # Base return from score
            base_return = adjusted_score * 0.05  # 5% max return
            
            # Adjust based on factor performance
            performance_adjustment = 0.0
            for factor in factors:
                if factor.name in self.factor_performance:
                    perf = self.factor_performance[factor.name]
                    performance_adjustment += perf.get("avg_return", 0) * factor.weight
            
            return base_return + performance_adjustment
            
        except Exception as e:
            self.logger.error(f"Failed to calculate expected return: {e}")
            return 0.0
    
    def _calculate_risk_level(self, factors: List[Factor], microstructure: Optional[MicrostructureSignals]) -> float:
        """Calculate risk level."""
        try:
            risk_level = 0.5  # Base risk level
            
            # Risk from risk factors
            risk_factors = [f for f in factors if f.category == FactorCategory.RISK]
            if risk_factors:
                avg_risk = np.mean([abs(f.value) for f in risk_factors])
                risk_level += avg_risk * 0.3
            
            # Risk from microstructure
            if microstructure:
                if microstructure.liquidity_state == LiquidityState.VERY_LOW:
                    risk_level += 0.3
                elif microstructure.liquidity_state == LiquidityState.LOW:
                    risk_level += 0.2
                elif microstructure.market_pressure > 0.7:
                    risk_level += 0.1
            
            return min(1.0, max(0.0, risk_level))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate risk level: {e}")
            return 0.5
    
    def _generate_signal_reasoning(self, factors: List[Factor], direction: SignalDirection,
                                 strength: SignalStrength, confidence: float) -> str:
        """Generate reasoning for signal."""
        try:
            reasoning_parts = []
            
            # Top contributing factors
            top_factors = sorted(factors, key=lambda f: f.value * f.weight, reverse=True)[:3]
            
            reasoning_parts.append(f"Signal: {direction.value} ({strength.name})")
            reasoning_parts.append(f"Confidence: {confidence:.2f}")
            
            factor_descriptions = []
            for factor in top_factors:
                factor_descriptions.append(f"{factor.name}: {factor.value:.2f}")
            
            reasoning_parts.append(f"Key factors: {', '.join(factor_descriptions)}")
            
            return " | ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to generate signal reasoning: {e}")
            return f"Multi-factor signal: {direction.value}"
    
    def _summarize_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key market data."""
        return {
            "price": market_data.get("price", 0),
            "volume": market_data.get("volume", 0),
            "rsi": market_data.get("rsi", 50),
            "volatility": market_data.get("volatility", 0.02)
        }
    
    def _update_factor_performance(self, factors: List[Factor]) -> None:
        """Update factor performance tracking."""
        try:
            for factor in factors:
                if factor.name not in self.factor_performance:
                    self.factor_performance[factor.name] = {
                        "total_uses": 0,
                        "total_return": 0.0,
                        "avg_return": 0.0,
                        "success_rate": 0.0
                    }
                
                # Store factor in history
                self.factor_history[factor.name].append(factor)
                
                # Update performance metrics
                perf = self.factor_performance[factor.name]
                perf["total_uses"] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to update factor performance: {e}")
    
    def _trigger_adaptation(self) -> None:
        """Trigger adaptive learning if conditions are met."""
        try:
            # Check if enough time has passed
            if datetime.now() - self.last_adaptation < self.adaptive_params["adaptation_frequency"]:
                return
            
            # Check if enough samples
            total_samples = sum(len(history) for history in self.factor_history.values())
            if total_samples < self.adaptive_params["min_samples_for_adaptation"]:
                return
            
            # Perform adaptation
            self._adapt_factor_weights()
            self.last_adaptation = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to trigger adaptation: {e}")
    
    def _adapt_factor_weights(self) -> None:
        """Adapt factor weights based on performance."""
        try:
            for factor_name, performance in self.factor_performance.items():
                if performance["total_uses"] > 10:
                    # Calculate performance score
                    avg_return = performance["avg_return"]
                    success_rate = performance["success_rate"]
                    
                    performance_score = (avg_return * 0.7 + success_rate * 0.3)
                    
                    # Adapt weight
                    current_weight = self.factor_weights.get(factor_name, 0.1)
                    weight_adjustment = performance_score * self.adaptive_params["learning_rate"]
                    
                    new_weight = current_weight + weight_adjustment
                    new_weight = max(0.01, min(0.5, new_weight))  # Clamp between 1% and 50%
                    
                    self.factor_weights[factor_name] = new_weight
                    
                    self.logger.debug(f"Adapted weight for {factor_name}: {current_weight:.3f} -> {new_weight:.3f}")
            
            # Renormalize weights
            self._renormalize_weights()
            
        except Exception as e:
            self.logger.error(f"Failed to adapt factor weights: {e}")
    
    def _renormalize_weights(self) -> None:
        """Renormalize factor weights to sum to 1."""
        try:
            total_weight = sum(self.factor_weights.values())
            if total_weight > 0:
                for factor_name in self.factor_weights:
                    self.factor_weights[factor_name] /= total_weight
        except Exception as e:
            self.logger.error(f"Failed to renormalize weights: {e}")
    
    # Normalization functions
    def _normalize_rsi(self, rsi: float) -> float:
        """Normalize RSI to [-1, 1] range."""
        if rsi <= 30:
            return -1.0 + (rsi / 30)  # Oversold: -1 to 0
        elif rsi >= 70:
            return (rsi - 70) / 30  # Overbought: 0 to 1
        else:
            return 0.0  # Neutral
    
    def _normalize_macd(self, macd: float) -> float:
        """Normalize MACD to [-1, 1] range."""
        return np.tanh(macd * 10)  # Simple normalization
    
    def _normalize_bollinger(self, position: float) -> float:
        """Normalize Bollinger Band position to [-1, 1] range."""
        return (position - 0.5) * 2  # Center at 0.5, scale to [-1, 1]
    
    def _normalize_volume(self, volume_trend: float) -> float:
        """Normalize volume trend to [-1, 1] range."""
        return np.tanh(volume_trend)
    
    def _normalize_momentum(self, momentum: float) -> float:
        """Normalize momentum to [-1, 1] range."""
        return np.tanh(momentum * 5)
    
    def _normalize_earnings(self, earnings: float) -> float:
        """Normalize earnings momentum to [-1, 1] range."""
        return np.tanh(earnings * 3)
    
    def _normalize_sentiment(self, sentiment: float) -> float:
        """Normalize sentiment to [-1, 1] range."""
        return max(-1.0, min(1.0, sentiment))
    
    def _normalize_regime(self, regime: str) -> float:
        """Normalize market regime to [-1, 1] range."""
        regime_scores = {
            "risk_on": 0.8,
            "bullish": 0.6,
            "neutral": 0.0,
            "bearish": -0.6,
            "risk_off": -0.8
        }
        return regime_scores.get(regime, 0.0)
    
    def _normalize_volatility(self, volatility: float) -> float:
        """Normalize volatility to [-1, 1] range."""
        # Low volatility = positive, high volatility = negative
        if volatility < 0.01:
            return 0.5
        elif volatility < 0.02:
            return 0.0
        elif volatility < 0.05:
            return -0.5
        else:
            return -1.0
    
    def _normalize_risk(self, risk: float) -> float:
        """Normalize risk measure to [-1, 1] range."""
        return -np.tanh(abs(risk) * 2)  # Risk is always negative
    
    def _normalize_liquidity(self, liquidity: float) -> float:
        """Normalize liquidity measure to [-1, 1] range."""
        return np.tanh(liquidity - 1.0)  # Center around 1.0
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        try:
            return {
                "factor_count": len(self.factor_weights),
                "signal_history_size": len(self.signal_history),
                "factor_weights": self.factor_weights,
                "category_weights": {cat.value: weight for cat, weight in self.category_weights.items()},
                "factor_performance": self.factor_performance,
                "recent_signals": [
                    {
                        "symbol": s.symbol,
                        "direction": s.direction.value,
                        "strength": s.strength.name,
                        "confidence": s.confidence,
                        "factor_count": len(s.factors)
                    }
                    for s in self.signal_history[-10:]
                ]
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get model summary: {e}")
            return {"error": str(e)}
