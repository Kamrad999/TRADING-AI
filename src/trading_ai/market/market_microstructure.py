"""
Market Microstructure Analysis following institutional trading patterns.
Analyzes order book imbalance, volume spikes, funding rates, and liquidity conditions.
"""

from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict

from ..infrastructure.logging import get_logger
from ..core.models import MarketRegime


class LiquidityState(Enum):
    """Liquidity state of the market."""
    HIGH = "high"          # Deep order book, tight spreads
    MEDIUM = "medium"      # Normal liquidity
    LOW = "low"           # Thin order book, wide spreads
    VERY_LOW = "very_low" # Illiquid conditions


class VolumeProfile(Enum):
    """Volume profile classification."""
    NORMAL = "normal"      # Average volume
    ELEVATED = "elevated"  # Above average
    SPIKE = "spike"        # Volume spike
    ANOMALOUS = "anomalous" # Unusual volume pattern


class OrderBookImbalance(Enum):
    """Order book imbalance states."""
    BALANCED = "balanced"    # ~50/50 buy/sell
    BULLISH = "bullish"      # More buy pressure
    BEARISH = "bearish"      # More sell pressure
    EXTREME = "extreme"      # Very imbalanced


@dataclass
class OrderBookData:
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float
    bid_ask_ratio: float


@dataclass
class VolumeAnalysis:
    """Volume analysis results."""
    current_volume: float
    avg_volume_24h: float
    volume_ratio: float
    volume_profile: VolumeProfile
    volume_trend: str  # "increasing", "decreasing", "stable"
    unusual_patterns: List[str]


@dataclass
class FundingRateData:
    """Funding rate data for crypto markets."""
    symbol: str
    current_rate: float
    avg_rate_24h: float
    rate_trend: str
    next_funding_time: datetime
    annualized_rate: float
    market_sentiment: str  # "long", "short", "neutral"


@dataclass
class MicrostructureSignals:
    """Combined microstructure signals."""
    liquidity_state: LiquidityState
    volume_profile: VolumeProfile
    order_book_imbalance: OrderBookImbalance
    funding_signals: Optional[FundingRateData]
    market_pressure: float  # -1 to 1, negative = bearish pressure
    execution_quality: float  # 0 to 1, lower = harder execution
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketMicrostructure:
    """
    Market microstructure analysis following institutional trading patterns.
    
    Key features:
    - Real-time order book analysis
    - Volume pattern detection
    - Funding rate monitoring (crypto)
    - Liquidity assessment
    - Execution quality scoring
    - Market pressure indicators
    """
    
    def __init__(self):
        """Initialize market microstructure analyzer."""
        self.logger = get_logger("market_microstructure")
        
        # Data storage
        self.order_book_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24h of minute data
        self.funding_rate_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Analysis parameters
        self._initialize_analysis_parameters()
        
        # Pattern detection
        self._initialize_pattern_detectors()
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "liquidity_transitions": defaultdict(int),
            "volume_spike_detections": 0,
            "execution_quality_scores": deque(maxlen=1000)
        }
        
        self.logger.info("MarketMicrostructure initialized with institutional-grade analysis")
    
    def _initialize_analysis_parameters(self) -> None:
        """Initialize analysis parameters."""
        self.analysis_params = {
            # Liquidity thresholds
            "liquidity": {
                "spread_threshold": 0.001,  # 0.1% spread threshold
                "depth_threshold": 1000000,  # $1M depth threshold
                "volume_threshold": 10000000,  # $10M daily volume threshold
            },
            
            # Volume analysis
            "volume": {
                "spike_multiplier": 3.0,  # 3x average volume = spike
                "anomaly_threshold": 5.0,  # 5x average volume = anomaly
                "trend_periods": [5, 15, 60],  # Analysis periods in minutes
            },
            
            # Order book imbalance
            "order_book": {
                "imbalance_threshold": 0.6,  # 60% imbalance threshold
                "depth_levels": 10,  # Number of price levels to analyze
                "weight_factor": 0.8,  # Weight for near-the-money orders
            },
            
            # Funding rates
            "funding": {
                "extreme_threshold": 0.01,  # 1% annualized = extreme
                "sentiment_threshold": 0.005,  # 0.5% annualized = sentiment
                "prediction_window": 8,  # Hours to predict funding pressure
            }
        }
    
    def _initialize_pattern_detectors(self) -> None:
        """Initialize pattern detection algorithms."""
        self.pattern_detectors = {
            "volume_spike": self._detect_volume_spike,
            "liquidity_drain": self._detect_liquidity_drain,
            "order_book_pressure": self._detect_order_book_pressure,
            "funding_pressure": self._detect_funding_pressure,
            "microstructure_anomaly": self._detect_microstructure_anomaly
        }
    
    def analyze_microstructure(self, symbol: str, order_book: Dict[str, Any], 
                              volume_data: Dict[str, Any], 
                              funding_data: Optional[Dict[str, Any]] = None) -> MicrostructureSignals:
        """
        Analyze market microstructure for a symbol.
        
        Args:
            symbol: Trading symbol
            order_book: Order book data
            volume_data: Volume and price data
            funding_data: Funding rate data (crypto only)
            
        Returns:
            Complete microstructure signals
        """
        try:
            timestamp = datetime.now()
            
            # Analyze order book
            order_book_analysis = self._analyze_order_book(symbol, order_book, timestamp)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(symbol, volume_data, timestamp)
            
            # Assess liquidity
            liquidity_state = self._assess_liquidity(order_book_analysis, volume_analysis)
            
            # Detect order book imbalance
            imbalance = self._detect_order_book_imbalance(order_book_analysis)
            
            # Analyze funding rates (if available)
            funding_signals = None
            if funding_data:
                funding_signals = self._analyze_funding_rates(symbol, funding_data, timestamp)
            
            # Calculate market pressure
            market_pressure = self._calculate_market_pressure(
                order_book_analysis, volume_analysis, imbalance, funding_signals
            )
            
            # Assess execution quality
            execution_quality = self._assess_execution_quality(
                liquidity_state, order_book_analysis, volume_analysis
            )
            
            # Detect patterns
            patterns = self._detect_patterns(symbol, order_book_analysis, volume_analysis, funding_signals)
            
            # Create signals
            signals = MicrostructureSignals(
                liquidity_state=liquidity_state,
                volume_profile=volume_analysis.volume_profile,
                order_book_imbalance=imbalance,
                funding_signals=funding_signals,
                market_pressure=market_pressure,
                execution_quality=execution_quality,
                timestamp=timestamp,
                metadata={
                    "symbol": symbol,
                    "patterns": patterns,
                    "order_book_spread": order_book_analysis.spread,
                    "volume_ratio": volume_analysis.volume_ratio,
                    "bid_ask_ratio": order_book_analysis.bid_ask_ratio
                }
            )
            
            # Store analysis
            self._store_analysis(symbol, signals)
            
            # Update stats
            self._update_analysis_stats(signals)
            
            self.logger.info(f"Microstructure analyzed for {symbol}: {liquidity_state.value} | Pressure: {market_pressure:.2f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis failed for {symbol}: {e}")
            # Return default signals
            return MicrostructureSignals(
                liquidity_state=LiquidityState.MEDIUM,
                volume_profile=VolumeProfile.NORMAL,
                order_book_imbalance=OrderBookImbalance.BALANCED,
                funding_signals=None,
                market_pressure=0.0,
                execution_quality=0.5,
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _analyze_order_book(self, symbol: str, order_book: Dict[str, Any], timestamp: datetime) -> OrderBookData:
        """Analyze order book data."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            # Create dummy data
            return OrderBookData(
                symbol=symbol,
                timestamp=timestamp,
                bids=[],
                asks=[],
                spread=0.0,
                mid_price=0.0,
                total_bid_volume=0.0,
                total_ask_volume=0.0,
                bid_ask_ratio=1.0
            )
        
        # Calculate spread and mid price
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        spread = best_ask - best_bid if best_ask > best_bid else 0.0
        mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        
        # Calculate total volumes
        total_bid_volume = sum(quantity for _, quantity in bids)
        total_ask_volume = sum(quantity for _, quantity in asks)
        
        # Calculate bid/ask ratio
        bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0
        
        return OrderBookData(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid_price,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            bid_ask_ratio=bid_ask_ratio
        )
    
    def _analyze_volume(self, symbol: str, volume_data: Dict[str, Any], timestamp: datetime) -> VolumeAnalysis:
        """Analyze volume patterns."""
        current_volume = volume_data.get("volume", 0.0)
        current_price = volume_data.get("price", 0.0)
        
        # Get historical volume data
        volume_history = self.volume_history[symbol]
        
        # Add current volume to history
        volume_history.append((timestamp, current_volume))
        
        # Calculate average volume
        if volume_history:
            volumes = [v for _, v in volume_history]
            avg_volume_24h = np.mean(volumes) if volumes else current_volume
        else:
            avg_volume_24h = current_volume
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 1.0
        
        # Determine volume profile
        if volume_ratio > self.analysis_params["volume"]["anomaly_threshold"]:
            volume_profile = VolumeProfile.ANOMALOUS
        elif volume_ratio > self.analysis_params["volume"]["spike_multiplier"]:
            volume_profile = VolumeProfile.SPIKE
        elif volume_ratio > 1.5:
            volume_profile = VolumeProfile.ELEVATED
        else:
            volume_profile = VolumeProfile.NORMAL
        
        # Analyze volume trend
        volume_trend = self._analyze_volume_trend(volume_history)
        
        # Detect unusual patterns
        unusual_patterns = self._detect_volume_patterns(volume_history, volume_profile)
        
        return VolumeAnalysis(
            current_volume=current_volume,
            avg_volume_24h=avg_volume_24h,
            volume_ratio=volume_ratio,
            volume_profile=volume_profile,
            volume_trend=volume_trend,
            unusual_patterns=unusual_patterns
        )
    
    def _analyze_volume_trend(self, volume_history: deque) -> str:
        """Analyze volume trend over time."""
        if len(volume_history) < 10:
            return "stable"
        
        # Get recent volumes
        recent_volumes = [v for _, v in list(volume_history)[-10:]]
        older_volumes = [v for _, v in list(volume_history)[-30:-10]] if len(volume_history) > 20 else recent_volumes
        
        if not older_volumes:
            return "stable"
        
        recent_avg = np.mean(recent_volumes)
        older_avg = np.mean(older_volumes)
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_volume_patterns(self, volume_history: deque, volume_profile: VolumeProfile) -> List[str]:
        """Detect unusual volume patterns."""
        patterns = []
        
        if len(volume_history) < 20:
            return patterns
        
        volumes = [v for _, v in volume_history]
        
        # Detect sudden volume changes
        if len(volumes) >= 5:
            recent_avg = np.mean(volumes[-5:])
            previous_avg = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else recent_avg
            
            if recent_avg > previous_avg * 2.0:
                patterns.append("sudden_volume_increase")
            elif recent_avg < previous_avg * 0.5:
                patterns.append("sudden_volume_decrease")
        
        # Detect volume exhaustion
        if volume_profile == VolumeProfile.SPIKE:
            peak_volume = max(volumes[-20:])
            current_volume = volumes[-1]
            if current_volume < peak_volume * 0.3:
                patterns.append("volume_exhaustion")
        
        # Detect volume accumulation
        if len(volumes) >= 10:
            recent_trend = np.polyfit(range(10), volumes[-10:], 1)[0]
            if recent_trend > 0 and volume_profile == VolumeProfile.ELEVATED:
                patterns.append("volume_accumulation")
        
        return patterns
    
    def _assess_liquidity(self, order_book: OrderBookData, volume: VolumeAnalysis) -> LiquidityState:
        """Assess overall liquidity state."""
        liquidity_score = 0.0
        
        # Spread analysis
        if order_book.mid_price > 0:
            spread_pct = order_book.spread / order_book.mid_price
            if spread_pct < 0.0001:  # < 0.01%
                liquidity_score += 3
            elif spread_pct < 0.001:  # < 0.1%
                liquidity_score += 2
            elif spread_pct < 0.005:  # < 0.5%
                liquidity_score += 1
            else:
                liquidity_score -= 1
        
        # Depth analysis
        total_depth = order_book.total_bid_volume + order_book.total_ask_volume
        if total_depth > 10000000:  # > $10M
            liquidity_score += 3
        elif total_depth > 1000000:  # > $1M
            liquidity_score += 2
        elif total_depth > 100000:  # > $100K
            liquidity_score += 1
        else:
            liquidity_score -= 1
        
        # Volume analysis
        if volume.volume_profile == VolumeProfile.NORMAL:
            liquidity_score += 1
        elif volume.volume_profile == VolumeProfile.ELEVATED:
            liquidity_score += 2
        elif volume.volume_profile == VolumeProfile.SPIKE:
            liquidity_score += 1  # Spike can improve liquidity temporarily
        elif volume.volume_profile == VolumeProfile.ANOMALOUS:
            liquidity_score -= 2
        
        # Determine liquidity state
        if liquidity_score >= 5:
            return LiquidityState.HIGH
        elif liquidity_score >= 2:
            return LiquidityState.MEDIUM
        elif liquidity_score >= 0:
            return LiquidityState.LOW
        else:
            return LiquidityState.VERY_LOW
    
    def _detect_order_book_imbalance(self, order_book: OrderBookData) -> OrderBookImbalance:
        """Detect order book imbalance."""
        if order_book.total_bid_volume == 0 and order_book.total_ask_volume == 0:
            return OrderBookImbalance.BALANCED
        
        # Calculate weighted imbalance (near-the-money orders weighted more)
        weight_factor = self.analysis_params["order_book"]["weight_factor"]
        
        # Calculate weighted volumes
        weighted_bid_volume = 0.0
        weighted_ask_volume = 0.0
        
        for i, (price, quantity) in enumerate(order_book.bids[:10]):
            weight = weight_factor ** i
            weighted_bid_volume += quantity * weight
        
        for i, (price, quantity) in enumerate(order_book.asks[:10]):
            weight = weight_factor ** i
            weighted_ask_volume += quantity * weight
        
        # Calculate imbalance ratio
        total_weighted_volume = weighted_bid_volume + weighted_ask_volume
        if total_weighted_volume > 0:
            bid_ratio = weighted_bid_volume / total_weighted_volume
        else:
            bid_ratio = 0.5
        
        # Determine imbalance state
        threshold = self.analysis_params["order_book"]["imbalance_threshold"]
        
        if bid_ratio > 0.7:
            return OrderBookImbalance.BULLISH
        elif bid_ratio < 0.3:
            return OrderBookImbalance.BEARISH
        elif bid_ratio > 0.9 or bid_ratio < 0.1:
            return OrderBookImbalance.EXTREME
        else:
            return OrderBookImbalance.BALANCED
    
    def _analyze_funding_rates(self, symbol: str, funding_data: Dict[str, Any], timestamp: datetime) -> FundingRateData:
        """Analyze funding rates for crypto markets."""
        current_rate = funding_data.get("current_rate", 0.0)
        next_funding_time = funding_data.get("next_funding_time", timestamp + timedelta(hours=8))
        
        # Get historical funding rates
        funding_history = self.funding_rate_history[symbol]
        
        # Add current rate to history
        funding_history.append((timestamp, current_rate))
        
        # Calculate average rate
        if funding_history:
            rates = [r for _, r in funding_history]
            avg_rate_24h = np.mean(rates) if rates else current_rate
        else:
            avg_rate_24h = current_rate
        
        # Determine trend
        if len(funding_history) >= 10:
            recent_rates = [r for _, r in list(funding_history)[-10:]]
            older_rates = [r for _, r in list(funding_history)[-30:-10]] if len(funding_history) > 20 else recent_rates
            
            recent_avg = np.mean(recent_rates)
            older_avg = np.mean(older_rates)
            
            if recent_avg > older_avg * 1.1:
                rate_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                rate_trend = "decreasing"
            else:
                rate_trend = "stable"
        else:
            rate_trend = "stable"
        
        # Calculate annualized rate
        annualized_rate = current_rate * 3 * 365  # 8h funding, 3 times per day, 365 days
        
        # Determine market sentiment
        sentiment_threshold = self.analysis_params["funding"]["sentiment_threshold"]
        if current_rate > sentiment_threshold:
            market_sentiment = "long"  # Longs pay shorts -> bullish sentiment
        elif current_rate < -sentiment_threshold:
            market_sentiment = "short"  # Shorts pay longs -> bearish sentiment
        else:
            market_sentiment = "neutral"
        
        return FundingRateData(
            symbol=symbol,
            current_rate=current_rate,
            avg_rate_24h=avg_rate_24h,
            rate_trend=rate_trend,
            next_funding_time=next_funding_time,
            annualized_rate=annualized_rate,
            market_sentiment=market_sentiment
        )
    
    def _calculate_market_pressure(self, order_book: OrderBookData, volume: VolumeAnalysis,
                                imbalance: OrderBookImbalance, funding: Optional[FundingRateData]) -> float:
        """Calculate overall market pressure (-1 to 1)."""
        pressure = 0.0
        
        # Order book pressure
        if imbalance == OrderBookImbalance.BULLISH:
            pressure += 0.3
        elif imbalance == OrderBookImbalance.BEARISH:
            pressure -= 0.3
        elif imbalance == OrderBookImbalance.EXTREME:
            # Check direction of extreme imbalance
            if order_book.bid_ask_ratio > 2.0:
                pressure += 0.5
            elif order_book.bid_ask_ratio < 0.5:
                pressure -= 0.5
        
        # Volume pressure
        if volume.volume_profile == VolumeProfile.SPIKE:
            if volume.volume_trend == "increasing":
                pressure += 0.2
            else:
                pressure -= 0.2
        elif volume.volume_profile == VolumeProfile.ANOMALOUS:
            pressure -= 0.1  # Anomalous volume often indicates uncertainty
        
        # Funding pressure (crypto only)
        if funding:
            if funding.market_sentiment == "long":
                pressure += 0.1
            elif funding.market_sentiment == "short":
                pressure -= 0.1
            
            # Extreme funding rates
            if abs(funding.annualized_rate) > self.analysis_params["funding"]["extreme_threshold"]:
                pressure *= 1.5  # Amplify pressure for extreme funding
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, pressure))
    
    def _assess_execution_quality(self, liquidity: LiquidityState, order_book: OrderBookData, 
                                volume: VolumeAnalysis) -> float:
        """Assess execution quality (0 to 1, higher = better)."""
        quality_score = 0.5  # Base score
        
        # Liquidity impact
        liquidity_scores = {
            LiquidityState.HIGH: 0.3,
            LiquidityState.MEDIUM: 0.1,
            LiquidityState.LOW: -0.2,
            LiquidityState.VERY_LOW: -0.4
        }
        quality_score += liquidity_scores.get(liquidity, 0.0)
        
        # Spread impact
        if order_book.mid_price > 0:
            spread_pct = order_book.spread / order_book.mid_price
            if spread_pct < 0.0001:
                quality_score += 0.2
            elif spread_pct < 0.001:
                quality_score += 0.1
            elif spread_pct > 0.01:
                quality_score -= 0.2
        
        # Volume impact
        if volume.volume_profile == VolumeProfile.ELEVATED:
            quality_score += 0.1
        elif volume.volume_profile == VolumeProfile.SPIKE:
            quality_score += 0.05  # Spike can help execution
        elif volume.volume_profile == VolumeProfile.ANOMALOUS:
            quality_score -= 0.1
        
        # Order book depth impact
        total_depth = order_book.total_bid_volume + order_book.total_ask_volume
        if total_depth > 1000000:  # > $1M
            quality_score += 0.1
        elif total_depth < 100000:  # < $100K
            quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _detect_patterns(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                         funding: Optional[FundingRateData]) -> List[str]:
        """Detect microstructure patterns."""
        patterns = []
        
        for pattern_name, detector in self.pattern_detectors.items():
            try:
                if detector(symbol, order_book, volume, funding):
                    patterns.append(pattern_name)
            except Exception as e:
                self.logger.warning(f"Pattern detector {pattern_name} failed: {e}")
        
        return patterns
    
    def _detect_volume_spike(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                           funding: Optional[FundingRateData]) -> bool:
        """Detect volume spike pattern."""
        return volume.volume_profile == VolumeProfile.SPIKE
    
    def _detect_liquidity_drain(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                               funding: Optional[FundingRateData]) -> bool:
        """Detect liquidity drain pattern."""
        return (order_book.total_bid_volume + order_book.total_ask_volume) < 100000  # < $100K depth
    
    def _detect_order_book_pressure(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                                   funding: Optional[FundingRateData]) -> bool:
        """Detect order book pressure pattern."""
        return order_book.bid_ask_ratio > 2.0 or order_book.bid_ask_ratio < 0.5
    
    def _detect_funding_pressure(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                                funding: Optional[FundingRateData]) -> bool:
        """Detect funding pressure pattern."""
        if not funding:
            return False
        return abs(funding.annualized_rate) > self.analysis_params["funding"]["extreme_threshold"]
    
    def _detect_microstructure_anomaly(self, symbol: str, order_book: OrderBookData, volume: VolumeAnalysis,
                                     funding: Optional[FundingRateData]) -> bool:
        """Detect microstructure anomaly."""
        anomaly_score = 0
        
        # Anomaly indicators
        if volume.volume_profile == VolumeProfile.ANOMALOUS:
            anomaly_score += 1
        
        if order_book.bid_ask_ratio > 5.0 or order_book.bid_ask_ratio < 0.2:
            anomaly_score += 1
        
        if order_book.spread / order_book.mid_price > 0.01:  # > 1% spread
            anomaly_score += 1
        
        return anomaly_score >= 2
    
    def _store_analysis(self, symbol: str, signals: MicrostructureSignals) -> None:
        """Store analysis results."""
        # Store in history
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=1000)
        
        self.order_book_history[symbol].append(signals)
        
        # Store execution quality
        self.analysis_stats["execution_quality_scores"].append(signals.execution_quality)
    
    def _update_analysis_stats(self, signals: MicrostructureSignals) -> None:
        """Update analysis statistics."""
        self.analysis_stats["total_analyses"] += 1
        self.analysis_stats["liquidity_transitions"][signals.liquidity_state.value] += 1
        
        if signals.volume_profile == VolumeProfile.SPIKE:
            self.analysis_stats["volume_spike_detections"] += 1
    
    def get_microstructure_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get microstructure analysis summary."""
        summary = {
            "total_analyses": self.analysis_stats["total_analyses"],
            "volume_spike_detections": self.analysis_stats["volume_spike_detections"],
            "liquidity_distribution": dict(self.analysis_stats["liquidity_transitions"]),
            "avg_execution_quality": 0.0
        }
        
        if self.analysis_stats["execution_quality_scores"]:
            summary["avg_execution_quality"] = np.mean(list(self.analysis_stats["execution_quality_scores"]))
        
        # Symbol-specific summary
        if symbol and symbol in self.order_book_history:
            recent_signals = list(self.order_book_history[symbol])[-10:]
            summary["symbol_summary"] = {
                "symbol": symbol,
                "recent_analyses": len(recent_signals),
                "current_liquidity": recent_signals[-1].liquidity_state.value if recent_signals else "unknown",
                "avg_market_pressure": np.mean([s.market_pressure for s in recent_signals]) if recent_signals else 0.0,
                "volume_spike_count": len([s for s in recent_signals if s.volume_profile == VolumeProfile.SPIKE])
            }
        
        return summary
