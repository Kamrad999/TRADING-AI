"""
Production-grade Backtesting Engine for TRADING-AI system.
Following VectorBT patterns for signal feed, trade simulation, and performance metrics.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..portfolio.position_manager import PositionManager, PositionRequest
from ..strategies.news_strategy import NewsStrategy
from ..strategies.technical_strategy import TechnicalStrategy
from ..strategies.hybrid_strategy import HybridStrategy
from ..strategies.strategy_interface import StrategyContext, StrategyOutput
from ..core.models import Signal, SignalDirection, MarketRegime, MarketSession
from ..market.data_provider import DataProvider
from ..infrastructure.logging import get_logger


@dataclass
class BacktestConfig:
    """Configuration for backtest following VectorBT patterns."""
    initial_cash: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    start_date: datetime = None
    end_date: datetime = None
    symbols: List[str] = None
    timeframe: str = "1h"
    
    # Strategy configuration
    strategies: List[str] = None
    
    # Risk management
    max_position_size: float = 0.2
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    
    # Performance tracking
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.02


@dataclass
class BacktestResult:
    """Results from backtest following VectorBT output format."""
    # Core data (no defaults)
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    positions: pd.DataFrame
    signals: pd.DataFrame
    
    # Summary statistics (no defaults)
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    
    # Trade statistics (no defaults)
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Optional fields with defaults (must come last)
    benchmark: Optional[pd.DataFrame] = None


class BacktestEngine:
    """
    Production-grade backtesting engine following VectorBT patterns.
    
    Key features:
    - Signal feed processing
    - Realistic trade simulation
    - Comprehensive performance metrics
    - Portfolio management
    - Benchmark comparison
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine."""
        self.logger = get_logger("backtest_engine")
        self.config = config
        
        # Initialize components
        self.position_manager = PositionManager(config.initial_cash)
        self.data_provider = DataProvider()
        
        # Initialize strategies
        self.strategies = self._initialize_strategies(config.strategies or ["HybridStrategy"])
        
        # Data storage
        self.price_data = {}
        self.signals = []
        self.equity_curve = []
        self.benchmark_data = None
        
        # Performance tracking
        self.trade_log = []
        self.position_log = []
        self.signal_log = []
        
        self.logger.info(f"Backtest engine initialized for {config.symbols}")
    
    def _initialize_strategies(self, strategy_names: List[str]) -> Dict[str, Any]:
        """Initialize trading strategies."""
        strategies = {}
        
        for strategy_name in strategy_names:
            if strategy_name == "NewsStrategy":
                strategies[strategy_name] = NewsStrategy(
                    min_confidence=0.1,
                    max_position_size=self.config.max_position_size,
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct
                )
            elif strategy_name == "TechnicalStrategy":
                strategies[strategy_name] = TechnicalStrategy(
                    min_confidence=0.1,
                    max_position_size=self.config.max_position_size,
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct
                )
            elif strategy_name == "HybridStrategy":
                strategies[strategy_name] = HybridStrategy(
                    min_confidence=0.1,
                    max_position_size=self.config.max_position_size,
                    stop_loss_pct=self.config.stop_loss_pct,
                    take_profit_pct=self.config.take_profit_pct
                )
            else:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
        
        return strategies
    
    def run_backtest(self) -> BacktestResult:
        """
        Run complete backtest following VectorBT patterns.
        
        Returns:
            BacktestResult with comprehensive metrics
        """
        try:
            self.logger.info(f"Starting backtest for {self.config.symbols}")
            
            # Load historical data
            self._load_historical_data()
            
            # Generate signals
            self._generate_signals()
            
            # Simulate trades
            self._simulate_trades()
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            
            # Create result object
            result = BacktestResult(
                equity_curve=self._create_equity_curve(),
                trades=self._create_trades_dataframe(),
                performance_metrics=metrics,
                positions=self._create_positions_dataframe(),
                signals=self._create_signals_dataframe(),
                benchmark=self.benchmark_data,
                **metrics
            )
            
            self.logger.info(f"Backtest completed: {metrics['total_return']:.2%} return")
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _load_historical_data(self) -> None:
        """Load historical price data for all symbols."""
        self.logger.info("Loading historical data...")
        
        for symbol in self.config.symbols:
            # Get OHLC data
            ohlc_data = self.data_provider.fetch_ohlc_data(
                symbol, 
                self.config.timeframe,
                self.config.start_date,
                self.config.end_date
            )
            
            if ohlc_data:
                # Convert to pandas DataFrame
                df = pd.DataFrame(ohlc_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                self.price_data[symbol] = df
                self.logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                self.logger.error(f"No data loaded for {symbol}")
        
        # Load benchmark data if specified
        if self.config.benchmark:
            self._load_benchmark_data()
    
    def _load_benchmark_data(self) -> None:
        """Load benchmark data for comparison."""
        try:
            benchmark_data = self.data_provider.fetch_ohlc_data(
                self.config.benchmark,
                self.config.timeframe,
                self.config.start_date,
                self.config.end_date
            )
            
            if benchmark_data:
                df = pd.DataFrame(benchmark_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate benchmark returns
                df['returns'] = df['close_price'].pct_change()
                df['cumulative'] = (1 + df['returns']).cumprod()
                
                self.benchmark_data = df
                self.logger.info(f"Loaded benchmark data for {self.config.benchmark}")
        except Exception as e:
            self.logger.warning(f"Failed to load benchmark data: {e}")
    
    def _generate_signals(self) -> None:
        """Generate trading signals using strategies."""
        self.logger.info("Generating trading signals...")
        
        # Get all timestamps from price data
        all_timestamps = set()
        for symbol_data in self.price_data.values():
            all_timestamps.update(symbol_data.index)
        
        timestamps = sorted(all_timestamps)
        
        for timestamp in timestamps:
            # Create strategy context
            context = self._create_strategy_context(timestamp)
            
            # Generate signals from each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    output = strategy.execute(context)
                    
                    # Log signals
                    for signal in output.signals:
                        # Get symbol-specific price from market_data
                        symbol_market_data = context.market_data.get(signal.symbol, {})
                        symbol_price = symbol_market_data.get('price', 0.0)
                        
                        signal_dict = {
                            'timestamp': timestamp,
                            'strategy': strategy_name,
                            'symbol': signal.symbol,
                            'direction': signal.direction.value,
                            'confidence': signal.confidence,
                            'price': symbol_price,
                            'metadata': signal.metadata
                        }
                        self.signals.append(signal_dict)
                        self.signal_log.append(signal_dict)
                        
                except Exception as e:
                    self.logger.error(f"Signal generation failed for {strategy_name}: {e}")
        
        self.logger.info(f"Generated {len(self.signals)} signals")
    
    def _create_strategy_context(self, timestamp: datetime) -> StrategyContext:
        """Create strategy context for a specific timestamp."""
        # Get current market data
        market_data = {}
        current_prices = {}
        
        for symbol, df in self.price_data.items():
            if timestamp in df.index:
                symbol_data = df.loc[timestamp]
                
                # Calculate technical indicators from price history
                indicators = self._calculate_indicators(df, timestamp)
                
                market_data[symbol] = {
                    'price': symbol_data['close_price'],
                    'volume': symbol_data['volume'],
                    'high': symbol_data['high_price'],
                    'low': symbol_data['low_price'],
                    'open': symbol_data['open_price'],
                    'indicators': indicators
                }
                current_prices[symbol] = symbol_data['close_price']
        
        # Get current positions
        positions = {}
        for position in self.position_manager.get_open_positions():
            positions[position.symbol] = position.quantity
        
        # Create context (match StrategyContext dataclass fields)
        context = StrategyContext(
            current_time=timestamp,
            market_session=MarketSession.REGULAR,
            market_regime=MarketRegime.SIDEWAYS,  # Could be calculated
            portfolio_value=self.position_manager.current_balance,
            available_cash=self.position_manager.current_balance,
            positions=positions,
            market_data=market_data,
            news_data=[],  # Could be loaded from external source
            metadata={
                'symbols': self.config.symbols,
                'current_price': current_prices.get(self.config.symbols[0], 0.0) if self.config.symbols else 0.0,
                'volatility': 0.02,  # Could be calculated
            }
        )
        
        return context
    
    def _calculate_indicators(self, df: pd.DataFrame, timestamp: datetime) -> Dict[str, float]:
        """Calculate technical indicators from price history."""
        try:
            # Get data up to current timestamp
            hist = df.loc[:timestamp]
            if len(hist) < 20:
                return {}  # Not enough data
            
            closes = hist['close_price'].values
            current_price = closes[-1]
            
            # Simple Moving Averages
            sma_20 = closes[-20:].mean() if len(closes) >= 20 else current_price
            sma_50 = closes[-50:].mean() if len(closes) >= 50 else sma_20
            
            # RSI (simplified)
            deltas = np.diff(closes[-15:])
            if len(deltas) > 0:
                gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
                losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.001
                rs = gains / losses if losses > 0 else 100
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50.0
            
            # MACD (simplified)
            ema_12 = pd.Series(closes).ewm(span=12).mean().iloc[-1] if len(closes) >= 12 else current_price
            ema_26 = pd.Series(closes).ewm(span=26).mean().iloc[-1] if len(closes) >= 26 else current_price
            macd = ema_12 - ema_26
            macd_signal = macd * 0.9  # Simplified signal line
            
            return {
                'rsi': float(rsi),
                'macd': float(macd),
                'macd_signal': float(macd_signal),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'current_price': float(current_price)
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate indicators: {e}")
            return {}
    
    def _simulate_trades(self) -> None:
        """Simulate trade execution following VectorBT patterns."""
        self.logger.info("Simulating trades...")
        
        # Sort signals by timestamp
        sorted_signals = sorted(self.signals, key=lambda x: x['timestamp'])
        
        for signal in sorted_signals:
            try:
                self._process_signal(signal)
            except Exception as e:
                self.logger.error(f"Failed to process signal: {e}")
        
        # Close all positions at the end
        final_positions = self.position_manager.get_open_positions()
        if final_positions:
            self.logger.info(f"Closing {len(final_positions)} positions at end of backtest")
            self.position_manager.close_all_positions("end_of_backtest")
    
    def _process_signal(self, signal: Dict[str, Any]) -> None:
        """Process a single trading signal."""
        symbol = signal['symbol']
        direction = signal['direction']
        confidence = signal['confidence']
        price = signal['price']
        
        # Check if we already have a position
        existing_position = self.position_manager.get_open_position(symbol)
        
        # Handle different signal types
        if direction == SignalDirection.BUY.value:
            if existing_position is None:
                # Open new long position
                self._open_position(symbol, SignalDirection.BUY, price, confidence, signal)
            elif existing_position.side.value == 'short':
                # Close short position and open long
                self.position_manager.close_position(existing_position.id, "signal_reverse")
                self._open_position(symbol, SignalDirection.BUY, price, confidence, signal)
        
        elif direction == SignalDirection.SELL.value:
            if existing_position is None:
                # Open new short position
                self._open_position(symbol, SignalDirection.SELL, price, confidence, signal)
            elif existing_position.side.value == 'long':
                # Close long position and open short
                self.position_manager.close_position(existing_position.id, "signal_reverse")
                self._open_position(symbol, SignalDirection.SELL, price, confidence, signal)
        
        elif direction == SignalDirection.HOLD.value:
            # HOLD signals don't trigger trades
            pass
    
    def _open_position(self, symbol: str, direction: SignalDirection, price: float, 
                     confidence: float, signal: Dict[str, Any]) -> None:
        """Open a new position."""
        # Calculate position size
        portfolio_value = self.position_manager.current_balance
        max_position_value = portfolio_value * self.config.max_position_size
        
        # Size based on confidence
        base_size = max_position_value * confidence
        quantity = base_size / price
        
        # Calculate stop loss and take profit
        if direction == SignalDirection.BUY:
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)
        
        # Create position request
        request = PositionRequest(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=signal['strategy'],
            reason=signal.get('metadata', {}).get('reason', ''),
            metadata=signal['metadata']
        )
        
        # Open position
        position = self.position_manager.open_position(request)
        
        if position:
            # Log the trade
            trade_dict = {
                'timestamp': signal['timestamp'],
                'action': 'OPEN',
                'symbol': symbol,
                'direction': direction.value,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'strategy': signal['strategy'],
                'position_id': position.id
            }
            self.trade_log.append(trade_dict)
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        portfolio_summary = self.position_manager.get_portfolio_summary()
        
        # Calculate additional metrics
        equity_curve = self._create_equity_curve_series()
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades = self.position_manager.get_closed_positions()
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades if t.exit_time]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': max([t.realized_pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.realized_pnl for t in losing_trades]) if losing_trades else 0
        }
    
    def _create_equity_curve_series(self) -> pd.Series:
        """Create equity curve as pandas Series."""
        timestamps = []
        values = []
        
        # Start with initial cash
        timestamps.append(self.config.start_date)
        values.append(self.config.initial_cash)
        
        # Add portfolio value changes over time
        for trade in sorted(self.trade_log, key=lambda x: x['timestamp']):
            timestamps.append(trade['timestamp'])
            values.append(self.position_manager.current_balance)
        
        return pd.Series(values, index=pd.to_datetime(timestamps, utc=True))
    
    def _create_equity_curve(self) -> pd.DataFrame:
        """Create equity curve DataFrame."""
        equity_series = self._create_equity_curve_series()
        
        df = pd.DataFrame({
            'equity': equity_series,
            'returns': equity_series.pct_change(),
            'cumulative_returns': (equity_series / equity_series.iloc[0]) - 1
        })
        
        return df
    
    def _create_trades_dataframe(self) -> pd.DataFrame:
        """Create trades DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _create_positions_dataframe(self) -> pd.DataFrame:
        """Create positions DataFrame."""
        positions = self.position_manager.get_all_positions()
        
        if not positions:
            return pd.DataFrame()
        
        data = []
        for position in positions:
            data.append({
                'position_id': position.id,
                'symbol': position.symbol,
                'side': position.side.value,
                'status': position.status.value,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'entry_value': position.entry_value,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'strategy': position.strategy
            })
        
        df = pd.DataFrame(data)
        return df
    
    def _create_signals_dataframe(self) -> pd.DataFrame:
        """Create signals DataFrame."""
        if not self.signal_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.signal_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
