"""
Comprehensive Debug Logger for TRADING-AI.
Provides detailed debugging and validation logging for production-grade trading.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import traceback
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..infrastructure.logging import get_logger


class LogLevel(Enum):
    """Log levels for debugging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DebugCategory(Enum):
    """Categories for debugging logs."""
    DECISION_ENGINE = "decision_engine"
    STRATEGY = "strategy"
    POSITION = "position"
    SIGNAL = "signal"
    MARKET_DATA = "market_data"
    RISK = "risk"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class DebugEntry:
    """Debug log entry with comprehensive information."""
    timestamp: datetime
    level: LogLevel
    category: DebugCategory
    component: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    execution_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)


class DebugLogger:
    """
    Comprehensive debug logger for production-grade trading system.
    
    Key features:
    - Structured logging with categories and components
    - Performance timing and profiling
    - Error tracking and stack traces
    - Context preservation across calls
    - Validation and debugging helpers
    """
    
    def __init__(self, log_file: Optional[str] = None, max_entries: int = 10000):
        """Initialize debug logger."""
        self.logger = get_logger("debug_logger")
        
        # Configuration
        self.max_entries = max_entries
        self.log_file = log_file
        
        # Storage
        self.entries: List[DebugEntry] = []
        self.context_stack: List[Dict[str, Any]] = [{}]
        
        # Performance tracking
        self.performance_timers: Dict[str, datetime] = {}
        self.performance_stats: Dict[str, List[float]] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.error_patterns: Dict[str, List[DebugEntry]] = {}
        
        # Validation results
        self.validation_results: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger.info("DebugLogger initialized with comprehensive logging")
    
    def debug(self, category: DebugCategory, component: str, message: str, 
              data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, category, component, message, data, **kwargs)
    
    def info(self, category: DebugCategory, component: str, message: str, 
             data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, category, component, message, data, **kwargs)
    
    def warning(self, category: DebugCategory, component: str, message: str, 
                data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, category, component, message, data, **kwargs)
    
    def error(self, category: DebugCategory, component: str, message: str, 
              data: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        stack_trace = traceback.format_exc() if exception else None
        
        # Track error patterns
        error_key = f"{category.value}_{component}_{message}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        entry = self._log(LogLevel.ERROR, category, component, message, data, 
                        stack_trace=stack_trace, **kwargs)
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = []
        self.error_patterns[error_key].append(entry)
    
    def critical(self, category: DebugCategory, component: str, message: str, 
                 data: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        stack_trace = traceback.format_exc() if exception else None
        self._log(LogLevel.CRITICAL, category, component, message, data, 
                 stack_trace=stack_trace, **kwargs)
    
    def _log(self, level: LogLevel, category: DebugCategory, component: str, 
             message: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> DebugEntry:
        """Internal logging method."""
        # Create debug entry
        entry = DebugEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            component=component,
            message=message,
            data=data or {},
            context=self.context_stack[-1].copy(),
            **kwargs
        )
        
        # Add to storage
        self.entries.append(entry)
        
        # Limit entries
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        
        # Log to standard logger
        log_message = f"[{category.value}] {component}: {message}"
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(entry)
        
        return entry
    
    def start_timer(self, timer_name: str) -> None:
        """Start performance timer."""
        self.performance_timers[timer_name] = datetime.now()
        self.debug(DebugCategory.SYSTEM, "timer", f"Started timer: {timer_name}")
    
    def stop_timer(self, timer_name: str) -> Optional[float]:
        """Stop performance timer and return elapsed time."""
        if timer_name not in self.performance_timers:
            self.warning(DebugCategory.SYSTEM, "timer", f"Timer not found: {timer_name}")
            return None
        
        start_time = self.performance_timers.pop(timer_name)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Track performance stats
        if timer_name not in self.performance_stats:
            self.performance_stats[timer_name] = []
        self.performance_stats[timer_name].append(elapsed)
        
        self.debug(DebugCategory.PERFORMANCE, "timer", 
                  f"Stopped timer: {timer_name} | Elapsed: {elapsed:.4f}s",
                  data={"elapsed_seconds": elapsed, "timer_name": timer_name})
        
        return elapsed
    
    def with_timer(self, timer_name: str):
        """Context manager for timing operations."""
        class TimerContext:
            def __init__(self, debug_logger: 'DebugLogger', name: str):
                self.debug_logger = debug_logger
                self.name = name
            
            def __enter__(self):
                self.debug_logger.start_timer(self.name)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = self.debug_logger.stop_timer(self.name)
                if exc_type:
                    self.debug_logger.error(DebugCategory.SYSTEM, "timer", 
                                          f"Timer {self.name} completed with exception",
                                          exception=exc_val)
                return False  # Don't suppress exceptions
        
        return TimerContext(self, timer_name)
    
    def push_context(self, **context) -> None:
        """Push context to context stack."""
        self.context_stack.append(context)
        self.debug(DebugCategory.SYSTEM, "context", f"Pushed context: {context}")
    
    def pop_context(self) -> Dict[str, Any]:
        """Pop context from context stack."""
        if len(self.context_stack) > 1:
            context = self.context_stack.pop()
            self.debug(DebugCategory.SYSTEM, "context", f"Popped context: {context}")
            return context
        return {}
    
    def validate_decision_engine(self, decision_data: Dict[str, Any]) -> bool:
        """Validate decision engine output."""
        validation_result = {
            "timestamp": datetime.now(),
            "component": "decision_engine",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            required_fields = ["action", "confidence", "entry", "stop_loss", "take_profit", "reasoning"]
            for field in required_fields:
                if field not in decision_data:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Validate action
            valid_actions = ["BUY", "SELL", "HOLD"]
            if decision_data.get("action") not in valid_actions:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid action: {decision_data.get('action')}")
            
            # Validate confidence
            confidence = decision_data.get("confidence")
            if confidence is not None and (confidence < 0 or confidence > 1):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid confidence: {confidence}")
            
            # Validate prices
            entry_price = decision_data.get("entry")
            stop_loss = decision_data.get("stop_loss")
            take_profit = decision_data.get("take_profit")
            
            if entry_price and stop_loss and take_profit:
                if entry_price <= 0:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Invalid entry price: {entry_price}")
                
                if stop_loss <= 0:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Invalid stop loss: {stop_loss}")
                
                if take_profit <= 0:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Invalid take profit: {take_profit}")
                
                # Check price relationships
                action = decision_data.get("action")
                if action == "BUY":
                    if stop_loss >= entry_price:
                        validation_result["errors"].append("BUY: Stop loss must be below entry price")
                    if take_profit <= entry_price:
                        validation_result["errors"].append("BUY: Take profit must be above entry price")
                elif action == "SELL":
                    if stop_loss <= entry_price:
                        validation_result["errors"].append("SELL: Stop loss must be above entry price")
                    if take_profit >= entry_price:
                        validation_result["errors"].append("SELL: Take profit must be below entry price")
            
            # Log validation result
            if validation_result["valid"]:
                self.debug(DebugCategory.VALIDATION, "decision_engine", 
                          "Decision engine validation passed",
                          data=decision_data)
            else:
                self.error(DebugCategory.VALIDATION, "decision_engine", 
                          "Decision engine validation failed",
                          data={"errors": validation_result["errors"], "decision_data": decision_data})
            
            # Store validation result
            if "decision_engine" not in self.validation_results:
                self.validation_results["decision_engine"] = []
            self.validation_results["decision_engine"].append(validation_result)
            
            return validation_result["valid"]
            
        except Exception as e:
            self.error(DebugCategory.VALIDATION, "decision_engine", 
                      "Decision engine validation exception",
                      exception=e, data=decision_data)
            return False
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validate signal data."""
        validation_result = {
            "timestamp": datetime.now(),
            "component": "signal",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            required_fields = ["symbol", "direction", "confidence", "timestamp"]
            for field in required_fields:
                if field not in signal_data:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Validate direction
            valid_directions = ["BUY", "SELL", "HOLD"]
            if signal_data.get("direction") not in valid_directions:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid direction: {signal_data.get('direction')}")
            
            # Validate confidence
            confidence = signal_data.get("confidence")
            if confidence is not None and (confidence < 0 or confidence > 1):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid confidence: {confidence}")
            
            # Validate symbol
            symbol = signal_data.get("symbol")
            if symbol and (not isinstance(symbol, str) or len(symbol.strip()) == 0):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid symbol: {symbol}")
            
            # Log validation result
            if validation_result["valid"]:
                self.debug(DebugCategory.VALIDATION, "signal", 
                          "Signal validation passed",
                          data=signal_data)
            else:
                self.error(DebugCategory.VALIDATION, "signal", 
                          "Signal validation failed",
                          data={"errors": validation_result["errors"], "signal_data": signal_data})
            
            # Store validation result
            if "signal" not in self.validation_results:
                self.validation_results["signal"] = []
            self.validation_results["signal"].append(validation_result)
            
            return validation_result["valid"]
            
        except Exception as e:
            self.error(DebugCategory.VALIDATION, "signal", 
                      "Signal validation exception",
                      exception=e, data=signal_data)
            return False
    
    def validate_position(self, position_data: Dict[str, Any]) -> bool:
        """Validate position data."""
        validation_result = {
            "timestamp": datetime.now(),
            "component": "position",
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check required fields
            required_fields = ["symbol", "side", "quantity", "entry_price"]
            for field in required_fields:
                if field not in position_data:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Missing required field: {field}")
            
            # Validate side
            valid_sides = ["long", "short"]
            if position_data.get("side") not in valid_sides:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid side: {position_data.get('side')}")
            
            # Validate quantity
            quantity = position_data.get("quantity")
            if quantity is not None and quantity <= 0:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid quantity: {quantity}")
            
            # Validate entry price
            entry_price = position_data.get("entry_price")
            if entry_price is not None and entry_price <= 0:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Invalid entry price: {entry_price}")
            
            # Log validation result
            if validation_result["valid"]:
                self.debug(DebugCategory.VALIDATION, "position", 
                          "Position validation passed",
                          data=position_data)
            else:
                self.error(DebugCategory.VALIDATION, "position", 
                          "Position validation failed",
                          data={"errors": validation_result["errors"], "position_data": position_data})
            
            # Store validation result
            if "position" not in self.validation_results:
                self.validation_results["position"] = []
            self.validation_results["position"].append(validation_result)
            
            return validation_result["valid"]
            
        except Exception as e:
            self.error(DebugCategory.VALIDATION, "position", 
                      "Position validation exception",
                      exception=e, data=position_data)
            return False
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary."""
        summary = {
            "total_entries": len(self.entries),
            "error_count": len([e for e in self.entries if e.level == LogLevel.ERROR]),
            "warning_count": len([e for e in self.entries if e.level == LogLevel.WARNING]),
            "critical_count": len([e for e in self.entries if e.level == LogLevel.CRITICAL]),
            "category_breakdown": {},
            "recent_errors": [],
            "performance_summary": {},
            "validation_summary": {},
            "error_patterns": {}
        }
        
        # Category breakdown
        for category in DebugCategory:
            category_entries = [e for e in self.entries if e.category == category]
            summary["category_breakdown"][category.value] = len(category_entries)
        
        # Recent errors
        recent_errors = [e for e in self.entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        summary["recent_errors"] = recent_errors[-10:] if recent_errors else []
        
        # Performance summary
        for timer_name, times in self.performance_stats.items():
            if times:
                summary["performance_summary"][timer_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        # Validation summary
        for component, results in self.validation_results.items():
            valid_count = len([r for r in results if r["valid"]])
            total_count = len(results)
            summary["validation_summary"][component] = {
                "total": total_count,
                "valid": valid_count,
                "invalid": total_count - valid_count,
                "success_rate": valid_count / total_count if total_count > 0 else 0.0
            }
        
        # Error patterns
        for error_key, entries in self.error_patterns.items():
            summary["error_patterns"][error_key] = {
                "count": len(entries),
                "first_occurrence": entries[0].timestamp,
                "last_occurrence": entries[-1].timestamp,
                "frequency": len(entries) / max(1, (datetime.now() - entries[0].timestamp).total_seconds() / 3600)  # per hour
            }
        
        return summary
    
    def _write_to_file(self, entry: DebugEntry) -> None:
        """Write debug entry to file."""
        try:
            log_data = {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "category": entry.category.value,
                "component": entry.component,
                "message": entry.message,
                "data": entry.data,
                "context": entry.context,
                "execution_time": entry.execution_time,
                "stack_trace": entry.stack_trace
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_data, default=str) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write debug entry to file: {e}")
    
    def clear_logs(self) -> None:
        """Clear all debug logs."""
        self.entries.clear()
        self.performance_timers.clear()
        self.performance_stats.clear()
        self.error_counts.clear()
        self.error_patterns.clear()
        self.validation_results.clear()
        
        self.info(DebugCategory.SYSTEM, "debug_logger", "Debug logs cleared")
    
    def export_logs(self, filepath: str, level_filter: Optional[LogLevel] = None) -> None:
        """Export debug logs to file."""
        try:
            entries_to_export = self.entries
            
            if level_filter:
                entries_to_export = [e for e in self.entries if e.level == level_filter]
            
            export_data = []
            for entry in entries_to_export:
                export_data.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level.value,
                    "category": entry.category.value,
                    "component": entry.component,
                    "message": entry.message,
                    "data": entry.data,
                    "context": entry.context,
                    "execution_time": entry.execution_time,
                    "stack_trace": entry.stack_trace
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.info(DebugCategory.SYSTEM, "debug_logger", 
                     f"Exported {len(export_data)} debug entries to {filepath}")
            
        except Exception as e:
            self.error(DebugCategory.SYSTEM, "debug_logger", 
                      f"Failed to export debug logs: {e}", exception=e)
