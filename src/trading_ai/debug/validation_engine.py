"""
Validation Engine for TRADING-AI.
Comprehensive validation system for production-grade trading components.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from .debug_logger import DebugLogger, DebugCategory


class ValidationStatus(Enum):
    """Validation status levels."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class ValidationSeverity(Enum):
    """Validation severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: ValidationSeverity
    category: str
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    status: ValidationStatus
    message: str
    severity: ValidationSeverity
    category: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None


class ValidationEngine:
    """
    Comprehensive validation engine for production-grade trading system.
    
    Key features:
    - Extensible rule-based validation
    - Component-specific validation suites
    - Performance impact analysis
    - Validation history and trends
    - Automated validation scheduling
    """
    
    def __init__(self, debug_logger: Optional[DebugLogger] = None):
        """Initialize validation engine."""
        self.debug_logger = debug_logger or DebugLogger()
        
        # Validation rules
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_suites: Dict[str, List[str]] = {}
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.validation_stats: Dict[str, Dict[str, int]] = {}
        
        # Performance tracking
        self.validation_performance: Dict[str, List[float]] = {}
        
        # Initialize built-in validation rules
        self._initialize_builtin_rules()
        
        self.debug_logger.info(DebugCategory.SYSTEM, "validation_engine", 
                              "ValidationEngine initialized with comprehensive rules")
    
    def _initialize_builtin_rules(self) -> None:
        """Initialize built-in validation rules."""
        
        # Decision Engine Rules
        self.add_rule(ValidationRule(
            name="decision_engine_required_fields",
            description="Decision engine must have all required fields",
            validator=self._validate_decision_engine_fields,
            severity=ValidationSeverity.HIGH,
            category="decision_engine"
        ))
        
        self.add_rule(ValidationRule(
            name="decision_engine_action_valid",
            description="Decision engine action must be valid",
            validator=self._validate_decision_engine_action,
            severity=ValidationSeverity.HIGH,
            category="decision_engine"
        ))
        
        self.add_rule(ValidationRule(
            name="decision_engine_confidence_range",
            description="Decision engine confidence must be in valid range",
            validator=self._validate_decision_engine_confidence,
            severity=ValidationSeverity.MEDIUM,
            category="decision_engine"
        ))
        
        # Signal Rules
        self.add_rule(ValidationRule(
            name="signal_required_fields",
            description="Signal must have all required fields",
            validator=self._validate_signal_fields,
            severity=ValidationSeverity.HIGH,
            category="signal"
        ))
        
        self.add_rule(ValidationRule(
            name="signal_direction_valid",
            description="Signal direction must be valid",
            validator=self._validate_signal_direction,
            severity=ValidationSeverity.HIGH,
            category="signal"
        ))
        
        # Position Rules
        self.add_rule(ValidationRule(
            name="position_required_fields",
            description="Position must have all required fields",
            validator=self._validate_position_fields,
            severity=ValidationSeverity.HIGH,
            category="position"
        ))
        
        self.add_rule(ValidationRule(
            name="position_quantity_positive",
            description="Position quantity must be positive",
            validator=self._validate_position_quantity,
            severity=ValidationSeverity.HIGH,
            category="position"
        ))
        
        # Market Data Rules
        self.add_rule(ValidationRule(
            name="market_data_price_positive",
            description="Market data prices must be positive",
            validator=self._validate_market_data_prices,
            severity=ValidationSeverity.HIGH,
            category="market_data"
        ))
        
        self.add_rule(ValidationRule(
            name="market_data_timestamp_valid",
            description="Market data timestamps must be valid",
            validator=self._validate_market_data_timestamps,
            severity=ValidationSeverity.MEDIUM,
            category="market_data"
        ))
        
        # Risk Management Rules
        self.add_rule(ValidationRule(
            name="risk_position_size_limit",
            description="Position size must be within risk limits",
            validator=self._validate_risk_position_size,
            severity=ValidationSeverity.HIGH,
            category="risk"
        ))
        
        self.add_rule(ValidationRule(
            name="risk_stop_loss_valid",
            description="Stop loss must be valid for position direction",
            validator=self._validate_risk_stop_loss,
            severity=ValidationSeverity.HIGH,
            category="risk"
        ))
        
        # Create validation suites
        self.create_suite("decision_engine", [
            "decision_engine_required_fields",
            "decision_engine_action_valid", 
            "decision_engine_confidence_range"
        ])
        
        self.create_suite("signal", [
            "signal_required_fields",
            "signal_direction_valid"
        ])
        
        self.create_suite("position", [
            "position_required_fields",
            "position_quantity_positive"
        ])
        
        self.create_suite("market_data", [
            "market_data_price_positive",
            "market_data_timestamp_valid"
        ])
        
        self.create_suite("risk", [
            "risk_position_size_limit",
            "risk_stop_loss_valid"
        ])
        
        self.create_suite("comprehensive", [
            "decision_engine_required_fields",
            "decision_engine_action_valid",
            "decision_engine_confidence_range",
            "signal_required_fields",
            "signal_direction_valid",
            "position_required_fields",
            "position_quantity_positive",
            "market_data_price_positive",
            "market_data_timestamp_valid",
            "risk_position_size_limit",
            "risk_stop_loss_valid"
        ])
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules[rule.name] = rule
        self.debug_logger.debug(DebugCategory.VALIDATION, "validation_engine", 
                               f"Added validation rule: {rule.name}")
    
    def create_suite(self, suite_name: str, rule_names: List[str]) -> None:
        """Create a validation suite."""
        # Validate that all rules exist
        for rule_name in rule_names:
            if rule_name not in self.rules:
                self.debug_logger.error(DebugCategory.VALIDATION, "validation_engine", 
                                      f"Rule not found for suite: {rule_name}")
                return
        
        self.validation_suites[suite_name] = rule_names
        self.debug_logger.info(DebugCategory.VALIDATION, "validation_engine", 
                              f"Created validation suite: {suite_name} with {len(rule_names)} rules")
    
    def validate(self, data: Any, suite_name: Optional[str] = None, 
                 rule_names: Optional[List[str]] = None) -> List[ValidationResult]:
        """
        Validate data against specified rules or suite.
        
        Args:
            data: Data to validate
            suite_name: Name of validation suite to run
            rule_names: Specific rules to run
            
        Returns:
            List of validation results
        """
        results = []
        
        try:
            # Determine which rules to run
            if suite_name:
                if suite_name not in self.validation_suites:
                    self.debug_logger.error(DebugCategory.VALIDATION, "validation_engine", 
                                          f"Validation suite not found: {suite_name}")
                    return results
                
                rules_to_run = [self.rules[name] for name in self.validation_suites[suite_name] 
                               if name in self.rules and self.rules[name].enabled]
            
            elif rule_names:
                rules_to_run = [self.rules[name] for name in rule_names 
                               if name in self.rules and self.rules[name].enabled]
            
            else:
                # Run all enabled rules
                rules_to_run = [rule for rule in self.rules.values() if rule.enabled]
            
            # Run validations
            for rule in rules_to_run:
                result = self._run_validation(rule, data)
                results.append(result)
                
                # Log result
                if result.status == ValidationStatus.FAILED:
                    self.debug_logger.error(DebugCategory.VALIDATION, "validation_engine", 
                                          f"Validation failed: {rule.name} - {result.message}",
                                          data={"rule": rule.name, "validation_result": result.data})
                elif result.status == ValidationStatus.WARNING:
                    self.debug_logger.warning(DebugCategory.VALIDATION, "validation_engine", 
                                            f"Validation warning: {rule.name} - {result.message}",
                                            data={"rule": rule.name, "validation_result": result.data})
                else:
                    self.debug_logger.debug(DebugCategory.VALIDATION, "validation_engine", 
                                          f"Validation passed: {rule.name}")
            
            # Store results
            self.validation_history.extend(results)
            
            # Update stats
            self._update_validation_stats(results)
            
            self.debug_logger.info(DebugCategory.VALIDATION, "validation_engine", 
                                  f"Validation completed: {len(results)} rules checked")
            
        except Exception as e:
            self.debug_logger.error(DebugCategory.VALIDATION, "validation_engine", 
                                  f"Validation exception: {e}", exception=e)
        
        return results
    
    def _run_validation(self, rule: ValidationRule, data: Any) -> ValidationResult:
        """Run a single validation rule."""
        start_time = datetime.now()
        
        try:
            # Run validator
            is_valid = rule.validator(data)
            
            # Determine status
            if is_valid:
                status = ValidationStatus.PASSED
                message = f"Validation passed: {rule.description}"
            else:
                status = ValidationStatus.FAILED
                message = f"Validation failed: {rule.description}"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ValidationResult(
                rule_name=rule.name,
                status=status,
                message=message,
                severity=rule.severity,
                category=rule.category,
                timestamp=start_time,
                execution_time=execution_time
            )
            
            # Track performance
            if rule.name not in self.validation_performance:
                self.validation_performance[rule.name] = []
            self.validation_performance[rule.name].append(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Validation error: {rule.description} - {str(e)}",
                severity=ValidationSeverity.CRITICAL,
                category=rule.category,
                timestamp=start_time,
                execution_time=execution_time,
                data={"error": str(e)}
            )
    
    def _update_validation_stats(self, results: List[ValidationResult]) -> None:
        """Update validation statistics."""
        for result in results:
            if result.category not in self.validation_stats:
                self.validation_stats[result.category] = {
                    "passed": 0,
                    "failed": 0,
                    "warning": 0,
                    "total": 0
                }
            
            self.validation_stats[result.category]["total"] += 1
            
            if result.status == ValidationStatus.PASSED:
                self.validation_stats[result.category]["passed"] += 1
            elif result.status == ValidationStatus.FAILED:
                self.validation_stats[result.category]["failed"] += 1
            elif result.status == ValidationStatus.WARNING:
                self.validation_stats[result.category]["warning"] += 1
    
    # Built-in validators
    def _validate_decision_engine_fields(self, data: Dict[str, Any]) -> bool:
        """Validate decision engine has required fields."""
        required_fields = ["action", "confidence", "entry", "stop_loss", "take_profit", "reasoning"]
        return all(field in data for field in required_fields)
    
    def _validate_decision_engine_action(self, data: Dict[str, Any]) -> bool:
        """Validate decision engine action is valid."""
        valid_actions = ["BUY", "SELL", "HOLD"]
        return data.get("action") in valid_actions
    
    def _validate_decision_engine_confidence(self, data: Dict[str, Any]) -> bool:
        """Validate decision engine confidence is in valid range."""
        confidence = data.get("confidence")
        return confidence is not None and 0.0 <= confidence <= 1.0
    
    def _validate_signal_fields(self, data: Dict[str, Any]) -> bool:
        """Validate signal has required fields."""
        required_fields = ["symbol", "direction", "confidence", "timestamp"]
        return all(field in data for field in required_fields)
    
    def _validate_signal_direction(self, data: Dict[str, Any]) -> bool:
        """Validate signal direction is valid."""
        valid_directions = ["BUY", "SELL", "HOLD"]
        return data.get("direction") in valid_directions
    
    def _validate_position_fields(self, data: Dict[str, Any]) -> bool:
        """Validate position has required fields."""
        required_fields = ["symbol", "side", "quantity", "entry_price"]
        return all(field in data for field in required_fields)
    
    def _validate_position_quantity(self, data: Dict[str, Any]) -> bool:
        """Validate position quantity is positive."""
        quantity = data.get("quantity")
        return quantity is not None and quantity > 0
    
    def _validate_market_data_prices(self, data: Dict[str, Any]) -> bool:
        """Validate market data prices are positive."""
        if isinstance(data, dict):
            # Check for OHLC data
            for price_field in ["open", "high", "low", "close", "price"]:
                if price_field in data:
                    price = data[price_field]
                    if price is not None and price <= 0:
                        return False
            
            # Check nested market data
            for key, value in data.items():
                if isinstance(value, dict):
                    if not self._validate_market_data_prices(value):
                        return False
        
        return True
    
    def _validate_market_data_timestamps(self, data: Dict[str, Any]) -> bool:
        """Validate market data timestamps are valid."""
        if isinstance(data, dict):
            # Check for timestamp field
            if "timestamp" in data:
                timestamp = data["timestamp"]
                if timestamp is not None:
                    # Should be recent (not too old or too far in future)
                    if isinstance(timestamp, datetime):
                        now = datetime.now()
                        if timestamp < now - timedelta(days=1) or timestamp > now + timedelta(hours=1):
                            return False
            
            # Check nested data
            for key, value in data.items():
                if isinstance(value, dict):
                    if not self._validate_market_data_timestamps(value):
                        return False
        
        return True
    
    def _validate_risk_position_size(self, data: Dict[str, Any]) -> bool:
        """Validate position size is within risk limits."""
        # This is a simplified check - in production, would check against portfolio limits
        if isinstance(data, dict):
            if "position_size" in data:
                position_size = data["position_size"]
                if position_size is not None:
                    return 0.0 <= position_size <= 1.0  # Max 100% of portfolio
        
        return True
    
    def _validate_risk_stop_loss(self, data: Dict[str, Any]) -> bool:
        """Validate stop loss is valid for position direction."""
        if isinstance(data, dict):
            action = data.get("action")
            entry_price = data.get("entry")
            stop_loss = data.get("stop_loss")
            
            if all([action, entry_price, stop_loss]):
                if action == "BUY":
                    return stop_loss < entry_price
                elif action == "SELL":
                    return stop_loss > entry_price
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        summary = {
            "total_validations": len(self.validation_history),
            "validation_stats": self.validation_stats,
            "rule_performance": {},
            "recent_validations": [],
            "failure_rate": 0.0,
            "most_failing_rules": []
        }
        
        # Calculate failure rate
        if self.validation_history:
            failed_count = len([r for r in self.validation_history if r.status == ValidationStatus.FAILED])
            summary["failure_rate"] = failed_count / len(self.validation_history)
        
        # Rule performance
        for rule_name, times in self.validation_performance.items():
            if times:
                summary["rule_performance"][rule_name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        # Recent validations
        summary["recent_validations"] = self.validation_history[-20:] if self.validation_history else []
        
        # Most failing rules
        rule_failures = {}
        for result in self.validation_history:
            if result.status == ValidationStatus.FAILED:
                rule_failures[result.rule_name] = rule_failures.get(result.rule_name, 0) + 1
        
        summary["most_failing_rules"] = sorted(rule_failures.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return summary
    
    def enable_rule(self, rule_name: str) -> None:
        """Enable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            self.debug_logger.info(DebugCategory.VALIDATION, "validation_engine", 
                                  f"Enabled validation rule: {rule_name}")
    
    def disable_rule(self, rule_name: str) -> None:
        """Disable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            self.debug_logger.info(DebugCategory.VALIDATION, "validation_engine", 
                                  f"Disabled validation rule: {rule_name}")
    
    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        self.validation_stats.clear()
        self.validation_performance.clear()
        
        self.debug_logger.info(DebugCategory.SYSTEM, "validation_engine", "Validation history cleared")
    
    def export_validation_report(self, filepath: str, include_details: bool = False) -> None:
        """Export validation report to file."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_validation_summary(),
                "rules": {name: {
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.value,
                    "category": rule.category,
                    "enabled": rule.enabled
                } for name, rule in self.rules.items()},
                "validation_suites": self.validation_suites
            }
            
            if include_details:
                report["validation_history"] = [
                    {
                        "rule_name": result.rule_name,
                        "status": result.status.value,
                        "message": result.message,
                        "severity": result.severity.value,
                        "category": result.category,
                        "timestamp": result.timestamp.isoformat(),
                        "execution_time": result.execution_time,
                        "data": result.data
                    }
                    for result in self.validation_history
                ]
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.debug_logger.info(DebugCategory.SYSTEM, "validation_engine", 
                                  f"Validation report exported to {filepath}")
            
        except Exception as e:
            self.debug_logger.error(DebugCategory.SYSTEM, "validation_engine", 
                                  f"Failed to export validation report: {e}", exception=e)
