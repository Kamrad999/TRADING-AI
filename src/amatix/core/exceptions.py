"""AMATIS Exception Hierarchy — Institutional Error Handling.

Defines a typed exception hierarchy for all AMATIS components.
All exceptions must inherit from appropriate base classes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorSeverity(Enum):
    """Severity levels for system errors."""
    CRITICAL = "critical"  # System must halt
    HIGH = "high"  # Operation must fail
    MEDIUM = "medium"  # Operation should fail but can continue
    LOW = "low"  # Warning, operation can continue


class AmatisException(Exception):
    """Base exception for all AMATIS errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.component = component
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception for logging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "component": self.component,
            "context": self.context,
        }


# =============================================================================
# CRITICAL SYSTEM ERRORS
# =============================================================================

class CriticalSystemError(AmatisException):
    """Critical system error requiring immediate halt."""
    
    def __init__(
        self,
        message: str,
        component: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            component=component,
            context=context,
        )


class ReplayCorruptionError(CriticalSystemError):
    """Replay data is corrupted or invalid."""
    
    def __init__(
        self,
        event_index: int,
        expected_checksum: str,
        actual_checksum: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Replay corruption at event {event_index}: checksum mismatch",
            component="replay_engine",
            context={
                "event_index": event_index,
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
                **(context or {}),
            },
        )


class DeterminismViolationError(CriticalSystemError):
    """Determinism invariant violated."""
    
    def __init__(
        self,
        divergence_point: int,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Determinism violation at event {divergence_point}",
            component="determinism_validator",
            context={"divergence_point": divergence_point, **(context or {})},
        )


class StateCorruptionError(CriticalSystemError):
    """System state is corrupted and unrecoverable."""
    
    def __init__(
        self,
        component: str,
        state_description: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"State corruption in {component}: {state_description}",
            component=component,
            context={"state_description": state_description, **(context or {})},
        )


# =============================================================================
# RISK ENGINE ERRORS
# =============================================================================

class RiskEngineError(AmatisException):
    """Base exception for risk engine errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="risk_engine",
            context=context,
        )


class RiskRuleEvaluationError(RiskEngineError):
    """Risk rule evaluation failed."""
    
    def __init__(
        self,
        rule_name: str,
        order_id: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Risk rule '{rule_name}' evaluation failed for order {order_id}",
            severity=ErrorSeverity.HIGH,
            context={
                "rule_name": rule_name,
                "order_id": order_id,
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {}),
            },
        )
        self.original_error = original_error


class RiskLimitBreachedError(RiskEngineError):
    """Risk limit exceeded."""
    
    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        order_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Risk limit '{limit_type}' breached: {current_value} > {limit_value}",
            severity=ErrorSeverity.HIGH,
            context={
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "order_id": order_id,
                **(context or {}),
            },
        )


class RiskConfigurationError(RiskEngineError):
    """Risk engine configuration is invalid."""
    
    def __init__(
        self,
        config_error: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Risk configuration error: {config_error}",
            severity=ErrorSeverity.HIGH,
            context={"config_error": config_error, **(context or {})},
        )


# =============================================================================
# ORDER MANAGEMENT ERRORS
# =============================================================================

class OrderManagementError(AmatisException):
    """Base exception for OMS errors."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="order_manager",
            context={"order_id": order_id, **(context or {})},
        )
        self.order_id = order_id


class InvalidStateTransitionError(OrderManagementError):
    """Invalid order state transition attempted."""
    
    def __init__(
        self,
        order_id: str,
        current_state: str,
        attempted_state: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Invalid state transition for order {order_id}: {current_state} → {attempted_state}",
            order_id=order_id,
            severity=ErrorSeverity.HIGH,
            context={
                "current_state": current_state,
                "attempted_state": attempted_state,
                **(context or {}),
            },
        )


class OrderNotFoundError(OrderManagementError):
    """Order not found in OMS."""
    
    def __init__(
        self,
        order_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Order {order_id} not found",
            order_id=order_id,
            severity=ErrorSeverity.MEDIUM,
            context=context,
        )


class DuplicateFillError(OrderManagementError):
    """Duplicate fill detected for order."""
    
    def __init__(
        self,
        order_id: str,
        fill_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Duplicate fill {fill_id} for order {order_id}",
            order_id=order_id,
            severity=ErrorSeverity.HIGH,
            context={"fill_id": fill_id, **(context or {})},
        )


class FillReconciliationError(OrderManagementError):
    """Fill reconciliation failed."""
    
    def __init__(
        self,
        order_id: str,
        expected_qty: float,
        actual_qty: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Fill reconciliation failed for order {order_id}: expected {expected_qty}, got {actual_qty}",
            order_id=order_id,
            severity=ErrorSeverity.HIGH,
            context={
                "expected_qty": expected_qty,
                "actual_qty": actual_qty,
                **(context or {}),
            },
        )


class OrderInconsistencyError(OrderManagementError):
    """Order state is inconsistent."""
    
    def __init__(
        self,
        order_id: str,
        inconsistency_type: str,
        details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Order inconsistency detected: {inconsistency_type}",
            order_id=order_id,
            severity=ErrorSeverity.CRITICAL,
            context={
                "inconsistency_type": inconsistency_type,
                "details": details,
                **(context or {}),
            },
        )


# =============================================================================
# EVENT BUS ERRORS
# =============================================================================

class EventBusError(AmatisException):
    """Base exception for event bus errors."""
    
    def __init__(
        self,
        message: str,
        event_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="event_bus",
            context={"event_type": event_type, **(context or {})},
        )


class EventValidationError(EventBusError):
    """Event payload validation failed."""
    
    def __init__(
        self,
        event_type: str,
        validation_error: str,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Event validation failed for {event_type}: {validation_error}",
            event_type=event_type,
            severity=ErrorSeverity.HIGH,
            context={
                "validation_error": validation_error,
                "payload": payload,
                **(context or {}),
            },
        )


class HandlerExecutionError(EventBusError):
    """Event handler execution failed."""
    
    def __init__(
        self,
        event_type: str,
        handler_name: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Handler '{handler_name}' failed for event {event_type}",
            event_type=event_type,
            severity=ErrorSeverity.MEDIUM,
            context={
                "handler_name": handler_name,
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {}),
            },
        )
        self.original_error = original_error


class EventQueueOverflowError(EventBusError):
    """Event queue overflow detected."""
    
    def __init__(
        self,
        queue_size: int,
        max_size: int,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Event queue overflow: {queue_size} > {max_size}",
            severity=ErrorSeverity.HIGH,
            context={
                "queue_size": queue_size,
                "max_size": max_size,
                **(context or {}),
            },
        )


# =============================================================================
# PERSISTENCE ERRORS
# =============================================================================

class PersistenceError(AmatisException):
    """Base exception for persistence errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="persistence",
            context={"operation": operation, **(context or {})},
        )


class SaveError(PersistenceError):
    """Database save operation failed."""
    
    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Save failed for {entity_type} {entity_id}",
            operation="save",
            severity=ErrorSeverity.HIGH,
            context={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {}),
            },
        )
        self.original_error = original_error


class QueryError(PersistenceError):
    """Database query failed."""
    
    def __init__(
        self,
        query_description: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Query failed: {query_description}",
            operation="query",
            severity=ErrorSeverity.HIGH,
            context={
                "query_description": query_description,
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {})},
        )
        self.original_error = original_error


class ConnectionError(PersistenceError):
    """Database connection failed."""
    
    def __init__(
        self,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message="Database connection failed",
            operation="connect",
            severity=ErrorSeverity.CRITICAL,
            context={
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {})},
        )
        self.original_error = original_error


class IdempotencyConflictError(PersistenceError):
    """Idempotency key conflict detected."""
    
    def __init__(
        self,
        idempotency_key: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Idempotency key conflict: {idempotency_key}",
            operation="save",
            severity=ErrorSeverity.LOW,
            context={"idempotency_key": idempotency_key, **(context or {})},
        )


# =============================================================================
# DATA PROVIDER ERRORS
# =============================================================================

class DataProviderError(AmatisException):
    """Base exception for data provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="data_provider",
            context={"provider": provider, **(context or {})},
        )


class ProviderConnectionError(DataProviderError):
    """Failed to connect to data provider."""
    
    def __init__(
        self,
        provider: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Failed to connect to provider {provider}",
            provider=provider,
            severity=ErrorSeverity.HIGH,
            context={
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {})},
        )
        self.original_error = original_error


class ProviderDataError(DataProviderError):
    """Invalid data received from provider."""
    
    def __init__(
        self,
        provider: str,
        data_description: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Invalid data from {provider}: {data_description}",
            provider=provider,
            severity=ErrorSeverity.HIGH,
            context={"data_description": data_description, **(context or {})},
        )


# =============================================================================
# SIGNAL ENGINE ERRORS
# =============================================================================

class SignalEngineError(AmatisException):
    """Base exception for signal engine errors."""
    
    def __init__(
        self,
        message: str,
        engine: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="signal_engine",
            context={"engine": engine, **(context or {})},
        )


class SignalGenerationError(SignalEngineError):
    """Signal generation failed."""
    
    def __init__(
        self,
        engine: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Signal generation failed in engine {engine}",
            engine=engine,
            severity=ErrorSeverity.MEDIUM,
            context={
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {})},
        )
        self.original_error = original_error


class SignalValidationError(SignalEngineError):
    """Signal validation failed."""
    
    def __init__(
        self,
        signal_id: str,
        validation_error: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Signal validation failed for {signal_id}: {validation_error}",
            severity=ErrorSeverity.HIGH,
            context={
                "signal_id": signal_id,
                "validation_error": validation_error,
                **(context or {})},
        )


# =============================================================================
# SAFETY / KILL SWITCH ERRORS
# =============================================================================

class SafetySystemError(AmatisException):
    """Base exception for safety system errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="safety",
            context=context,
        )


class KillSwitchAuthenticationError(SafetySystemError):
    """Kill switch authentication failed."""
    
    def __init__(
        self,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Kill switch authentication failed: {reason}",
            severity=ErrorSeverity.HIGH,
            context={"reason": reason, **(context or {})},
        )


class KillSwitchSystemError(SafetySystemError):
    """Kill switch system error (not authentication)."""
    
    def __init__(
        self,
        operation: str,
        original_error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Kill switch system error during {operation}",
            severity=ErrorSeverity.CRITICAL,
            context={
                "operation": operation,
                "original_error": str(original_error),
                "original_type": type(original_error).__name__,
                **(context or {})},
        )
        self.original_error = original_error


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================

class ConfigurationError(AmatisException):
    """Configuration error."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            severity=severity,
            component="configuration",
            context={"config_key": config_key, **(context or {})},
        )


class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration value."""
    
    def __init__(
        self,
        config_key: str,
        value: Any,
        expected_type: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Invalid configuration {config_key}: expected {expected_type}, got {type(value).__name__}",
            config_key=config_key,
            severity=ErrorSeverity.HIGH,
            context={
                "value": str(value),
                "expected_type": expected_type,
                **(context or {})},
        )


class MissingConfigurationError(ConfigurationError):
    """Required configuration missing."""
    
    def __init__(
        self,
        config_key: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Required configuration missing: {config_key}",
            config_key=config_key,
            severity=ErrorSeverity.HIGH,
            context=context,
        )
