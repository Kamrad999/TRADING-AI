"""Order state machine for OMS.

Manages order lifecycle states and transitions.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, ClassVar

from amatix.core.observability import get_logger

logger = get_logger(__name__)


class OrderState(Enum):
    """Order lifecycle states.

    State flow:
        CREATED → VALIDATED → SUBMITTED → ACKNOWLEDGED
                                    ↓
                        PARTIALLY_FILLED → FILLED
                                    ↓
                            CANCELLED / REJECTED / EXPIRED
    """

    CREATED = auto()
    VALIDATED = auto()
    SUBMITTED = auto()
    ACKNOWLEDGED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class OrderStateMachine:
    """State machine for order lifecycle management.

    Enforces valid state transitions and emits state change events.

    Valid transitions:
        CREATED → VALIDATED → SUBMITTED → ACKNOWLEDGED
        ACKNOWLEDGED → PARTIALLY_FILLED → FILLED
        ACKNOWLEDGED → CANCELLED (if not filled)
        SUBMITTED → REJECTED
        SUBMITTED → EXPIRED
    """

    # Valid state transitions
    VALID_TRANSITIONS: ClassVar[dict[OrderState, set[OrderState]]] = {
        OrderState.CREATED: {OrderState.VALIDATED, OrderState.CANCELLED, OrderState.REJECTED},
        OrderState.VALIDATED: {OrderState.SUBMITTED, OrderState.CANCELLED},
        OrderState.SUBMITTED: {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        },
        OrderState.ACKNOWLEDGED: {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
        },
        OrderState.PARTIALLY_FILLED: {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        },
        OrderState.FILLED: set(),  # Terminal state
        OrderState.CANCELLED: set(),  # Terminal state
        OrderState.REJECTED: set(),  # Terminal state
        OrderState.EXPIRED: set(),  # Terminal state
    }

    TERMINAL_STATES: ClassVar[set[OrderState]] = {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    }

    def __init__(self, initial_state: OrderState = OrderState.CREATED) -> None:
        """Initialize state machine.

        Args:
            initial_state: Starting state (default: CREATED)
        """
        self._state = initial_state
        self._state_history: list[tuple] = [(initial_state, {})]

    @property
    def current_state(self) -> OrderState:
        """Get current state."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self._state in self.TERMINAL_STATES

    @property
    def state_history(self) -> list[tuple]:
        """Get state transition history."""
        return self._state_history.copy()

    def can_transition(self, new_state: OrderState) -> bool:
        """Check if transition to new state is valid.

        Args:
            new_state: Target state

        Returns:
            True if transition is valid
        """
        if self.is_terminal:
            return False  # Can't leave terminal states

        valid_next = self.VALID_TRANSITIONS.get(self._state, set())
        return new_state in valid_next

    def transition(
        self,
        new_state: OrderState,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Attempt state transition.

        Args:
            new_state: Target state
            metadata: Optional transition metadata

        Returns:
            True if transition succeeded

        Raises:
            InvalidStateTransitionError: If transition is invalid
        """
        if not self.can_transition(new_state):
            logger.error(
                "Invalid state transition",
                current=self._state.name,
                target=new_state.name,
            )
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )

        old_state = self._state
        self._state = new_state

        # Record transition
        transition_data = {
            "from": old_state.name,
            "to": new_state.name,
            **(metadata or {}),
        }
        self._state_history.append((new_state, transition_data))

        logger.debug(
            "Order state transition",
            from_state=old_state.name,
            to_state=new_state.name,
        )

        return True

    def transition_to_validated(self, validation_data: dict | None = None) -> bool:
        """Transition to VALIDATED state."""
        return self.transition(OrderState.VALIDATED, validation_data)

    def transition_to_submitted(self, broker_order_id: str | None = None) -> bool:
        """Transition to SUBMITTED state."""
        metadata = {"broker_order_id": broker_order_id} if broker_order_id else {}
        return self.transition(OrderState.SUBMITTED, metadata)

    def transition_to_acknowledged(self) -> bool:
        """Transition to ACKNOWLEDGED state."""
        return self.transition(OrderState.ACKNOWLEDGED)

    def transition_to_partially_filled(
        self,
        filled_qty: float,
        avg_price: float,
    ) -> bool:
        """Transition to PARTIALLY_FILLED state."""
        return self.transition(
            OrderState.PARTIALLY_FILLED,
            {"filled_qty": filled_qty, "avg_price": avg_price},
        )

    def transition_to_filled(
        self,
        filled_qty: float,
        avg_price: float,
        commission: float = 0.0,
    ) -> bool:
        """Transition to FILLED state."""
        return self.transition(
            OrderState.FILLED,
            {
                "filled_qty": filled_qty,
                "avg_price": avg_price,
                "commission": commission,
            },
        )

    def transition_to_cancelled(self, reason: str = "") -> bool:
        """Transition to CANCELLED state."""
        return self.transition(OrderState.CANCELLED, {"reason": reason})

    def transition_to_rejected(self, reason: str, error_code: str | None = None) -> bool:
        """Transition to REJECTED state."""
        return self.transition(
            OrderState.REJECTED,
            {"reason": reason, "error_code": error_code},
        )

    def transition_to_expired(self, reason: str = "Time in force expired") -> bool:
        """Transition to EXPIRED state."""
        return self.transition(OrderState.EXPIRED, {"reason": reason})

    def get_time_in_state(self) -> float | None:
        """Get time spent in current state (placeholder)."""
        # Would track actual time in production
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state machine."""
        return {
            "current_state": self._state.name,
            "is_terminal": self.is_terminal,
            "history": [
                {"state": state.name, "metadata": metadata}
                for state, metadata in self._state_history
            ],
        }


class InvalidStateTransitionError(Exception):
    """Raised when invalid state transition attempted."""

    pass
