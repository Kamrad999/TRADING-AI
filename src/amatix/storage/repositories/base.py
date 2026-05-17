"""Base repository abstractions for AMATIS.

Implements institutional patterns:
    - Unit of Work pattern
    - Repository pattern
    - Optimistic locking
    - Idempotency keys
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import joinedload
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError, StaleDataError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConcurrencyError(Exception):
    """Raised when optimistic locking fails."""
    pass


class IdempotencyError(Exception):
    """Raised when idempotency check fails."""
    pass


@dataclass
class SaveResult:
    """Result of a save operation."""
    success: bool
    id: Optional[UUID] = None
    version: Optional[int] = None
    error: Optional[str] = None
    is_duplicate: bool = False


class Repository(ABC, Generic[T]):
    """Base repository with institutional-grade data access.
    
    Features:
        - Async session management
        - Optimistic locking
        - Idempotency key support
        - Automatic retry on deadlock
        - Audit logging hooks
    """
    
    def __init__(
        self,
        session_factory: async_sessionmaker,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> None:
        self._session_factory = session_factory
        self._max_retries = max_retries
        self._retry_delay = retry_delay
    
    @abstractmethod
    def _get_entity_type(self) -> type:
        """Return the SQLAlchemy entity type."""
        pass
    
    @abstractmethod
    def _to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for audit logging."""
        pass
    
    async def get_by_id(
        self,
        entity_id: UUID,
        options: Optional[List[Any]] = None,
    ) -> Optional[T]:
        """Get entity by ID with optional eager loading."""
        async with self._session_factory() as session:
            stmt = select(self._get_entity_type()).where(
                self._get_entity_type().id == entity_id
            )
            if options:
                for opt in options:
                    stmt = stmt.options(opt)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_all(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[T]:
        """Get all entities with pagination."""
        async with self._session_factory() as session:
            stmt = (
                select(self._get_entity_type())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def save(
        self,
        entity: T,
        idempotency_key: Optional[str] = None,
    ) -> SaveResult:
        """Save entity with optimistic locking and idempotency.
        
        Args:
            entity: Entity to save
            idempotency_key: Optional key for duplicate prevention
        
        Returns:
            SaveResult with success status and metadata
        """
        for attempt in range(self._max_retries):
            try:
                return await self._save_with_session(entity, idempotency_key)
            except StaleDataError as e:
                logger.warning(
                    f"Optimistic lock failed (attempt {attempt + 1}): {e}"
                )
                if attempt == self._max_retries - 1:
                    return SaveResult(
                        success=False,
                        error=f"Concurrency conflict after {self._max_retries} retries"
                    )
                await asyncio.sleep(self._retry_delay * (2 ** attempt))
            except IntegrityError as e:
                if idempotency_key and "duplicate" in str(e).lower():
                    logger.info(f"Idempotent duplicate detected: {idempotency_key}")
                    return SaveResult(success=True, is_duplicate=True)
                logger.error(f"Integrity error: {e}")
                return SaveResult(success=False, error=str(e))
            except Exception as e:
                logger.error(f"Save failed: {e}")
                return SaveResult(success=False, error=str(e))
    
    async def _save_with_session(
        self,
        entity: T,
        idempotency_key: Optional[str] = None,
    ) -> SaveResult:
        """Internal save with fresh session."""
        async with self._session_factory() as session:
            async with session.begin():
                # Check idempotency if key provided
                if idempotency_key:
                    existing = await self._check_idempotency(
                        session, idempotency_key
                    )
                    if existing:
                        return SaveResult(
                            success=True,
                            id=existing.id,
                            is_duplicate=True,
                        )
                
                # Add/update entity
                session.add(entity)
                await session.flush()
                
                # Extract ID and version
                entity_id = getattr(entity, "id", None)
                version = getattr(entity, "version", None)
                
                # Audit logging hook
                await self._log_audit(session, "save", entity)
                
                return SaveResult(
                    success=True,
                    id=entity_id,
                    version=version,
                )
    
    async def _check_idempotency(
        self,
        session: AsyncSession,
        key: str,
    ) -> Optional[T]:
        """Check if entity with idempotency key already exists."""
        # Override in specific repositories
        return None
    
    async def _log_audit(
        self,
        session: AsyncSession,
        operation: str,
        entity: T,
    ) -> None:
        """Log audit trail entry."""
        # Hook for audit logging - can be overridden or extended
        pass
    
    async def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        for attempt in range(self._max_retries):
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        entity_type = self._get_entity_type()
                        stmt = delete(entity_type).where(
                            entity_type.id == entity_id
                        )
                        result = await session.execute(stmt)
                        return result.rowcount > 0
            except Exception as e:
                logger.error(f"Delete failed (attempt {attempt + 1}): {e}")
                if attempt == self._max_retries - 1:
                    return False
                await asyncio.sleep(self._retry_delay * (2 ** attempt))
        return False


class UnitOfWork:
    """Unit of Work pattern for transactional consistency.
    
    Ensures atomic operations across multiple repositories:
        - All succeed or all fail
        - Optimistic locking across entities
        - Event publication after commit
    
    Example:
        async with UnitOfWork(session_factory) as uow:
            order = await uow.orders.get_by_id(order_id)
            fills = await uow.fills.get_by_order_id(order_id)
            
            order.status = "filled"
            await uow.orders.save(order)
            
            # Commit happens automatically on exit
    """
    
    def __init__(
        self,
        session_factory: async_sessionmaker,
    ) -> None:
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
        self._committed = False
    
    async def __aenter__(self) -> UnitOfWork:
        """Enter context, create session."""
        self._session = self._session_factory()
        self._transaction = await self._session.begin()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, commit or rollback."""
        if exc_type is None:
            # No exception - commit
            try:
                await self._transaction.commit()
                self._committed = True
            except Exception as e:
                logger.error(f"Commit failed: {e}")
                await self._transaction.rollback()
                raise
        else:
            # Exception occurred - rollback
            await self._transaction.rollback()
        
        await self._session.close()
    
    @property
    def session(self) -> AsyncSession:
        """Get current session."""
        if self._session is None:
            raise RuntimeError("Unit of Work not in context")
        return self._session
    
    @property
    def is_committed(self) -> bool:
        """Check if transaction was committed."""
        return self._committed
    
    async def commit(self) -> None:
        """Explicit commit (usually handled by context manager)."""
        if self._transaction:
            await self._transaction.commit()
            self._committed = True
    
    async def rollback(self) -> None:
        """Explicit rollback."""
        if self._transaction:
            await self._transaction.rollback()


@asynccontextmanager
async def transaction_scope(session_factory: async_sessionmaker):
    """Provide a transactional scope around a series of operations.
    
    Usage:
        async with transaction_scope(session_factory) as session:
            # All operations in this block are atomic
            session.add(entity1)
            session.add(entity2)
            # Commit happens automatically on success
    """
    session = session_factory()
    async with session.begin():
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
