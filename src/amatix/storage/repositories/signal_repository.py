"""Signal repository for AMATIS.

Signal persistence with:
    - Idempotency by signal_id
    - Expiration tracking
    - Feature storage
    - Query by confidence/time
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import async_sessionmaker

from amatix.storage.postgres.models import SignalRecord
from amatix.storage.repositories.base import Repository, SaveResult


class SignalRepository(Repository[SignalRecord]):
    """Repository for trading signal persistence."""
    
    def __init__(self, session_factory: async_sessionmaker) -> None:
        super().__init__(session_factory)
    
    def _get_entity_type(self) -> type:
        return SignalRecord
    
    def _to_dict(self, entity: SignalRecord) -> Dict[str, Any]:
        return {
            "id": str(entity.id),
            "signal_id": entity.signal_id,
            "symbol": entity.symbol,
            "direction": entity.direction,
            "confidence": float(entity.confidence),
        }
    
    async def get_by_signal_id(self, signal_id: str) -> Optional[SignalRecord]:
        """Get signal by signal_id."""
        async with self._session_factory() as session:
            stmt = select(SignalRecord).where(SignalRecord.signal_id == signal_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_recent(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> List[SignalRecord]:
        """Get recent signals with filters."""
        async with self._session_factory() as session:
            stmt = select(SignalRecord)
            
            if symbol:
                stmt = stmt.where(SignalRecord.symbol == symbol)
            if since:
                stmt = stmt.where(SignalRecord.generated_at >= since)
            if min_confidence is not None:
                stmt = stmt.where(SignalRecord.confidence >= min_confidence)
            
            stmt = stmt.order_by(desc(SignalRecord.generated_at)).limit(limit)
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def get_active_signals(
        self,
        symbol: Optional[str] = None,
    ) -> List[SignalRecord]:
        """Get non-expired signals."""
        now = datetime.utcnow()
        
        async with self._session_factory() as session:
            stmt = select(SignalRecord).where(
                and_(
                    SignalRecord.status == "active",
                    or_(
                        SignalRecord.expires_at.is_(None),
                        SignalRecord.expires_at > now,
                    ),
                )
            )
            
            if symbol:
                stmt = stmt.where(SignalRecord.symbol == symbol)
            
            stmt = stmt.order_by(desc(SignalRecord.confidence))
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def expire_old_signals(self, max_age_hours: float = 24.0) -> int:
        """Mark old signals as expired.
        
        Returns:
            Number of signals expired
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(SignalRecord).where(
                    and_(
                        SignalRecord.status == "active",
                        SignalRecord.generated_at < cutoff,
                    )
                )
                result = await session.execute(stmt)
                signals = result.scalars().all()
                
                count = 0
                for signal in signals:
                    signal.status = "expired"
                    count += 1
                
                return count
    
    async def save(
        self,
        entity: SignalRecord,
        idempotency_key: Optional[str] = None,
    ) -> SaveResult:
        """Save signal with idempotency by signal_id."""
        # Use signal_id as idempotency key if not provided
        key = idempotency_key or entity.signal_id
        return await super().save(entity, idempotency_key=key)
    
    async def _check_idempotency(
        self,
        session,
        key: str,
    ) -> Optional[SignalRecord]:
        """Check for existing signal with same signal_id."""
        stmt = select(SignalRecord).where(SignalRecord.signal_id == key)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_signal_stats(
        self,
        since: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get signal statistics."""
        async with self._session_factory() as session:
            stmt = select(func.count(SignalRecord.id))
            
            if since:
                stmt = stmt.where(SignalRecord.generated_at >= since)
            if symbol:
                stmt = stmt.where(SignalRecord.symbol == symbol)
            
            total_result = await session.execute(stmt)
            total = total_result.scalar()
            
            # Count by direction
            dir_stmt = select(
                SignalRecord.direction,
                func.count(SignalRecord.id),
            ).group_by(SignalRecord.direction)
            
            if since:
                dir_stmt = dir_stmt.where(SignalRecord.generated_at >= since)
            if symbol:
                dir_stmt = dir_stmt.where(SignalRecord.symbol == symbol)
            
            dir_result = await session.execute(dir_stmt)
            by_direction = {row[0]: row[1] for row in dir_result.all()}
            
            # Average confidence
            conf_stmt = select(func.avg(SignalRecord.confidence))
            if since:
                conf_stmt = conf_stmt.where(SignalRecord.generated_at >= since)
            if symbol:
                conf_stmt = conf_stmt.where(SignalRecord.symbol == symbol)
            
            conf_result = await session.execute(conf_stmt)
            avg_confidence = conf_result.scalar()
            
            return {
                "total": total,
                "by_direction": by_direction,
                "avg_confidence": float(avg_confidence) if avg_confidence else 0.0,
            }
