"""Position repository for AMATIS.

Portfolio position persistence with:
    - Atomic position updates
    - P&L tracking
    - Exposure aggregation
    - Historical snapshots
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import async_sessionmaker

from amatix.storage.postgres.models import PositionRecord, PortfolioSnapshot
from amatix.storage.repositories.base import Repository, SaveResult


class PositionRepository(Repository[PositionRecord]):
    """Repository for position management.
    
    Features:
        - Atomic position updates with P&L
        - Realized/unrealized P&L tracking
        - Exposure percentage calculation
    """
    
    def __init__(self, session_factory: async_sessionmaker) -> None:
        super().__init__(session_factory)
    
    def _get_entity_type(self) -> type:
        return PositionRecord
    
    def _to_dict(self, entity: PositionRecord) -> Dict[str, Any]:
        return {
            "id": str(entity.id),
            "symbol": entity.symbol,
            "side": entity.side,
            "quantity": str(entity.quantity),
            "exposure_pct": float(entity.exposure_pct) if entity.exposure_pct else 0.0,
        }
    
    async def get_by_symbol(
        self,
        symbol: str,
        account_id: str = "default",
    ) -> Optional[PositionRecord]:
        """Get position by symbol."""
        async with self._session_factory() as session:
            stmt = select(PositionRecord).where(
                and_(
                    PositionRecord.symbol == symbol,
                    PositionRecord.account_id == account_id,
                )
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_all_positions(
        self,
        account_id: str = "default",
        active_only: bool = True,
    ) -> List[PositionRecord]:
        """Get all positions for account."""
        async with self._session_factory() as session:
            stmt = select(PositionRecord).where(
                PositionRecord.account_id == account_id
            )
            
            if active_only:
                stmt = stmt.where(PositionRecord.side != "flat")
            
            stmt = stmt.order_by(desc(PositionRecord.market_value))
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def get_exposure_by_sector(
        self,
        account_id: str = "default",
    ) -> Dict[str, Decimal]:
        """Get exposure aggregated by sector."""
        # Note: This would require sector data in PositionRecord
        # For now, return by symbol as placeholder
        positions = await self.get_all_positions(account_id)
        
        exposure = {}
        for pos in positions:
            if pos.market_value:
                sector = pos.metadata.get("sector", "unknown") if pos.metadata else "unknown"
                exposure[sector] = exposure.get(sector, Decimal("0")) + pos.market_value
        
        return exposure
    
    async def update_position_pnl(
        self,
        symbol: str,
        current_price: Decimal,
        account_id: str = "default",
    ) -> Optional[PositionRecord]:
        """Update position unrealized P&L based on current price."""
        async with self._session_factory() as session:
            async with session.begin():
                position = await self.get_by_symbol(symbol, account_id)
                if not position:
                    return None
                
                if position.side == "flat" or position.quantity == 0:
                    return position
                
                # Calculate market value
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                
                # Calculate unrealized P&L
                if position.side == "long":
                    position.unrealized_pnl = (
                        position.market_value - 
                        (position.avg_entry_price * position.quantity)
                    )
                else:  # short
                    position.unrealized_pnl = (
                        (position.avg_entry_price * position.quantity) - 
                        position.market_value
                    )
                
                position.total_pnl = position.unrealized_pnl + position.realized_pnl
                position.last_updated = datetime.utcnow()
                
                return position
    
    async def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        account_id: str = "default",
    ) -> Optional[PositionRecord]:
        """Close a position and calculate realized P&L."""
        async with self._session_factory() as session:
            async with session.begin():
                position = await self.get_by_symbol(symbol, account_id)
                if not position:
                    return None
                
                if position.side == "flat":
                    return position  # Already closed
                
                # Calculate realized P&L
                exit_value = position.quantity * exit_price
                entry_value = position.quantity * position.avg_entry_price
                
                if position.side == "long":
                    trade_pnl = exit_value - entry_value
                else:
                    trade_pnl = entry_value - exit_value
                
                position.realized_pnl += trade_pnl
                position.total_pnl = position.realized_pnl + position.unrealized_pnl
                position.unrealized_pnl = Decimal("0")
                
                # Mark as closed
                position.side = "flat"
                position.quantity = Decimal("0")
                position.closed_at = datetime.utcnow()
                position.last_updated = datetime.utcnow()
                
                return position
    
    async def get_position_summary(
        self,
        account_id: str = "default",
    ) -> Dict[str, Any]:
        """Get summary of all positions."""
        positions = await self.get_all_positions(account_id)
        
        if not positions:
            return {
                "total_positions": 0,
                "long_positions": 0,
                "short_positions": 0,
                "total_market_value": Decimal("0"),
                "total_unrealized_pnl": Decimal("0"),
                "total_realized_pnl": Decimal("0"),
            }
        
        long_count = sum(1 for p in positions if p.side == "long")
        short_count = sum(1 for p in positions if p.side == "short")
        
        total_value = sum(
            (p.market_value or Decimal("0")) for p in positions
        )
        
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_realized = sum(p.realized_pnl for p in positions)
        
        return {
            "total_positions": len(positions),
            "long_positions": long_count,
            "short_positions": short_count,
            "total_market_value": total_value,
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
        }


class PortfolioSnapshotRepository(Repository[PortfolioSnapshot]):
    """Repository for portfolio snapshots over time."""
    
    def __init__(self, session_factory: async_sessionmaker) -> None:
        super().__init__(session_factory)
    
    def _get_entity_type(self) -> type:
        return PortfolioSnapshot
    
    async def get_latest(
        self,
        account_id: str = "default",
    ) -> Optional[PortfolioSnapshot]:
        """Get most recent portfolio snapshot."""
        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.account_id == account_id)
                .order_by(desc(PortfolioSnapshot.timestamp))
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_range(
        self,
        start: datetime,
        end: datetime,
        account_id: str = "default",
    ) -> List[PortfolioSnapshot]:
        """Get snapshots in time range."""
        async with self._session_factory() as session:
            stmt = (
                select(PortfolioSnapshot)
                .where(
                    and_(
                        PortfolioSnapshot.account_id == account_id,
                        PortfolioSnapshot.timestamp >= start,
                        PortfolioSnapshot.timestamp <= end,
                    )
                )
                .order_by(PortfolioSnapshot.timestamp)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def get_drawdown_analysis(
        self,
        days: int = 30,
        account_id: str = "default",
    ) -> Dict[str, Any]:
        """Analyze drawdown over period."""
        since = datetime.utcnow() - timedelta(days=days)
        snapshots = await self.get_range(since, datetime.utcnow(), account_id)
        
        if not snapshots:
            return {"error": "No snapshots available"}
        
        values = [s.total_value for s in snapshots]
        peak = max(values)
        trough = min(values)
        current = values[-1]
        
        max_drawdown = (peak - trough) / peak if peak > 0 else Decimal("0")
        current_drawdown = (peak - current) / peak if peak > 0 else Decimal("0")
        
        return {
            "period_days": days,
            "peak_value": peak,
            "trough_value": trough,
            "current_value": current,
            "max_drawdown_pct": float(max_drawdown),
            "current_drawdown_pct": float(current_drawdown),
            "snapshot_count": len(snapshots),
        }
