"""PostgreSQL connection management.

Async SQLAlchemy engine with connection pooling.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from amatix.core.observability import get_logger

logger = get_logger(__name__)


class PostgresEngine:
    """PostgreSQL connection manager.
    
    Provides:
        - Async connection pooling
        - Session management
        - Health checking
    
    Example:
        >>> db = PostgresEngine("postgresql+asyncpg://user:pass@localhost/amatix")
        >>> await db.connect()
        >>> 
        >>> async with db.session() as session:
        ...     result = await session.execute(query)
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        """Initialize PostgreSQL engine.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Max overflow connections
        """
        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker] = None
    
    async def connect(self) -> None:
        """Create database engine and connection pool."""
        logger.info("Connecting to PostgreSQL")
        
        self._engine = create_async_engine(
            self._database_url,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            echo=False,
        )
        
        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Test connection
        async with self._engine.connect() as conn:
            await conn.execute("SELECT 1")
        
        logger.info("PostgreSQL connected")
    
    async def disconnect(self) -> None:
        """Close all connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("PostgreSQL disconnected")
    
    def session(self) -> AsyncSession:
        """Get new database session."""
        if not self._session_maker:
            raise RuntimeError("Database not connected")
        return self._session_maker()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        if not self._engine:
            return False
        
        try:
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
