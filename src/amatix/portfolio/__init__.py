"""Portfolio intelligence foundation for AMATIS.

Institutional-grade portfolio management with:
    - Exposure aggregation
    - Risk parity allocation
    - Dynamic position sizing
    - Capital allocation engine
"""

from amatix.portfolio.manager import PortfolioManager
from amatix.portfolio.analytics import PortfolioAnalytics
from amatix.portfolio.allocation import CapitalAllocator
from amatix.portfolio.sizing import PositionSizer
from amatix.portfolio.exposure import ExposureMonitor

__all__ = [
    "PortfolioManager",
    "PortfolioAnalytics",
    "CapitalAllocator",
    "PositionSizer",
    "ExposureMonitor",
]
