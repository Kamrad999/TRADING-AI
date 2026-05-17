"""Operational safety controls for AMATIS.

Production-grade safety systems:
    - Authenticated kill switch
    - Startup validation
    - Runtime invariant checks
    - Emergency shutdown modes
"""

from amatix.safety.kill_switch import KillSwitch, KillSwitchAuth
from amatix.safety.validator import StartupValidator, SafetyInvariant
from amatix.safety.modes import SafeMode, EmergencyShutdown

__all__ = [
    "KillSwitch",
    "KillSwitchAuth",
    "StartupValidator",
    "SafetyInvariant",
    "SafeMode",
    "EmergencyShutdown",
]
