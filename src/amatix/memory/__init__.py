"""AMATIS Memory - Decision journaling and explainability.

This module provides the foundation for:
    - Decision journaling (every decision recorded)
    - Explainability (why decisions were made)
    - Audit trails (for compliance and debugging)
    - Reinforcement learning datasets
    - Alpha decay detection
"""

from amatix.memory.decision_journal import DecisionJournal, DecisionRecord
from amatix.memory.feature_attribution import FeatureAttribution

__all__ = [
    "DecisionJournal",
    "DecisionRecord",
    "FeatureAttribution",
]
