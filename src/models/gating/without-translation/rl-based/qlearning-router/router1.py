"""
Hierarchical Routing System - Backward Compatibility Module.

This module maintains backward compatibility by re-exporting all components
from the new modular structure. Existing code that imports from router1.py
will continue to work without changes.

New code should import directly from the components package:
    from components import PromptRoutingSystem, LanguageDetector, etc.

Legacy import (still supported):
    from router1 import PromptRoutingSystem
"""

# Re-export all components for backward compatibility
from components import (
    # Language detection
    LanguageDetector,

    # Domain classification
    DomainClassifier,
    _DomainDataset,

    # Q-learning task routing
    TransformersEncoder,
    QRouter,
    DomainTaskDataset,
    QLearningTaskClassifier,

    # Main routing system
    PromptRoutingSystem,
)

# Maintain __all__ for explicit exports
__all__ = [
    "LanguageDetector",
    "DomainClassifier",
    "_DomainDataset",
    "TransformersEncoder",
    "QRouter",
    "DomainTaskDataset",
    "QLearningTaskClassifier",
    "PromptRoutingSystem",
]
