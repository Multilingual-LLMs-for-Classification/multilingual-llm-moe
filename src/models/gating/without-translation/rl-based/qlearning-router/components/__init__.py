"""
Routing system components.

This package contains the modular components of the hierarchical routing system:
- LanguageDetector: FastText-based language identification
- DomainClassifier: XLM-RoBERTa-based domain classification
- QLearningTaskClassifier: Q-learning task router
- PromptRoutingSystem: Main orchestrator

All components can be imported directly from this package:
    from components import LanguageDetector, DomainClassifier, PromptRoutingSystem
"""

from .language_detector import LanguageDetector
from .domain_classifier import DomainClassifier, _DomainDataset
from .q_learning_router import (
    TransformersEncoder,
    QRouter,
    DomainTaskDataset,
    QLearningTaskClassifier
)
from .routing_system import PromptRoutingSystem

__all__ = [
    # Language detection
    "LanguageDetector",

    # Domain classification
    "DomainClassifier",
    "_DomainDataset",

    # Q-learning task routing
    "TransformersEncoder",
    "QRouter",
    "DomainTaskDataset",
    "QLearningTaskClassifier",

    # Main routing system
    "PromptRoutingSystem",
]
