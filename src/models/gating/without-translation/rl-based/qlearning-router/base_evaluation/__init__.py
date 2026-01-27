"""
Base Model Evaluation Module

This module provides tools for evaluating a single base model WITHOUT any
fine-tuned adapters, for comparison with the adapter-based routing system.

Usage:
    python main.py --base-model llama-2-7b-hf --test-data ../test2.json
"""

from .base_config import BaseEvaluationConfig, BaseModelConfig, GenerationConfig
from .base_model_pool import BaseModelPool
