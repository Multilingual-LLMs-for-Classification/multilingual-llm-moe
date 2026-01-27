"""
Configuration dataclasses for base model evaluation.

This module provides configuration for evaluating a single base model
without any adapters, for comparison with the adapter-based routing system.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class BaseModelConfig:
    """Configuration for the base model to evaluate."""
    # Base model to use (key from experts_registry.json)
    base_model_key: str = "llama-2-7b-hf"

    # HuggingFace model name (will be resolved from registry)
    hf_name: Optional[str] = None

    # Quantization settings
    load_in_4bit: bool = True
    device_map: str = "auto"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


@dataclass
class TaskConfig:
    """Configuration for task-specific settings."""
    # Task name (rating, news, pii)
    name: str = "rating"

    # Label set for the task
    label_set: List[str] = field(default_factory=list)

    # Whether to use strict label decoding
    strict_label_decoding: bool = True

    # Task-specific generation overrides
    generation: Optional[GenerationConfig] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation phase."""
    test_data_path: str = "../test2_grouped_languages_flat.json"
    test_n: Optional[int] = None  # None = use all samples
    output_path: str = "base_model_predictions.csv"


@dataclass
class BaseEvaluationConfig:
    """
    Main configuration for base model evaluation.

    This configuration is used to evaluate a single base model without adapters
    against the same test data used by the routing system.
    """

    # Core paths
    expert_registry_path: Path = field(
        default_factory=lambda: Path("src/models/experts/config/experts_registry.json")
    )

    # Base model configuration
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)

    # Generation configuration
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Tasks to evaluate
    tasks: List[str] = field(default_factory=lambda: ["rating", "news", "pii"])

    @classmethod
    def from_json(cls, path: Path) -> "BaseEvaluationConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Convert string paths to Path objects
        if "expert_registry_path" in data:
            data["expert_registry_path"] = Path(data["expert_registry_path"])

        # Recursively construct nested dataclasses
        if "base_model" in data:
            data["base_model"] = BaseModelConfig(**data["base_model"])
        if "generation" in data:
            data["generation"] = GenerationConfig(**data["generation"])
        if "evaluation" in data:
            data["evaluation"] = EvaluationConfig(**data["evaluation"])

        return cls(**data)

    def to_json(self, path: Path):
        """Save configuration to JSON file."""
        data = asdict(self)

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        data = convert_paths(data)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Default configuration instance
if __name__ == "__main__":
    config = BaseEvaluationConfig()
    config.to_json(Path("base_eval_config.json"))
    print("Created default configuration: base_eval_config.json")
