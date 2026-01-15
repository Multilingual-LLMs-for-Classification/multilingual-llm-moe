"""
Configuration dataclasses for the hierarchical routing system.

Provides type-safe configuration with defaults for all routing components.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LanguageDetectorConfig:
    """Configuration for language detection."""
    registry_path: Optional[Path] = None
    fasttext_model_path: Optional[Path] = None
    chunk_size: int = 1000


@dataclass
class DomainClassifierConfig:
    """Configuration for domain classification."""
    model_name: str = "xlm-roberta-base"
    model_dir: Optional[Path] = None
    max_len: int = 128
    alpha_proto: float = 0.30
    proto_temp: float = 10.0

    # Training parameters
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5
    freeze_encoder: bool = True
    class_weighting: bool = True


@dataclass
class QLearningConfig:
    """Configuration for Q-learning task classifier."""
    encoder_name: str = "xlm-roberta-base"
    model_dir: Optional[Path] = None
    max_len: int = 128
    batch_size: int = 16
    lr: float = 1e-5
    epochs: int = 1
    eps_start: float = 0.2
    eps_end: float = 0.01
    eps_decay_steps: int = 10000

    # Training parameters
    val_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training phase."""
    data_path: str = "train1.json"
    enable_training: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation phase."""
    test_data_path: str = "test2_grouped_languages_flat.json"
    test_n: Optional[int] = None  # None = use all samples
    output_path: str = "predictions_with_raw_responses.csv"
    enable_evaluation: bool = True


@dataclass
class RouterSystemConfig:
    """
    Main configuration for the hierarchical prompt routing system.

    This is the top-level configuration that aggregates all component configs.
    Can be loaded from JSON or created programmatically.
    """

    # Core paths
    expert_registry_path: Path = field(
        default_factory=lambda: Path("src/models/experts/config/experts_registry.json")
    )
    domain_tasks_path: Path = field(
        default_factory=lambda: Path("src/models/experts/config/domain_tasks.json")
    )
    model_config_path: Optional[Path] = field(
        default_factory=lambda: Path("src/models/experts/config/model_config.json")
    )

    # Component configurations
    language_config: LanguageDetectorConfig = field(
        default_factory=LanguageDetectorConfig
    )
    domain_config: DomainClassifierConfig = field(
        default_factory=DomainClassifierConfig
    )
    qlearning_config: QLearningConfig = field(
        default_factory=QLearningConfig
    )

    # Training and evaluation configs
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    evaluation: EvaluationConfig = field(
        default_factory=EvaluationConfig
    )

    @classmethod
    def from_json(cls, path: Path) -> "RouterSystemConfig":
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            RouterSystemConfig instance

        Example JSON structure:
        ```json
        {
            "expert_registry_path": "src/models/experts/config/experts_registry.json",
            "domain_config": {
                "model_name": "xlm-roberta-base",
                "epochs": 1,
                "batch_size": 32
            },
            "training": {
                "data_path": "train1.json"
            },
            "evaluation": {
                "test_data_path": "test2.json",
                "test_n": 1000
            }
        }
        ```
        """
        with open(path) as f:
            data = json.load(f)

        # Convert string paths to Path objects
        if "expert_registry_path" in data:
            data["expert_registry_path"] = Path(data["expert_registry_path"])
        if "domain_tasks_path" in data:
            data["domain_tasks_path"] = Path(data["domain_tasks_path"])
        if "model_config_path" in data and data["model_config_path"]:
            data["model_config_path"] = Path(data["model_config_path"])

        # Recursively construct nested dataclasses
        if "language_config" in data:
            data["language_config"] = LanguageDetectorConfig(**data["language_config"])
        if "domain_config" in data:
            data["domain_config"] = DomainClassifierConfig(**data["domain_config"])
        if "qlearning_config" in data:
            data["qlearning_config"] = QLearningConfig(**data["qlearning_config"])
        if "training" in data:
            data["training"] = TrainingConfig(**data["training"])
        if "evaluation" in data:
            data["evaluation"] = EvaluationConfig(**data["evaluation"])

        return cls(**data)

    def to_json(self, path: Path):
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON configuration

        Example usage:
            config = RouterSystemConfig()
            config.to_json(Path("my_config.json"))
        """
        # Convert to dict and handle Path objects
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

    @classmethod
    def from_legacy_dict(cls, config_dict: Dict[str, Any]) -> "RouterSystemConfig":
        """
        Create config from legacy dictionary format (backward compatibility).

        Args:
            config_dict: Legacy configuration dictionary

        Returns:
            RouterSystemConfig instance

        Example legacy format:
        ```python
        config_dict = {
            "training": {
                "data_path": "train1.json",
                "domain_classifier": {
                    "epochs": 1,
                    "batch_size": 32,
                    ...
                },
                "q_routers": {
                    "val_split": 0.1
                }
            },
            "evaluation": {
                "test_data_path": "test2.json",
                "test_n": 1000,
                "output_path": "predictions.csv"
            }
        }
        ```
        """
        # Extract domain classifier config from training section
        domain_clf_config = config_dict.get("training", {}).get("domain_classifier", {})
        domain_config = DomainClassifierConfig(
            epochs=domain_clf_config.get("epochs", 1),
            batch_size=domain_clf_config.get("batch_size", 32),
            lr=domain_clf_config.get("lr", 2e-5),
            freeze_encoder=domain_clf_config.get("freeze_encoder", True),
            class_weighting=domain_clf_config.get("class_weighting", True)
        )

        # Extract Q-learning config from training section
        q_routers_config = config_dict.get("training", {}).get("q_routers", {})
        qlearning_config = QLearningConfig(
            val_split=q_routers_config.get("val_split", 0.1)
        )

        # Extract training config
        training_section = config_dict.get("training", {})
        training = TrainingConfig(
            data_path=training_section.get("data_path", "train1.json")
        )

        # Extract evaluation config
        eval_section = config_dict.get("evaluation", {})
        evaluation = EvaluationConfig(
            test_data_path=eval_section.get("test_data_path", "test2_grouped_languages_flat.json"),
            test_n=eval_section.get("test_n"),
            output_path=eval_section.get("output_path", "predictions_with_raw_responses.csv")
        )

        return cls(
            domain_config=domain_config,
            qlearning_config=qlearning_config,
            training=training,
            evaluation=evaluation
        )


# Example usage and defaults
if __name__ == "__main__":
    # Create default config
    config = RouterSystemConfig()

    # Save to JSON
    config.to_json(Path("router_config_default.json"))
    print("✅ Created default configuration file: router_config_default.json")

    # Load from JSON
    loaded_config = RouterSystemConfig.from_json(Path("router_config_default.json"))
    print(f"✅ Loaded configuration with {loaded_config.domain_config.epochs} epochs")

    # Example: Create custom config
    custom_config = RouterSystemConfig(
        domain_config=DomainClassifierConfig(
            epochs=3,
            batch_size=64,
            lr=1e-5
        ),
        evaluation=EvaluationConfig(
            test_n=500,
            output_path="my_predictions.csv"
        )
    )

    print(f"\n✅ Custom config: {custom_config.domain_config.epochs} epochs, "
          f"{custom_config.evaluation.test_n} test samples")
