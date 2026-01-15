"""
Training script for the hierarchical routing system.

This script trains the domain classifier and Q-learning task routers
on labeled training data.
"""

import json
from pathlib import Path
from typing import List, Dict

from components import PromptRoutingSystem
from router_config import RouterSystemConfig


def load_training_data(filepath: str, limit: int = None) -> List[Dict]:
    """
    Load training data from JSON file.

    Args:
        filepath: Path to training data JSON file
        limit: Optional limit on number of samples (None = use all)

    Returns:
        List of training samples with 'prompt', 'domain', 'task', 'label' fields
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    return data


def train_domain_classifier(system: PromptRoutingSystem,
                            training_data: List[Dict],
                            config: RouterSystemConfig):
    """
    Train the domain classifier.

    Args:
        system: PromptRoutingSystem instance
        training_data: List of training samples
        config: System configuration
    """
    print("\n" + "=" * 80)
    print("TRAINING DOMAIN CLASSIFIER")
    print("=" * 80)

    print(f"\n📊 Training parameters:")
    print(f"   - Epochs: {config.domain_config.epochs}")
    print(f"   - Batch size: {config.domain_config.batch_size}")
    print(f"   - Learning rate: {config.domain_config.lr}")
    print(f"   - Freeze encoder: {config.domain_config.freeze_encoder}")
    print(f"   - Class weighting: {config.domain_config.class_weighting}")

    print(f"\n🚀 Starting training...")
    system.train_domain_classifier(
        training_data,
        epochs=config.domain_config.epochs,
        batch_size=config.domain_config.batch_size,
        lr=config.domain_config.lr,
        freeze_encoder=config.domain_config.freeze_encoder,
        class_weighting=config.domain_config.class_weighting
    )

    print("✅ Domain classifier training complete")


def train_q_routers(system: PromptRoutingSystem,
                   training_data: List[Dict],
                   config: RouterSystemConfig):
    """
    Train Q-learning task routers for each domain.

    Args:
        system: PromptRoutingSystem instance
        training_data: List of training samples
        config: System configuration
    """
    print("\n" + "=" * 80)
    print("TRAINING Q-LEARNING TASK ROUTERS")
    print("=" * 80)

    print(f"\n📊 Training parameters:")
    print(f"   - Validation split: {config.qlearning_config.val_split}")
    print(f"   - Epsilon start: {config.qlearning_config.eps_start}")
    print(f"   - Epsilon end: {config.qlearning_config.eps_end}")
    print(f"   - Epsilon decay steps: {config.qlearning_config.eps_decay_steps}")

    print(f"\n🚀 Starting training...")
    system.train_q_routers(training_data)

    print("✅ Q-learning routers training complete")


def save_models(system: PromptRoutingSystem):
    """
    Save all trained models to disk.

    Args:
        system: PromptRoutingSystem instance with trained models
    """
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    print("\n💾 Saving trained models...")
    system.save_all_models()
    print("✅ All models saved successfully")


def main(config: RouterSystemConfig):
    """
    Main training function.

    Args:
        config: System configuration
    """
    print("=" * 80)
    print("HIERARCHICAL ROUTING SYSTEM - TRAINING MODE")
    print("=" * 80)

    # Initialize routing system
    print("\n🔧 Initializing routing system...")
    system = PromptRoutingSystem()

    # Display system info
    stats = system.get_system_stats()
    print(f"✅ System initialized:")
    print(f"   - Domains: {stats['total_domains']}")
    print(f"   - Tasks: {stats['total_tasks']}")
    print(f"   - Supported languages: {stats['supported_languages']}")
    print(f"   - Available domains: {stats['domains']}")

    # Load training data
    print(f"\n📂 Loading training data from: {config.training.data_path}")
    training_data = load_training_data(config.training.data_path)
    print(f"✅ Loaded {len(training_data)} training samples")

    # Analyze training data
    domains = set(item['domain'] for item in training_data if isinstance(item, dict))
    tasks = set(item['task'] for item in training_data if isinstance(item, dict))
    languages = set(item.get('language', 'unknown') for item in training_data if isinstance(item, dict))

    print(f"\n📊 Training data statistics:")
    print(f"   - Unique domains: {len(domains)} ({', '.join(sorted(domains))})")
    print(f"   - Unique tasks: {len(tasks)} ({', '.join(sorted(tasks))})")
    print(f"   - Unique languages: {len(languages)} ({', '.join(sorted(languages))})")

    # Train domain classifier
    if config.training.enable_training:
        train_domain_classifier(system, training_data, config)
        train_q_routers(system, training_data, config)
        save_models(system)
    else:
        print("\n⚠️  Training disabled in configuration. Skipping training phase.")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)

    return system


if __name__ == "__main__":
    import sys

    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "router_config.json"

    try:
        config = RouterSystemConfig.from_json(Path(config_path))
        print(f"✅ Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"⚠️  Config file '{config_path}' not found. Using defaults.")
        config = RouterSystemConfig()

    # Run training
    main(config)
