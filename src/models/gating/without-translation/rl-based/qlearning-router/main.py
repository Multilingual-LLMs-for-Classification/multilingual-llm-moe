"""
Main CLI entry point for the hierarchical routing system.

This script provides a unified interface for training, evaluation, and
combined workflows.
"""

import argparse
import sys
from pathlib import Path

from router_config import RouterSystemConfig
import train
import evaluate


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Prompt Routing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and evaluate with default config
  python main.py --mode all

  nohup python main.py > output.log 2>&1 &

  # Evaluation only with N samples
nohup python main.py --mode eval --test-n 100 > output.log 2>&1 &

# Train + evaluate with N samples
nohup python main.py --mode all --test-n 100 > output.log 2>&1 &

# With custom paths
nohup python main.py --mode eval --test-n 500 --test-data /path/to/test.json --output results.csv > output.log 2>&1 &

  # Train only
  python main.py --mode train --config my_config.json

  # Evaluate only (uses pre-trained models)
  python main.py --mode eval --test-data test.json --output results.csv

  # Custom configuration
  python main.py --mode all --train-data train.json --test-data test.json
        """
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['train', 'eval', 'all'],
        default='all',
        help='Run mode: train (training only), eval (evaluation only), or all (both)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration JSON file (default: router_config.json)'
    )

    # Data paths
    parser.add_argument(
        '--train-data',
        type=Path,
        help='Override training data path from config'
    )
    parser.add_argument(
        '--test-data',
        type=Path,
        help='Override test data path from config'
    )
    parser.add_argument(
        '--test-n',
        type=int,
        help='Override number of test samples (None = all)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=Path,
        help='Override output CSV path from config'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            config = RouterSystemConfig.from_json(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        except FileNotFoundError:
            print(f"❌ Config file not found: {args.config}")
            sys.exit(1)
    else:
        # Try default location
        default_config = Path("router_config.json")
        if default_config.exists():
            config = RouterSystemConfig.from_json(default_config)
            print(f"✅ Loaded configuration from: {default_config}")
        else:
            print("⚠️  No config file found. Using default configuration.")
            config = RouterSystemConfig()

    # Apply command-line overrides
    if args.train_data:
        config.training.data_path = str(args.train_data)
    if args.test_data:
        config.evaluation.test_data_path = str(args.test_data)
    if args.test_n is not None:
        config.evaluation.test_n = args.test_n
    if args.output:
        config.evaluation.output_path = str(args.output)
    if args.epochs is not None:
        config.domain_config.epochs = args.epochs
    if args.batch_size is not None:
        config.domain_config.batch_size = args.batch_size
    if args.lr is not None:
        config.domain_config.lr = args.lr

    # Display configuration summary
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    if args.mode in ['train', 'all']:
        print(f"Training data: {config.training.data_path}")
        print(f"Domain classifier epochs: {config.domain_config.epochs}")
        print(f"Batch size: {config.domain_config.batch_size}")
        print(f"Learning rate: {config.domain_config.lr}")
    if args.mode in ['eval', 'all']:
        print(f"Test data: {config.evaluation.test_data_path}")
        print(f"Test samples: {config.evaluation.test_n or 'all'}")
        print(f"Output path: {config.evaluation.output_path}")
    print("=" * 80 + "\n")

    # Execute requested mode
    try:
        if args.mode == 'train':
            train.main(config)

        elif args.mode == 'eval':
            evaluate.main(config)

        elif args.mode == 'all':
            # Train first
            print("\n📚 PHASE 1: TRAINING")
            train.main(config)

            # Then evaluate
            print("\n\n📊 PHASE 2: EVALUATION")
            evaluate.main(config)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL OPERATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
