"""
Main CLI entry point for base model evaluation.

This script provides an interface for evaluating a base model WITHOUT adapters
against the same test data used by the routing system, for comparison purposes.
"""

import argparse
import sys
from pathlib import Path

from base_config import BaseEvaluationConfig
import evaluate


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Base Model Evaluation (No Adapters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default config (llama-2-7b-hf)
  python main.py

  # Run in background
  nohup python main.py > base_eval_output.log 2>&1 &

  # Evaluate with custom base model
  python main.py --base-model aya-23

  # Evaluate with limited test samples
  python main.py --test-n 100

  # Custom configuration file
  python main.py --config my_config.json

  # Custom test data and output
  python main.py --test-data ../test2.json --output results.csv
        """
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration JSON file (default: base_eval_config.json)'
    )

    # Base model selection
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        choices=[
            'llama-2-7b-hf',
            'llama-3-8B-Instruct',
            'mistral-7B-Instruct-v0.3',
            'google/gemma-7b',
            'aya-23',
            'deepseek-llm-7b-chat',
            'bloomz-7b1'
        ],
        help='Base model to evaluate (overrides config file)'
    )

    # Data paths
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

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            config = BaseEvaluationConfig.from_json(args.config)
            print(f"Loaded configuration from: {args.config}")
        except FileNotFoundError:
            print(f"Config file not found: {args.config}")
            sys.exit(1)
    else:
        # Try default location
        default_config = Path("base_eval_config.json")
        if default_config.exists():
            config = BaseEvaluationConfig.from_json(default_config)
            print(f"Loaded configuration from: {default_config}")
        else:
            print("No config file found. Using default configuration.")
            config = BaseEvaluationConfig()

    # Apply command-line overrides
    if args.base_model:
        config.base_model.base_model_key = args.base_model
    if args.test_data:
        config.evaluation.test_data_path = str(args.test_data)
    if args.test_n is not None:
        config.evaluation.test_n = args.test_n
    if args.output:
        config.evaluation.output_path = str(args.output)

    # Display configuration summary
    print("\n" + "=" * 80)
    print("BASE MODEL EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Base Model      : {config.base_model.base_model_key}")
    print(f"Test data       : {config.evaluation.test_data_path}")
    print(f"Test samples    : {config.evaluation.test_n or 'all'}")
    print(f"Output path     : {config.evaluation.output_path}")
    print(f"Adapters        : NONE (pure base model)")
    print("=" * 80 + "\n")

    # Execute evaluation
    try:
        evaluate.main(config)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("ALL OPERATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
