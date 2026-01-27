"""
Evaluation script for base model (without adapters).

This script evaluates a single base model without any fine-tuned adapters,
using the same test data and metrics as the routing system for comparison.
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter

# Add project root to path
project_root = Path(__file__).parents[7]
sys.path.insert(0, str(project_root))

from base_model_pool import BaseModelPool
from base_config import BaseEvaluationConfig

# Import expert cleaners for output processing
from src.models.experts.llms.adapters.finance.sentiment_analysis.SentimentAnalysisExpert import SentimentAnalysisExpert
from src.models.experts.llms.adapters.finance.news_classification.NewsClassificationExpert import NewsClassificationExpert
from src.models.experts.llms.adapters.finance.pii.PIIExpert import PIIExpert

# Import evaluation metrics from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation_metrics import pct, compute_prf_bal_kappa, print_confusion_matrix


# Global expert instances
_sentiment_expert = SentimentAnalysisExpert()
_news_expert = NewsClassificationExpert()
_pii_expert = PIIExpert()


def preprocess_pii_raw_output(raw_output: str) -> str:
    """
    Preprocess raw PII output from base model to add missing 'occurrence' fields.

    Base models often output valid JSON but without the 'occurrence' field that
    PIIExpert requires. This function extracts JSON, adds default occurrence values,
    and returns a properly formatted JSON string.

    Args:
        raw_output: Raw string output from the base model

    Returns:
        JSON string with occurrence fields added to each entity
    """
    import re

    if not raw_output or not isinstance(raw_output, str):
        return "[]"

    raw_output = raw_output.strip()

    # Try to extract JSON array from the output
    # Strategy 1: Find JSON array boundaries
    json_str = None
    if "[" in raw_output:
        start = raw_output.find("[")
        # Find matching closing bracket
        bracket_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(raw_output[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = raw_output[start:i+1]
                    break

        # Fallback: take from [ to last ]
        if json_str is None and "]" in raw_output[start:]:
            last_bracket = raw_output.rindex("]")
            if last_bracket > start:
                json_str = raw_output[start:last_bracket+1]

    if not json_str:
        return "[]"

    # Try to parse and add occurrence fields
    try:
        entities = json.loads(json_str)
        if not isinstance(entities, list):
            return "[]"

        # Track occurrences per (text, label) pair
        occurrence_counter = {}
        processed_entities = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            # Must have at least text and label
            if "text" not in entity or "label" not in entity:
                continue

            text = str(entity.get("text", "")).strip()
            label = str(entity.get("label", "")).strip()

            # Skip empty text
            if not text:
                continue

            # Calculate occurrence if not present
            if "occurrence" not in entity:
                key = (text.lower(), label)
                occurrence_counter[key] = occurrence_counter.get(key, 0) + 1
                entity["occurrence"] = occurrence_counter[key]
            else:
                # Ensure occurrence is an integer
                try:
                    entity["occurrence"] = int(entity["occurrence"])
                except (ValueError, TypeError):
                    entity["occurrence"] = 1

            processed_entities.append({
                "text": text,
                "label": label,
                "occurrence": entity["occurrence"]
            })

        return json.dumps(processed_entities, ensure_ascii=False)

    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            # Fix missing commas between objects
            fixed = re.sub(r'}\s*{', '},{', json_str)
            # Fix trailing commas
            fixed = re.sub(r',\s*]', ']', fixed)
            fixed = re.sub(r',\s*}', '}', fixed)

            entities = json.loads(fixed)
            if isinstance(entities, list):
                processed = []
                occurrence_counter = {}
                for entity in entities:
                    if isinstance(entity, dict) and "text" in entity and "label" in entity:
                        text = str(entity.get("text", "")).strip()
                        label = str(entity.get("label", "")).strip()
                        if text:
                            if "occurrence" not in entity:
                                key = (text.lower(), label)
                                occurrence_counter[key] = occurrence_counter.get(key, 0) + 1
                                entity["occurrence"] = occurrence_counter[key]
                            processed.append({
                                "text": text,
                                "label": label,
                                "occurrence": int(entity.get("occurrence", 1))
                            })
                return json.dumps(processed, ensure_ascii=False)
        except:
            pass

        return "[]"


def get_expert_for_task(task: str):
    """Get the appropriate expert cleaner for a task."""
    if task == "rating":
        return _sentiment_expert
    elif task == "news":
        return _news_expert
    elif task == "pii":
        return _pii_expert
    return None


def get_task_key(task: str) -> str:
    """Convert task name to task key format."""
    task_map = {
        "rating": "finance/rating",
        "news": "finance/news",
        "pii": "finance/pii"
    }
    return task_map.get(task, f"finance/{task}")


def load_test_data(filepath: str, test_n: int = None) -> List[Dict]:
    """Load test data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if test_n is not None:
        data = data[:test_n]

    return data


def compute_task_specific_metrics(per_task_data: Dict[str, Dict]) -> Dict:
    """
    Compute expert-specific evaluation metrics for each task.

    Uses the task expert's compute_metrics() method to get task-appropriate
    metrics (e.g., MAE/RMSE for rating, entity-level F1 for PII, etc.)
    """
    task_specific_metrics = {}

    for task_name, data in per_task_data.items():
        predictions = data['predictions']
        ground_truths = data['ground_truths']

        if not predictions:
            task_specific_metrics[task_name] = {
                'status': 'no_samples',
                'message': f'No samples for task: {task_name}'
            }
            continue

        try:
            expert = get_expert_for_task(task_name)
            if not expert:
                task_specific_metrics[task_name] = {
                    'status': 'unknown_task',
                    'message': f'Unknown task: {task_name}'
                }
                continue

            if task_name == 'pii':
                # For PII, compute accuracy metrics using raw strings
                accuracy_metrics = expert.compute_accuracy_metrics(predictions, ground_truths)

                # Parse predictions and ground truths for entity-level metrics
                parsed_predictions = []
                parsed_ground_truths = []
                for pred, gt in zip(predictions, ground_truths):
                    try:
                        pred_entities = json.loads(pred) if isinstance(pred, str) else pred
                    except:
                        pred_entities = []
                    parsed_predictions.append(
                        pred_entities if isinstance(pred_entities, list) else []
                    )

                    try:
                        gt_entities = json.loads(gt) if isinstance(gt, str) else gt
                    except:
                        gt_entities = []
                    parsed_ground_truths.append(
                        gt_entities if isinstance(gt_entities, list) else []
                    )

                # Compute entity-level metrics
                metrics = expert.compute_metrics(parsed_predictions, parsed_ground_truths)
                metrics['accuracy_metrics'] = accuracy_metrics

                task_specific_metrics[task_name] = {
                    'status': 'success',
                    'metrics': metrics,
                    'num_samples': len(predictions)
                }
            else:
                # Rating and News tasks
                metrics = expert.compute_metrics(predictions, ground_truths)
                task_specific_metrics[task_name] = {
                    'status': 'success',
                    'metrics': metrics,
                    'num_samples': len(predictions)
                }

        except Exception as e:
            task_specific_metrics[task_name] = {
                'status': 'error',
                'message': f'Error computing metrics: {str(e)}',
                'num_samples': len(predictions)
            }

    return task_specific_metrics


def print_task_specific_metrics(task_specific_metrics: Dict):
    """Print task-specific evaluation metrics."""
    print("\n" + "=" * 80)
    print("TASK-SPECIFIC EVALUATION (BASE MODEL)")
    print("=" * 80)

    for task_name, result in task_specific_metrics.items():
        print(f"\nTask: {task_name.upper()}")
        print("-" * 80)

        if result['status'] != 'success':
            print(f"  {result['message']}")
            continue

        metrics = result['metrics']
        num_samples = result['num_samples']

        print(f"  Samples evaluated: {num_samples}")
        print()

        if task_name == 'rating':
            print("  Classification Metrics:")
            print(f"    Accuracy           : {pct(metrics.get('accuracy', 0.0))}")
            print(f"    Macro  P/R/F1      : {pct(metrics.get('macro_precision', 0.0))} / {pct(metrics.get('macro_recall', 0.0))} / {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Micro  P/R/F1      : {pct(metrics.get('micro_precision', 0.0))} / {pct(metrics.get('micro_recall', 0.0))} / {pct(metrics.get('micro_f1', 0.0))}")
            print()
            print("  Ordinal Metrics:")
            print(f"    MAE                : {metrics.get('mae', 0.0):.4f} stars")
            print(f"    RMSE               : {metrics.get('rmse', 0.0):.4f} stars")
            print(f"    Coverage           : {pct(metrics.get('coverage', 0.0))}")

        elif task_name == 'news':
            print("  Classification Metrics:")
            print(f"    Accuracy           : {pct(metrics.get('accuracy', 0.0))}")
            print(f"    Macro F1           : {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Weighted F1        : {pct(metrics.get('weighted_f1', 0.0))}")
            print(f"    Coverage           : {pct(metrics.get('coverage', 0.0))}")

            per_class = metrics.get('per_class_f1', {})
            if per_class:
                print()
                print("  Per-Class F1 Scores:")
                for label, f1 in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {label:20s} : {pct(f1)}")

        elif task_name == 'pii':
            acc_metrics = metrics.get('accuracy_metrics', {})
            print("  Accuracy Metrics:")
            print(f"    Token-Level Accuracy : {pct(acc_metrics.get('token_level_accuracy', 0.0))}")
            print(f"    Exact Match Accuracy : {pct(acc_metrics.get('exact_match_accuracy', 0.0))}")
            print(f"    Average F1 Score     : {pct(acc_metrics.get('average_f1_score', 0.0))}")
            print(f"    Perfect F1 (=100%)   : {acc_metrics.get('perfect_f1_count', 0)} ({pct(acc_metrics.get('perfect_f1_pct', 0.0))})")
            print(f"    Zero F1 (=0%)        : {acc_metrics.get('zero_f1_count', 0)} ({pct(acc_metrics.get('zero_f1_pct', 0.0))})")

            print()
            print("  Entity-Level Metrics:")
            print(f"    Micro F1             : {pct(metrics.get('micro_f1', 0.0))}")
            print(f"    Macro F1             : {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Micro Precision      : {pct(metrics.get('micro_precision', 0.0))}")
            print(f"    Micro Recall         : {pct(metrics.get('micro_recall', 0.0))}")
            print(f"    Total gold entities  : {metrics.get('total_gold_entities', 0)}")
            print(f"    Total pred entities  : {metrics.get('total_pred_entities', 0)}")

            per_label = metrics.get('per_label_metrics', {})
            if per_label:
                print()
                print("  Per-Label Metrics:")
                print(f"    {'Label':20s} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
                print("    " + "-" * 56)
                for label, label_metrics in sorted(
                    per_label.items(), key=lambda x: x[1].get('f1', 0), reverse=True
                ):
                    p = label_metrics.get('precision', 0.0)
                    r = label_metrics.get('recall', 0.0)
                    f1 = label_metrics.get('f1', 0.0)
                    print(f"    {label:20s} | {pct(p):>10} | {pct(r):>10} | {pct(f1):>10}")

    print()


def evaluate_base_model(pool: BaseModelPool,
                        base_model_key: str,
                        test_data: List[Dict],
                        config: BaseEvaluationConfig) -> Dict:
    """
    Evaluate the base model on test data.

    Args:
        pool: BaseModelPool instance
        base_model_key: Key of the base model to use
        test_data: List of test samples
        config: Evaluation configuration

    Returns:
        Dictionary containing evaluation results and metrics
    """
    print(f"\nEvaluating base model '{base_model_key}' on {len(test_data)} test samples...")
    print("=" * 80)
    print("NOTE: Using base model WITHOUT any fine-tuned adapters")
    print("=" * 80)

    # Get generation config for each task
    task_gen_configs = {
        "finance/rating": {"max_new_tokens": 4, "temperature": 0.0, "top_p": 1.0},
        "finance/news": {"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0},
        "finance/pii": {"max_new_tokens": 4096, "temperature": 0.0, "top_p": 1.0}
    }

    # Extract unique labels for confusion matrices
    task_labels = sorted({item['task'] for item in test_data
                         if isinstance(item, dict) and 'task' in item})
    expert_labels = sorted({item['label'] for item in test_data
                           if isinstance(item, dict) and 'label' in item
                           and item.get('task') != 'pii'})
    # Add PII F1 score categories if PII task exists
    if 'pii' in task_labels:
        pii_categories = ["pii_gt"] + [f"F1_{i}%" for i in range(0, 110, 10)]
        expert_labels = expert_labels + pii_categories

    # Confusion matrices
    cm_task = Counter()
    cm_expert = Counter()

    # Per-language statistics
    per_lang_total = Counter()
    per_lang_correct = Counter()

    # Per-task evaluation data
    per_task_data = {
        'rating': {'predictions': [], 'ground_truths': []},
        'news': {'predictions': [], 'ground_truths': []},
        'pii': {'predictions': [], 'ground_truths': []}
    }

    # CSV data collection
    csv_data = []

    # Process each test sample
    for i, item in enumerate(test_data):
        if not isinstance(item, dict):
            continue

        prompt = item['prompt']
        gt_task = item['task']
        gt_label = item['label']
        lang_tag = item.get('language', 'english')

        # Get input data
        input_data = {k: v for k, v in item.items()
                     if k not in ['prompt', 'domain', 'task', 'label']}

        # Get task key and expert
        task_key = get_task_key(gt_task)
        expert = get_expert_for_task(gt_task)

        if not expert:
            print(f"  [{i+1}/{len(test_data)}] Unknown task: {gt_task}")
            continue

        # Prepare input
        if hasattr(expert, 'prepare_input'):
            classification_text, review_title = expert.prepare_input(input_data)
        else:
            classification_text = input_data.get('text',
                                    input_data.get('classification_text',
                                    input_data.get('generated_text', '')))
            review_title = input_data.get('title',
                                         input_data.get('review_title', ''))

        # Generate with base model
        try:
            gen_kwargs = task_gen_configs.get(task_key, {})
            raw_output, confidence = pool.generate(
                base_model_key,
                task_key,
                classification_text,
                review_title,
                language=lang_tag,
                **gen_kwargs
            )

            # Clean output - special handling for PII task
            if gt_task == "pii":
                # Preprocess PII output to add missing occurrence fields
                preprocessed = preprocess_pii_raw_output(raw_output)
                pred_label = preprocessed  # Already cleaned by preprocess function
            else:
                pred_label = expert.clean_output(raw_output)

        except Exception as e:
            print(f"  [{i+1}/{len(test_data)}] Error: {e}")
            raw_output = ""
            pred_label = ""

        # Progress update every 10 samples
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] Task: {gt_task}, Lang: {lang_tag}")

        # Extract text fields for CSV
        main_text = input_data.get('text',
                                   input_data.get('classification_text',
                                   input_data.get('generated_text', '')))
        title_text = input_data.get('title',
                                    input_data.get('review_title', ''))

        # Collect CSV data
        csv_data.append({
            'title': title_text if title_text else '',
            'text': main_text[:100] + '...' if len(main_text) > 100 else main_text,
            'language': lang_tag,
            'task': gt_task,
            'expected_label': gt_label,
            'predicted_label': pred_label,
            'raw_response': raw_output
        })

        # Determine correctness
        if gt_task == 'pii':
            is_correct = _pii_expert.is_correct(pred_label, gt_label, threshold=0.5)
        else:
            is_correct = (pred_label == gt_label)

        # Update confusion matrices
        cm_task[(gt_task, gt_task)] += 1  # Always correct task (no routing)
        if gt_task == 'pii':
            pii_category = _pii_expert.get_score_category(pred_label, gt_label)
            cm_expert[("pii_gt", pii_category)] += 1
        else:
            cm_expert[(gt_label, pred_label)] += 1

        # Per-language stats
        per_lang_total[lang_tag] += 1
        per_lang_correct[lang_tag] += int(is_correct)

        # Collect task-specific data
        if gt_task in per_task_data:
            per_task_data[gt_task]['predictions'].append(pred_label)
            per_task_data[gt_task]['ground_truths'].append(gt_label)

    # Calculate metrics
    task_metrics = compute_prf_bal_kappa(cm_task, task_labels)
    expert_metrics = compute_prf_bal_kappa(cm_expert, expert_labels)

    # Compute task-specific metrics
    task_specific_metrics = compute_task_specific_metrics(per_task_data)

    # Print results
    print_evaluation_results(
        task_metrics, expert_metrics,
        cm_task, cm_expert,
        task_labels, expert_labels,
        per_lang_total, per_lang_correct,
        task_specific_metrics,
        base_model_key
    )

    # Save CSV file
    save_csv_results(csv_data, config.evaluation.output_path)

    return {
        'task_metrics': task_metrics,
        'expert_metrics': expert_metrics,
        'task_specific_metrics': task_specific_metrics,
        'csv_data': csv_data,
        'per_lang_stats': {
            'total': dict(per_lang_total),
            'correct': dict(per_lang_correct)
        }
    }


def print_evaluation_results(task_metrics, expert_metrics,
                            cm_task, cm_expert,
                            task_labels, expert_labels,
                            per_lang_total, per_lang_correct,
                            task_specific_metrics,
                            base_model_key):
    """Print comprehensive evaluation results."""

    print("\n" + "=" * 80)
    print(f"BASE MODEL EVALUATION RESULTS: {base_model_key}")
    print("(No adapters - pure base model performance)")
    print("=" * 80)

    # Task classification (always 100% since no routing)
    print("\nTask Classification (No Routing):")
    print(f"  Accuracy           : {pct(task_metrics['accuracy'])}")

    # Expert (final output) metrics
    print("\nExpert Output Classification:")
    print(f"  Accuracy           : {pct(expert_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {pct(expert_metrics['macro_p'])} / {pct(expert_metrics['macro_r'])} / {pct(expert_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {pct(expert_metrics['micro_p'])} / {pct(expert_metrics['micro_r'])} / {pct(expert_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {pct(expert_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {pct(expert_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa      : {pct(expert_metrics['kappa'])}")

    # Task-specific metrics
    print_task_specific_metrics(task_specific_metrics)

    # Confusion matrices
    print_confusion_matrix(cm_expert, expert_labels, "\nExpert Output Confusion Matrix (GT rows x Pred cols)")

    # Per-language breakdown
    if per_lang_total:
        print("\nPer-Language Performance:")
        print("-" * 80)
        print(f"{'Language':>12} | {'N':>5} | {'Correct':>8} | {'Accuracy':>10}")
        print("-" * 80)
        for lang in sorted(per_lang_total):
            n_l = per_lang_total[lang]
            correct_l = per_lang_correct[lang]
            acc_l = correct_l / n_l if n_l else 0.0
            print(f"{lang:>12} | {n_l:>5} | {correct_l:>8} | {pct(acc_l):>10}")

        # Overall
        total_n = sum(per_lang_total.values())
        total_correct = sum(per_lang_correct.values())
        total_acc = total_correct / total_n if total_n else 0.0
        print("-" * 80)
        print(f"{'TOTAL':>12} | {total_n:>5} | {total_correct:>8} | {pct(total_acc):>10}")


def save_csv_results(csv_data: List[Dict], output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'text', 'language', 'task',
                     'expected_label', 'predicted_label', 'raw_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                               quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\nResults saved to: {output_path}")
    print(f"   Total samples: {len(csv_data)}")


def main(config: BaseEvaluationConfig):
    """Main evaluation function."""
    print("=" * 80)
    print("BASE MODEL EVALUATION (WITHOUT ADAPTERS)")
    print("=" * 80)

    # Initialize base model pool
    print(f"\nInitializing base model pool...")
    pool = BaseModelPool(project_root / config.expert_registry_path)

    # Display configuration
    print(f"Base model: {config.base_model.base_model_key}")
    print(f"Test data: {config.evaluation.test_data_path}")
    print(f"Output: {config.evaluation.output_path}")

    # Load test data
    print(f"\nLoading test data...")
    test_data = load_test_data(
        config.evaluation.test_data_path,
        config.evaluation.test_n
    )
    print(f"Loaded {len(test_data)} test samples")

    # Run evaluation
    results = evaluate_base_model(
        pool,
        config.base_model.base_model_key,
        test_data,
        config
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "base_eval_config.json"

    try:
        config = BaseEvaluationConfig.from_json(Path(config_path))
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Using defaults.")
        config = BaseEvaluationConfig()

    # Run evaluation
    main(config)
