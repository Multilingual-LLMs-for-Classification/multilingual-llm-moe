"""
Evaluation script for the hierarchical routing system.

This script loads a trained routing system and evaluates it on test data,
producing detailed metrics and confusion matrices.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
from collections import Counter

from components import PromptRoutingSystem
from router_config import RouterSystemConfig
from evaluation_metrics import (
    pct, compute_prf_bal_kappa, print_confusion_matrix,
    get_expert_used, print_expert_selection_summary,
    print_expert_performance, print_language_group_comparison,
    print_expert_confusion_matrices
)

# Import PIIExpert for task-specific evaluation
from src.models.experts.llms.adapters.finance.pii.PIIExpert import PIIExpert

# Global PIIExpert instance for evaluation
_pii_expert = PIIExpert()


def load_test_data(filepath: str, test_n: int = None) -> List[Dict]:
    """
    Load test data from JSON file.

    Args:
        filepath: Path to test data JSON file
        test_n: Optional limit on number of samples (None = use all)

    Returns:
        List of test samples
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if test_n is not None:
        data = data[:test_n]

    return data


def compute_task_specific_metrics(system: PromptRoutingSystem,
                                   per_task_data: Dict[str, Dict]) -> Dict:
    """
    Compute expert-specific evaluation metrics for each task.

    Uses the task expert's compute_metrics() method to get task-appropriate
    metrics (e.g., MAE/RMSE for rating, entity-level F1 for PII, etc.)

    Args:
        system: PromptRoutingSystem with expert_pool
        per_task_data: Dictionary with predictions and ground truths per task

    Returns:
        Dictionary of task-specific metrics
    """
    task_specific_metrics = {}

    for task_name, data in per_task_data.items():
        predictions = data['predictions']
        ground_truths = data['ground_truths']

        if not predictions:
            # No samples for this task
            task_specific_metrics[task_name] = {
                'status': 'no_samples',
                'message': f'No correctly routed samples for task: {task_name}'
            }
            continue

        try:
            # Get the appropriate expert from the expert pool
            # We'll use 'finance' domain and 'english' language as default
            # since we just need the expert's compute_metrics method
            expert = None

            if task_name == 'rating':
                # Get sentiment analysis expert
                from src.models.experts.llms.adapters.finance.sentiment_analysis.SentimentAnalysisExpert import SentimentAnalysisExpert
                expert = SentimentAnalysisExpert()
            elif task_name == 'news':
                # Get news classification expert
                from src.models.experts.llms.adapters.finance.news_classification.NewsClassificationExpert import NewsClassificationExpert
                expert = NewsClassificationExpert()
            elif task_name == 'pii':
                # Get PII expert - use the global instance
                expert = _pii_expert
                # For PII, compute accuracy metrics using the raw strings
                # The expert methods handle parsing internally
                accuracy_metrics = expert.compute_accuracy_metrics(predictions, ground_truths)

                # For compute_metrics, we still need parsed entity lists
                parsed_predictions = []
                parsed_ground_truths = []
                for pred, gt in zip(predictions, ground_truths):
                    # Parse prediction (already cleaned by PIIExpert.clean_output)
                    try:
                        pred_entities = json.loads(pred) if isinstance(pred, str) else pred
                    except:
                        pred_entities = []
                    parsed_predictions.append(
                        pred_entities if isinstance(pred_entities, list) else []
                    )

                    # Parse ground truth
                    try:
                        gt_entities = json.loads(gt) if isinstance(gt, str) else gt
                    except:
                        gt_entities = []
                    parsed_ground_truths.append(
                        gt_entities if isinstance(gt_entities, list) else []
                    )

                # Compute entity-level metrics
                metrics = expert.compute_metrics(parsed_predictions, parsed_ground_truths)
                # Add accuracy metrics to the result
                metrics['accuracy_metrics'] = accuracy_metrics

                task_specific_metrics[task_name] = {
                    'status': 'success',
                    'metrics': metrics,
                    'num_samples': len(predictions)
                }
                continue  # Skip the generic expert.compute_metrics call below
            else:
                task_specific_metrics[task_name] = {
                    'status': 'unknown_task',
                    'message': f'Unknown task: {task_name}'
                }
                continue

            # Compute task-specific metrics using expert's method
            if expert:
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


def evaluate_routing_system(system: PromptRoutingSystem,
                            test_data: List[Dict],
                            config: RouterSystemConfig) -> Dict:
    """
    Evaluate the routing system on test data using expert-specific evaluation.

    Args:
        system: Initialized PromptRoutingSystem
        test_data: List of test samples
        config: System configuration

    Returns:
        Dictionary containing evaluation results and metrics
    """
    print(f"\nEvaluating on {len(test_data)} test samples...")
    print("=" * 80)

    # Extract unique labels for confusion matrices
    domain_labels = sorted({item['domain'] for item in test_data
                          if isinstance(item, dict) and 'domain' in item})
    task_labels = sorted({item['task'] for item in test_data
                        if isinstance(item, dict) and 'task' in item})
    # For expert labels, exclude PII task labels (they are JSON arrays, not suitable for confusion matrix)
    # PII uses F1 score categories instead
    expert_labels = sorted({item['label'] for item in test_data
                          if isinstance(item, dict) and 'label' in item
                          and item.get('task') != 'pii'})
    # Add PII F1 score categories if PII task exists
    if 'pii' in task_labels:
        pii_categories = ["pii_gt"] + [f"F1_{i}%" for i in range(0, 110, 10)]
        expert_labels = expert_labels + pii_categories

    # Confusion matrices
    cm_domain = Counter()
    cm_task = Counter()
    cm_expert = Counter()

    # Per-language statistics
    per_lang_total = Counter()
    per_lang_dom = Counter()
    per_lang_task = Counter()
    per_lang_exact = Counter()
    per_lang_correct = Counter()

    # Per-expert statistics
    per_expert_total = Counter()
    per_expert_correct = Counter()
    per_expert_cm = {}
    per_lang_expert = {}

    # Per-task evaluation data (for expert-specific metrics)
    per_task_data = {
        'rating': {'predictions': [], 'ground_truths': []},
        'news': {'predictions': [], 'ground_truths': []},
        'pii': {'predictions': [], 'ground_truths': []}
    }

    # CSV data collection
    csv_data = []

    # Process each test sample
    for item in test_data:
        if not isinstance(item, dict):
            continue

        prompt = item['prompt']
        gt_domain = item['domain']
        gt_task = item['task']
        gt_label = item['label']

        # Build input_data dict from item, excluding routing metadata
        input_data = {k: v for k, v in item.items()
                     if k not in ['prompt', 'domain', 'task', 'label']}

        # Route through system
        result = system.route_prompt(prompt, input_data=input_data)
        pred_domain = result['domain']
        pred_task = result['task']
        lang_tag = result.get('language', '?')
        pred_label = result['result']
        raw_response = result.get('raw_response', '')

        # Extract text fields for CSV (task-agnostic fallback chain)
        # Tries multiple field names to handle different task formats:
        # - Rating: may have 'text'/'title' or 'classification_text'/'review_title'
        # - News: typically has 'classification_text' only
        # - PII: typically has 'generated_text' only
        main_text = input_data.get('text',
                                   input_data.get('classification_text',
                                   input_data.get('generated_text', '')))
        title_text = input_data.get('title',
                                    input_data.get('review_title', ''))

        # Collect CSV data
        csv_data.append({
            'title': title_text if title_text else '',  # Empty for tasks without title
            'text': main_text[:100] + '...' if len(main_text) > 100 else main_text,
            'language': lang_tag,
            'domain': pred_domain,
            'task': pred_task,
            'expected_label': gt_label,
            'predicted_label': pred_label,
            'raw_response': raw_response
        })

        # Determine which expert was used
        expert_key = get_expert_used(lang_tag, pred_domain, pred_task)

        # Update statistics
        # For PII task, use entity-level F1 instead of exact match
        if gt_task == 'pii':
            is_correct = _pii_expert.is_correct(pred_label, gt_label, threshold=0.5)
        else:
            is_correct = (pred_label == gt_label)
        dom_ok = (pred_domain == gt_domain)
        task_ok = (pred_task == gt_task)
        both_ok = dom_ok and task_ok

        # Confusion matrices
        cm_domain[(gt_domain, pred_domain)] += 1
        cm_task[(gt_task, pred_task)] += 1
        # For PII, use F1 score category instead of raw labels for confusion matrix
        if gt_task == 'pii':
            pii_category = _pii_expert.get_score_category(pred_label, gt_label)
            cm_expert[(f"pii_gt", pii_category)] += 1
        else:
            cm_expert[(gt_label, pred_label)] += 1

        # Per-language stats
        per_lang_total[lang_tag] += 1
        per_lang_correct[lang_tag] += int(is_correct)
        per_lang_dom[lang_tag] += int(dom_ok)
        per_lang_task[lang_tag] += int(task_ok)
        per_lang_exact[lang_tag] += int(both_ok)

        # Per-expert stats
        per_expert_total[expert_key] += 1
        if is_correct:
            per_expert_correct[expert_key] += 1

        # Per-expert confusion matrix
        # For PII, use F1 score category instead of raw labels
        if expert_key not in per_expert_cm:
            per_expert_cm[expert_key] = Counter()
        if gt_task == 'pii':
            pii_category = _pii_expert.get_score_category(pred_label, gt_label)
            per_expert_cm[expert_key][("pii_gt", pii_category)] += 1
        else:
            per_expert_cm[expert_key][(gt_label, pred_label)] += 1

        # Track which expert was used for each language
        per_lang_expert[lang_tag] = expert_key

        # Collect task-specific data for expert evaluation
        # Only collect if routing was correct (otherwise task output is meaningless)
        if both_ok and gt_task in per_task_data:
            per_task_data[gt_task]['predictions'].append(pred_label)
            per_task_data[gt_task]['ground_truths'].append(gt_label)

    # Calculate metrics
    dom_metrics = compute_prf_bal_kappa(cm_domain, domain_labels)
    task_metrics = compute_prf_bal_kappa(cm_task, task_labels)
    expert_metrics = compute_prf_bal_kappa(cm_expert, expert_labels)

    # Compute expert-specific task metrics
    task_specific_metrics = compute_task_specific_metrics(system, per_task_data)

    # Print results
    print_evaluation_results(
        dom_metrics, task_metrics, expert_metrics,
        cm_domain, cm_task, cm_expert,
        domain_labels, task_labels, expert_labels,
        per_lang_total, per_lang_dom, per_lang_task, per_lang_exact,
        per_expert_cm, per_lang_expert, per_lang_correct,
        task_specific_metrics
    )

    # Save CSV file
    save_csv_results(csv_data, config.evaluation.output_path)

    return {
        'domain_metrics': dom_metrics,
        'task_metrics': task_metrics,
        'expert_metrics': expert_metrics,
        'task_specific_metrics': task_specific_metrics,
        'csv_data': csv_data,
        'per_lang_stats': {
            'total': dict(per_lang_total),
            'correct': dict(per_lang_correct),
            'dom_correct': dict(per_lang_dom),
            'task_correct': dict(per_lang_task),
            'exact_correct': dict(per_lang_exact)
        },
        'per_expert_stats': {
            'total': dict(per_expert_total),
            'correct': dict(per_expert_correct),
            'confusion_matrices': {k: dict(v) for k, v in per_expert_cm.items()}
        }
    }


def print_task_specific_metrics(task_specific_metrics: Dict):
    """
    Print task-specific evaluation metrics computed by expert evaluation methods.

    Args:
        task_specific_metrics: Dictionary of task-specific metrics from experts
    """
    print("\n" + "=" * 80)
    print("📊 TASK-SPECIFIC EXPERT EVALUATION")
    print("=" * 80)

    for task_name, result in task_specific_metrics.items():
        print(f"\n🔍 Task: {task_name.upper()}")
        print("-" * 80)

        if result['status'] != 'success':
            print(f"  ⚠️  {result['message']}")
            continue

        metrics = result['metrics']
        num_samples = result['num_samples']

        print(f"  Samples evaluated: {num_samples}")
        print()

        # Task-specific metric display
        if task_name == 'rating':
            # Sentiment analysis (star rating) metrics
            print("  📈 Classification Metrics:")
            print(f"    Accuracy           : {pct(metrics.get('accuracy', 0.0))}")
            print(f"    Macro  P/R/F1      : {pct(metrics.get('macro_precision', 0.0))} / {pct(metrics.get('macro_recall', 0.0))} / {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Micro  P/R/F1      : {pct(metrics.get('micro_precision', 0.0))} / {pct(metrics.get('micro_recall', 0.0))} / {pct(metrics.get('micro_f1', 0.0))}")
            print()
            print("  📉 Ordinal Metrics (star ratings as continuous):")
            print(f"    MAE (Mean Abs Err) : {metrics.get('mae', 0.0):.4f} stars")
            print(f"    RMSE               : {metrics.get('rmse', 0.0):.4f} stars")
            print(f"    Coverage           : {pct(metrics.get('coverage', 0.0))}")

        elif task_name == 'news':
            # News classification metrics
            print("  📈 Classification Metrics:")
            print(f"    Accuracy           : {pct(metrics.get('accuracy', 0.0))}")
            print(f"    Macro F1           : {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Weighted F1        : {pct(metrics.get('weighted_f1', 0.0))}")
            print(f"    Coverage           : {pct(metrics.get('coverage', 0.0))}")

            # Per-class F1 scores
            per_class = metrics.get('per_class_f1', {})
            if per_class:
                print()
                print("  📊 Per-Class F1 Scores:")
                for label, f1 in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {label:20s} : {pct(f1)}")

        elif task_name == 'pii':
            # PII extraction metrics - Accuracy Metrics
            acc_metrics = metrics.get('accuracy_metrics', {})
            print("  📈 Accuracy Metrics:")
            print(f"    Token-Level Accuracy : {pct(acc_metrics.get('token_level_accuracy', 0.0))}")
            print(f"    Exact Match Accuracy : {pct(acc_metrics.get('exact_match_accuracy', 0.0))}")
            print(f"    Average F1 Score     : {pct(acc_metrics.get('average_f1_score', 0.0))}")
            print(f"    Perfect F1 (=100%)   : {acc_metrics.get('perfect_f1_count', 0)} ({pct(acc_metrics.get('perfect_f1_pct', 0.0))})")
            print(f"    Zero F1 (=0%)        : {acc_metrics.get('zero_f1_count', 0)} ({pct(acc_metrics.get('zero_f1_pct', 0.0))})")

            print()
            print("  📈 Entity-Level Metrics:")
            print(f"    Micro F1             : {pct(metrics.get('micro_f1', 0.0))}")
            print(f"    Macro F1             : {pct(metrics.get('macro_f1', 0.0))}")
            print(f"    Micro Precision      : {pct(metrics.get('micro_precision', 0.0))}")
            print(f"    Micro Recall         : {pct(metrics.get('micro_recall', 0.0))}")
            print(f"    Total gold entities  : {metrics.get('total_gold_entities', 0)}")
            print(f"    Total pred entities  : {metrics.get('total_pred_entities', 0)}")
            print(f"    Correct tokens       : {acc_metrics.get('total_correct_tokens', 0)} / {acc_metrics.get('total_tokens', 0)}")

            # Per-label metrics
            per_label = metrics.get('per_label_metrics', {})
            if per_label:
                print()
                print("  📊 Per-Label Metrics:")
                print(f"    {'Label':20s} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
                print("    " + "-" * 56)
                for label, label_metrics in sorted(
                    per_label.items(), key=lambda x: x[1].get('f1', 0), reverse=True
                ):
                    p = label_metrics.get('precision', 0.0)
                    r = label_metrics.get('recall', 0.0)
                    f1 = label_metrics.get('f1', 0.0)
                    print(f"    {label:20s} | {pct(p):>10} | {pct(r):>10} | {pct(f1):>10}")

        else:
            # Unknown task - print all available metrics
            print("  📊 Available Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key:20s} : {value}")

    print()


def print_evaluation_results(dom_metrics, task_metrics, expert_metrics,
                            cm_domain, cm_task, cm_expert,
                            domain_labels, task_labels, expert_labels,
                            per_lang_total, per_lang_dom, per_lang_task, per_lang_exact,
                            per_expert_cm, per_lang_expert, per_lang_correct,
                            task_specific_metrics):
    """Print comprehensive evaluation results including task-specific metrics."""

    print("\n" + "=" * 80)
    print("ROUTING SYSTEM EVALUATION RESULTS")
    print("=" * 80)

    # Domain classification metrics
    print("\n📊 Domain Classification:")
    print(f"  Accuracy           : {pct(dom_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {pct(dom_metrics['macro_p'])} / {pct(dom_metrics['macro_r'])} / {pct(dom_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {pct(dom_metrics['micro_p'])} / {pct(dom_metrics['micro_r'])} / {pct(dom_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {pct(dom_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {pct(dom_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {pct(dom_metrics['kappa'])}")

    # Task classification metrics
    print("\n📊 Task Classification:")
    print(f"  Accuracy           : {pct(task_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {pct(task_metrics['macro_p'])} / {pct(task_metrics['macro_r'])} / {pct(task_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {pct(task_metrics['micro_p'])} / {pct(task_metrics['micro_r'])} / {pct(task_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {pct(task_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {pct(task_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {pct(task_metrics['kappa'])}")

    # Expert (final output) metrics
    print("\n📊 Expert Output Classification:")
    print(f"  Accuracy           : {pct(expert_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {pct(expert_metrics['macro_p'])} / {pct(expert_metrics['macro_r'])} / {pct(expert_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {pct(expert_metrics['micro_p'])} / {pct(expert_metrics['micro_r'])} / {pct(expert_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {pct(expert_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {pct(expert_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {pct(expert_metrics['kappa'])}")

    # Task-specific metrics (using expert evaluation methods)
    print_task_specific_metrics(task_specific_metrics)

    # Confusion matrices
    print_confusion_matrix(cm_domain, domain_labels, "\n📋 Domain Confusion Matrix (GT rows × Pred cols)")
    print_confusion_matrix(cm_task, task_labels, "📋 Task Confusion Matrix (GT rows × Pred cols)")
    print_confusion_matrix(cm_expert, expert_labels, "📋 Expert Output Confusion Matrix (GT rows × Pred cols)")

    # Per-language breakdown
    if per_lang_total:
        print("\n📊 Per-Language Performance:")
        print("-" * 80)
        print(f"{'Language':>10} | {'N':>4} | {'Domain Acc':>12} | {'Task Acc':>10} | {'Exact Acc':>10}")
        print("-" * 80)
        for lang in sorted(per_lang_total):
            n_l = per_lang_total[lang]
            dom_acc_l = per_lang_dom[lang] / n_l if n_l else 0.0
            task_acc_l = per_lang_task[lang] / n_l if n_l else 0.0
            exact_l = per_lang_exact[lang] / n_l if n_l else 0.0
            print(f"{lang:>10} | {n_l:>4} | {pct(dom_acc_l):>12} | {pct(task_acc_l):>10} | {pct(exact_l):>10}")

    # Expert-based analysis
    print_expert_selection_summary(per_lang_total, per_lang_expert)
    print_expert_performance(per_expert_cm, per_lang_total, per_lang_expert, per_lang_correct, expert_labels)
    print_language_group_comparison(per_expert_cm, per_lang_expert, expert_labels)
    print_expert_confusion_matrices(per_expert_cm, expert_labels)


def save_csv_results(csv_data: List[Dict], output_path: str):
    """Save results to CSV file with proper handling of different response types.

    Handles:
    - Rating task: Simple string ratings ("1"-"5")
    - News task: Category strings ("Finance", "Technology", etc.)
    - PII task: JSON arrays with entities (properly quoted/escaped)
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'text', 'language', 'domain', 'task',
                     'expected_label', 'predicted_label', 'raw_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                               quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"   Total samples: {len(csv_data)}")


def main(config: RouterSystemConfig):
    """
    Main evaluation function.

    Args:
        config: System configuration
    """
    print("=" * 80)
    print("HIERARCHICAL ROUTING SYSTEM - EVALUATION MODE")
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

    # Load test data
    print(f"\n📂 Loading test data from: {config.evaluation.test_data_path}")
    test_data = load_test_data(
        config.evaluation.test_data_path,
        config.evaluation.test_n
    )
    print(f"✅ Loaded {len(test_data)} test samples")

    # Run evaluation
    results = evaluate_routing_system(system, test_data, config)

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

    return results


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

    # Run evaluation
    main(config)
