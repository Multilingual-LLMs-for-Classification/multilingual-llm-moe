"""
Evaluation metrics and reporting functions for the routing system.

This module contains all evaluation-related utilities including:
- Metrics computation (PRF, accuracy, kappa)
- Confusion matrix printing
- Expert selection analysis
- Performance comparison across language groups
"""

import json
from pathlib import Path
from typing import List, Dict, Counter
from collections import Counter


def pct(x: float) -> str:
    """Format a float as a percentage string."""
    return f"{x*100:6.2f}%"


def compute_prf_bal_kappa(cm: Counter, labels: List[str]) -> Dict:
    """
    Compute precision, recall, F1, balanced accuracy, and Cohen's kappa.

    Args:
        cm: Confusion matrix as Counter with (ground_truth, predicted) tuples
        labels: List of class labels

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - macro_p, macro_r, macro_f1: Macro-averaged metrics
            - micro_p, micro_r, micro_f1: Micro-averaged metrics
            - weighted_f1: Weighted F1 score
            - balanced_acc: Balanced accuracy (mean recall)
            - kappa: Cohen's kappa coefficient
            - support: Support per class
            - pred_tot: Predictions per class
            - total: Total samples
    """
    support = {c: 0 for c in labels}
    pred_tot = {c: 0 for c in labels}
    tp = {c: cm[(c, c)] for c in labels}

    for g in labels:
        support[g] = sum(cm[(g, p)] for p in labels)
    for p in labels:
        pred_tot[p] = sum(cm[(g, p)] for g in labels)

    total = sum(support.values()) if support else 0
    precisions, recalls, f1s, weights, recalls_only = [], [], [], [], []

    # Micro-averaged counts
    micro_tp = sum(tp.values())
    micro_fp = sum(pred_tot[c] - tp[c] for c in labels)
    micro_fn = sum(support[c] - tp[c] for c in labels)

    # Per-class metrics
    for c in labels:
        p = tp[c] / pred_tot[c] if pred_tot[c] > 0 else 0.0
        r = tp[c] / support[c] if support[c] > 0 else 0.0
        f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        recalls_only.append(r)
        weights.append(support[c] / total if total else 0.0)

    # Macro-averaged metrics
    macro_p = sum(precisions)/len(labels) if labels else 0.0
    macro_r = sum(recalls)/len(labels) if labels else 0.0
    macro_f1 = sum(f1s)/len(labels) if labels else 0.0

    # Weighted F1
    weighted_f1 = sum(w*f for w, f in zip(weights, f1s)) if labels else 0.0

    # Balanced accuracy (mean recall across classes)
    balanced_acc = sum(recalls_only)/len(labels) if labels else 0.0

    # Micro-averaged metrics
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p + micro_r) > 0 else 0.0

    # Overall accuracy
    accuracy = micro_tp / total if total else 0.0

    # Cohen's kappa
    pe = sum((support[c]/total) * (pred_tot[c]/total) for c in labels) if total else 0.0
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1,
        'weighted_f1': weighted_f1, 'balanced_acc': balanced_acc, 'kappa': kappa,
        'support': support, 'pred_tot': pred_tot, 'total': total,
    }


def print_confusion_matrix(cm: Counter, labels: List[str], title: str):
    """
    Pretty print a confusion matrix.

    Args:
        cm: Confusion matrix as Counter with (ground_truth, predicted) tuples
        labels: List of class labels
        title: Title to print above the matrix
    """
    print(title)
    if not labels:
        print("  (no labels)\n")
        return

    header = "      " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)

    for gt in labels:
        row = [f"{gt:>6}"]
        for pr in labels:
            row.append(f"{cm[(gt, pr)]:>22}")
        print(" ".join(row))
    print()


def get_expert_used(language: str, domain: str, task: str,
                    registry_path: str = "experts/config/experts_registry.json") -> str:
    """
    Determine which base model/expert was used based on task+language combination.

    Queries experts_registry.json to find the correct base_model_key.
    This mirrors the logic in LLMAdapterPool._resolve_base_model_for_language()
    to determine which model was actually selected during routing.

    Args:
        language: Detected language (e.g., "english", "japanese")
        domain: Detected domain (e.g., "finance")
        task: Detected task (e.g., "rating", "news")
        registry_path: Path to experts_registry.json

    Returns:
        base_model_key (e.g., "llama-2-7b-hf", "aya-23", "google/gemma-7b")
    """
    # Construct task_key (e.g., "finance/rating")
    task_key = f"{domain}/{task}"

    # Load registry
    try:
        registry_file = Path(registry_path)
        if not registry_file.is_absolute():
            # Make it relative to this file's location
            registry_file = Path(__file__).parents[4] / registry_path

        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load registry from {registry_path}: {e}")
        return "unknown"

    # Get task config
    tcfg = registry.get("tasks", {}).get(task_key)
    if not tcfg:
        return "unknown"

    default_base = tcfg.get("base_model_key", "default")

    # Check if task has language_mapping
    lang_mapping = tcfg.get("language_mapping")
    if not lang_mapping:
        # No language mapping - uses default model for all languages
        return default_base

    # Normalize language
    lang_normalized = language.lower() if language else ""

    # Priority 1: Check for direct per-language mapping
    if lang_normalized in lang_mapping:
        lang_cfg = lang_mapping[lang_normalized]
        # Check if it's a per-language entry (no "languages" key)
        if "languages" not in lang_cfg:
            return lang_cfg.get("base_model_key", default_base)

    # Priority 2: Find which language group this language belongs to
    for group_name, group_cfg in lang_mapping.items():
        languages = group_cfg.get("languages", [])
        if lang_normalized in languages:
            # Found the language group - return its base_model_key
            return group_cfg.get("base_model_key", default_base)

    # Priority 3: Language not found in any group - use default
    return default_base


def print_expert_selection_summary(per_lang_total: Counter, per_lang_expert: Dict[str, str]):
    """
    Print which expert/model was selected for each language.

    Args:
        per_lang_total: Counter of samples per language
        per_lang_expert: Dict mapping language to expert name
    """
    print("\nEXPERT/MODEL SELECTION BY LANGUAGE")
    print("=" * 80)

    # Group by expert
    expert_langs = {}
    for lang, expert in per_lang_expert.items():
        if expert not in expert_langs:
            expert_langs[expert] = []
        expert_langs[expert].append(lang)

    # Define language groups (hardcoded for now, could be made configurable)
    groups = {
        "llama-2-7b-hf": ("European", ["english", "german", "spanish", "french"]),
        "aya-23": ("Asian", ["japanese", "chinese"])
    }

    total_samples = sum(per_lang_total.values())

    for expert, (group_name, expected_langs) in groups.items():
        if expert in expert_langs:
            print(f"\nLanguage Group: {group_name} ({expert})")
            print("-" * 80)

            group_total = 0
            for lang in sorted(expected_langs):
                if lang in per_lang_total:
                    count = per_lang_total[lang]
                    print(f"  {lang:<12} : {count:>4} samples")
                    group_total += count

            pct_val = (group_total / total_samples * 100) if total_samples > 0 else 0
            print(f"  {'Total':<12} : {group_total:>4} samples ({pct_val:.1f}%)")

    print("\n" + "=" * 80 + "\n")


def print_expert_performance(per_expert_cm: Dict[str, Counter],
                             per_lang_total: Counter,
                             per_lang_expert: Dict[str, str],
                             per_lang_correct: Counter,
                             expert_labels: List[str],
                             per_expert_correct: Counter = None,
                             per_expert_total: Counter = None):
    """
    Print performance metrics broken down by expert/model.

    Args:
        per_expert_cm: Confusion matrices per expert
        per_lang_total: Total samples per language
        per_lang_expert: Mapping of language to expert
        per_lang_correct: Correct predictions per language
        expert_labels: List of class labels
        per_expert_correct: Direct correct count per expert (avoids CM label issues)
        per_expert_total: Direct total count per expert
    """
    print("\nPERFORMANCE BY EXPERT/MODEL")
    print("=" * 80)

    # Group languages by expert
    expert_to_langs = {}
    for lang, expert in per_lang_expert.items():
        if expert not in expert_to_langs:
            expert_to_langs[expert] = []
        expert_to_langs[expert].append(lang)

    for expert in sorted(per_expert_cm.keys()):
        langs = sorted(expert_to_langs.get(expert, []))

        if not langs:
            continue

        print(f"\n{expert} (Languages: {', '.join(langs)})")
        print("-" * 80)

        cm = per_expert_cm[expert]

        # Compute accuracy from direct counts (avoids PII confusion matrix
        # label mismatch where ("pii_gt", "F1_X%") entries have no diagonal)
        if per_expert_correct is not None and per_expert_total is not None:
            total = per_expert_total.get(expert, 0)
            correct = per_expert_correct.get(expert, 0)
            accuracy = correct / total if total > 0 else 0.0
        else:
            total = sum(per_lang_total[lang] for lang in langs)
            accuracy = 0.0

        # For F1 metrics, use only non-PII entries with expert-specific labels.
        # PII entries use ("pii_gt", "F1_X%") which never form valid diagonal
        # entries for TP computation. PII metrics are reported separately in
        # the task-specific evaluation section.
        non_pii_cm = Counter({k: v for k, v in cm.items() if k[0] != 'pii_gt'})
        non_pii_labels = sorted({label for pair in non_pii_cm.keys() for label in pair})

        if non_pii_labels:
            metrics = compute_prf_bal_kappa(non_pii_cm, non_pii_labels)
            macro_f1 = metrics['macro_f1']
            weighted_f1 = metrics['weighted_f1']
        else:
            macro_f1 = 0.0
            weighted_f1 = 0.0

        print(f"  Samples      : {total}")
        print(f"  Accuracy     : {pct(accuracy)}")
        print(f"  Macro F1     : {pct(macro_f1)}")
        print(f"  Weighted F1  : {pct(weighted_f1)}")

        # Per-language breakdown for this expert
        print(f"\n  Per-language performance:")
        for lang in langs:
            if lang in per_lang_total:
                n = per_lang_total[lang]
                correct = per_lang_correct.get(lang, 0)
                acc = correct / n if n > 0 else 0.0
                print(f"    {lang:<12} : Accuracy = {pct(acc)}, Samples = {n}")

    print("\n" + "=" * 80 + "\n")


def print_language_group_comparison(per_expert_cm: Dict[str, Counter],
                                   per_lang_expert: Dict[str, str],
                                   expert_labels: List[str],
                                   per_expert_correct: Counter = None,
                                   per_expert_total: Counter = None):
    """
    Compare European vs Asian language group performance.

    Args:
        per_expert_cm: Confusion matrices per expert
        per_lang_expert: Mapping of language to expert
        expert_labels: List of class labels
        per_expert_correct: Direct correct count per expert
        per_expert_total: Direct total count per expert
    """
    print("\nLANGUAGE GROUP COMPARISON")
    print("=" * 80)

    # Calculate metrics for each group
    groups = {}
    for expert in ["llama-2-7b-hf", "aya-23"]:
        if expert in per_expert_cm:
            cm = per_expert_cm[expert]
            # Use non-PII entries with expert-specific labels for F1
            non_pii_cm = Counter({k: v for k, v in cm.items()
                                  if k[0] != 'pii_gt'})
            non_pii_labels = sorted(
                {label for pair in non_pii_cm.keys() for label in pair}
            )
            if non_pii_labels:
                metrics = compute_prf_bal_kappa(non_pii_cm, non_pii_labels)
            else:
                metrics = {
                    'accuracy': 0.0, 'macro_f1': 0.0,
                    'weighted_f1': 0.0, 'balanced_acc': 0.0,
                }
            # Override accuracy with direct counts
            if (per_expert_correct is not None
                    and per_expert_total is not None):
                total = per_expert_total.get(expert, 0)
                correct = per_expert_correct.get(expert, 0)
                metrics['accuracy'] = (correct / total
                                       if total > 0 else 0.0)
            groups[expert] = metrics

    if len(groups) == 2:
        print(f"\n{'Metric':<25} {'European (llama-2)':<20} {'Asian (aya-23)':<20} {'Difference':<15}")
        print("-" * 80)

        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('Macro F1', 'macro_f1'),
            ('Weighted F1', 'weighted_f1'),
            ('Balanced Accuracy', 'balanced_acc')
        ]

        for label, key in metrics_to_compare:
            euro_val = groups.get("llama-2-7b-hf", {}).get(key, 0.0)
            asian_val = groups.get("aya-23", {}).get(key, 0.0)
            diff = euro_val - asian_val

            print(f"{label:<25} {pct(euro_val):<20} {pct(asian_val):<20} {diff*100:+.1f}pp")

    print("\n" + "=" * 80 + "\n")


def print_expert_confusion_matrices(per_expert_cm: Dict[str, Counter],
                                    expert_labels: List[str]):
    """
    Print separate confusion matrices for each expert.

    Args:
        per_expert_cm: Confusion matrices per expert
        expert_labels: List of class labels
    """
    print("\nCONFUSION MATRICES BY EXPERT/MODEL")
    print("=" * 80)

    for expert in sorted(per_expert_cm.keys()):
        cm = per_expert_cm[expert]
        # Derive labels from this expert's own CM entries
        expert_specific_labels = sorted(
            {label for pair in cm.keys() for label in pair}
        )
        print_confusion_matrix(
            cm, expert_specific_labels,
            title=f"\n{expert} Confusion Matrix (GT rows x Pred cols)"
        )

    print("\n" + "=" * 80 + "\n")
