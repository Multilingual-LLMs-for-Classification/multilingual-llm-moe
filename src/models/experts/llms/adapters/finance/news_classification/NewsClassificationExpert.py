from typing import List, Dict
from collections import Counter


CATEGORIES = [
    "Finance",
    "Tax & Accounting",
    "Government & Controls",
    "Technology",
    "Industry",
    "Business & Management",
]


class NewsClassificationExpert:
    """
    Task-specific expert for news category classification.

    Handles:
    - Input data preparation (extracting news text and title)
    - Output cleaning (category name normalization)
    - Task-specific evaluation metrics
    """

    # Valid categories
    LABEL_SET = CATEGORIES

    def prepare_input(self, input_data: dict) -> tuple:
        """
        Extract and prepare input fields for news classification task.

        Args:
            input_data: Dictionary with task data

        Returns:
            Tuple of (main_text, title_text)
        """
        # News classification uses 'text' and 'title' fields
        main_text = input_data.get("text",
                                   input_data.get("classification_text", ""))
        title_text = input_data.get("title",
                                    input_data.get("review_title", ""))
        return main_text, title_text

    def clean_output(self, raw: str) -> str:
        """
        Clean and normalize LLM output for news-category classification.

        Steps:
          1. Converts text to lowercase
          2. Checks for exact matches with known categories
          3. Checks for partial substring matches
          4. Returns "unknown" if no match found
        """
        raw = raw.lower().strip()

        # Exact match
        for category in CATEGORIES:
            if category.lower() == raw:
                return category

        # Partial match
        for category in CATEGORIES:
            cat_l = category.lower()
            if raw in cat_l or cat_l in raw:
                return category

        return "unknown"

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is a valid news category."""
        return prediction in CATEGORIES

    def get_ground_truth(self, input_data: dict) -> str:
        """
        Extract ground truth label from input data.

        Args:
            input_data: Dictionary with task data

        Returns:
            Ground truth category as string
        """
        # Try different field names for ground truth
        label = input_data.get("label",
                              input_data.get("category",
                              input_data.get("class", "")))
        return str(label) if label else ""

    def compute_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        Compute task-specific evaluation metrics for news classification.

        Metrics:
        - Accuracy
        - Macro/Micro Precision, Recall, F1
        - Per-class F1 scores
        - Coverage (% of valid predictions)

        Args:
            predictions: List of predicted categories
            ground_truths: List of ground truth categories

        Returns:
            Dictionary of evaluation metrics
        """
        # Filter to only valid predictions
        valid_pairs = [
            (gt, pred)
            for gt, pred in zip(ground_truths, predictions)
            if self.is_valid_prediction(pred) and self.is_valid_prediction(gt)
        ]

        if not valid_pairs:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "micro_precision": 0.0,
                "micro_recall": 0.0,
                "micro_f1": 0.0,
                "weighted_f1": 0.0,
                "coverage": 0.0,
                "total_samples": len(predictions),
                "valid_predictions": 0,
                "per_class_f1": {}
            }

        gts, preds = zip(*valid_pairs)

        # Confusion matrix
        cm = Counter()
        for gt, pred in valid_pairs:
            cm[(gt, pred)] += 1

        # Calculate metrics per class
        tp = {label: cm[(label, label)] for label in self.LABEL_SET}
        support = {label: sum(cm[(label, p)] for p in self.LABEL_SET)
                  for label in self.LABEL_SET}
        pred_total = {label: sum(cm[(g, label)] for g in self.LABEL_SET)
                     for label in self.LABEL_SET}

        # Per-class metrics
        precisions, recalls, f1s, weights = [], [], [], []
        per_class_f1 = {}

        total_support = sum(support.values())

        for label in self.LABEL_SET:
            p = tp[label] / pred_total[label] if pred_total[label] > 0 else 0.0
            r = tp[label] / support[label] if support[label] > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            w = support[label] / total_support if total_support > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            weights.append(w)
            per_class_f1[label] = f

        # Macro metrics (unweighted average)
        macro_p = sum(precisions) / len(self.LABEL_SET) if self.LABEL_SET else 0.0
        macro_r = sum(recalls) / len(self.LABEL_SET) if self.LABEL_SET else 0.0
        macro_f1 = sum(f1s) / len(self.LABEL_SET) if self.LABEL_SET else 0.0

        # Weighted F1 (weighted by support)
        weighted_f1 = sum(w * f for w, f in zip(weights, f1s))

        # Micro metrics (aggregate counts)
        total_tp = sum(tp.values())
        total_fp = sum(pred_total[label] - tp[label] for label in self.LABEL_SET)
        total_fn = sum(support[label] - tp[label] for label in self.LABEL_SET)

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

        # Accuracy
        accuracy = total_tp / len(valid_pairs) if valid_pairs else 0.0

        # Coverage
        coverage = len(valid_pairs) / len(predictions)

        return {
            "accuracy": accuracy,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "coverage": coverage,
            "total_samples": len(predictions),
            "valid_predictions": len(valid_pairs),
            "per_class_f1": per_class_f1,
            "confusion_matrix": dict(cm)
        }
