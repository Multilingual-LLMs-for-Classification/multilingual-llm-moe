from typing import List, Dict
from collections import Counter


CATEGORIES = ["E", "S", "C", "I"]

CATEGORY_NAMES = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant"
}


class ESCIExpert:
    """
    Task-specific expert for ESCI (Amazon Shopping Queries) classification.

    Classifies the relationship between a query and product as:
    - E (Exact): Product is an exact match for the query
    - S (Substitute): Product is a substitute for the query
    - C (Complement): Product is a complement to the query
    - I (Irrelevant): Product is irrelevant to the query

    Handles:
    - Input data preparation (extracting query and product)
    - Output cleaning (label normalization)
    - Task-specific evaluation metrics
    """

    LABEL_SET = CATEGORIES

    def prepare_input(self, input_data: dict) -> tuple:
        """
        Extract and prepare input fields for ESCI classification task.

        ESCI data stores the combined query+product in classification_text.
        Reformats "Query: X Product: Y" into the structured format used
        during fine-tuning (separate labeled sections with newlines).

        Args:
            input_data: Dictionary with task data

        Returns:
            Tuple of (classification_text, empty_string)
        """
        main_text = input_data.get("classification_text", "")
        main_text = self._reformat_query_product(main_text)
        return main_text, ""

    @staticmethod
    def _reformat_query_product(text: str) -> str:
        """Reformat 'Query: X Product: Y' into structured newline format matching training."""
        t = str(text).strip()
        if "Product:" in t:
            parts = t.split("Product:", 1)
            query = parts[0].replace("Query:", "").strip()[:300]
            product = parts[1].strip()[:500]
            return f"Search Query:\n{query}\n\nProduct Description:\n{product}"
        return t

    def clean_output(self, raw: str) -> str:
        """
        Clean and normalize LLM output for ESCI classification.

        Steps:
          1. Strips whitespace and converts to uppercase
          2. Checks for exact single-letter matches (E, S, C, I)
          3. Checks for full word matches (Exact, Substitute, etc.)
          4. Returns "unknown" if no match found
        """
        raw = raw.strip().upper()

        # Direct single letter match
        if raw in CATEGORIES:
            return raw

        # Check first character
        if raw and raw[0] in CATEGORIES:
            return raw[0]

        # Full word match
        raw_lower = raw.lower()
        for letter, name in CATEGORY_NAMES.items():
            if name.lower() in raw_lower or raw_lower in name.lower():
                return letter

        return "unknown"

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is a valid ESCI label."""
        return prediction in CATEGORIES

    def get_ground_truth(self, input_data: dict) -> str:
        """
        Extract ground truth label from input data.

        Args:
            input_data: Dictionary with task data

        Returns:
            Ground truth label as string (E, S, C, or I)
        """
        label = input_data.get("label",
                              input_data.get("esci_label",
                              input_data.get("class", "")))
        return str(label).strip().upper() if label else ""

    def compute_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        Compute task-specific evaluation metrics for ESCI classification.

        Metrics:
        - Accuracy
        - Macro/Micro Precision, Recall, F1
        - Per-class F1 scores
        - Coverage (% of valid predictions)

        Args:
            predictions: List of predicted labels (E, S, C, I)
            ground_truths: List of ground truth labels

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
            per_class_f1[f"{label} ({CATEGORY_NAMES[label]})"] = f

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
        coverage = len(valid_pairs) / len(predictions) if predictions else 0.0

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
