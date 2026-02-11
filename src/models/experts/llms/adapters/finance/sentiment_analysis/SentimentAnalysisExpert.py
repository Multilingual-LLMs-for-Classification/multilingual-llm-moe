import re
import numpy as np
from typing import List, Dict
from collections import Counter


class SentimentAnalysisExpert:
    """
    Task-specific expert for sentiment analysis (star rating prediction).

    Handles:
    - Input data preparation (extracting review text and title)
    - Output cleaning (extracting star rating 1-5)
    - Task-specific evaluation metrics
    """

    # Valid star ratings
    VALID_RATINGS = {"1", "2", "3", "4", "5"}
    LABEL_SET = ["1", "2", "3", "4", "5"]

    def prepare_input(self, input_data: dict) -> tuple:
        """
        Extract and prepare input fields for sentiment analysis task.

        Args:
            input_data: Dictionary with task data

        Returns:
            Tuple of (main_text, title_text)
        """
        # Sentiment analysis uses 'classification_text' and 'review_title'
        main_text = input_data.get("classification_text", input_data.get("text", ""))
        title_text = input_data.get("review_title", input_data.get("title", ""))
        return main_text, title_text

    def clean_output(self, raw: str) -> str:
        """
        Extract a single digit between 1 and 5 from the raw model output.

        The method searches the text for an isolated digit 1-5 using a
        regular expression. If found, it returns the digit as a string.
        If no valid rating is found, an empty string is returned.

        Parameters
        ----------
        raw : str
            The raw output text produced by the LLM.

        Returns
        -------
        str
            A single character ("1"…"5") if detected, otherwise "".
        """
        m = re.search(r"\b([1-5])\b", raw)
        return m.group(1) if m else ""

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is a valid star rating."""
        return prediction in self.VALID_RATINGS

    def get_ground_truth(self, input_data: dict) -> str:
        """
        Extract ground truth label from input data.

        Args:
            input_data: Dictionary with task data

        Returns:
            Ground truth star rating as string
        """
        # Try different field names for ground truth
        label = input_data.get("label",
                              input_data.get("stars",
                              input_data.get("rating", "")))
        return str(label) if label else ""

    def compute_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict:
        """
        Compute task-specific evaluation metrics for star rating classification.

        Metrics:
        - Accuracy
        - Macro/Micro Precision, Recall, F1
        - MAE (Mean Absolute Error) - treating ratings as ordinal
        - RMSE (Root Mean Squared Error)
        - Coverage (% of valid predictions)

        Args:
            predictions: List of predicted star ratings
            ground_truths: List of ground truth star ratings

        Returns:
            Dictionary of evaluation metrics
        """
        # Filter to only valid predictions
        valid_pairs = [
            (int(gt), int(pred))
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
                "mae": float('inf'),
                "rmse": float('inf'),
                "coverage": 0.0,
                "total_samples": len(predictions),
                "valid_predictions": 0
            }

        gts, preds = zip(*valid_pairs)

        # Confusion matrix
        cm = Counter()
        for gt, pred in valid_pairs:
            cm[(str(gt), str(pred))] += 1

        # Calculate metrics per class
        tp = {label: cm[(label, label)] for label in self.LABEL_SET}
        support = {label: sum(cm[(label, p)] for p in self.LABEL_SET)
                  for label in self.LABEL_SET}
        pred_total = {label: sum(cm[(g, label)] for g in self.LABEL_SET)
                     for label in self.LABEL_SET}

        # Per-class metrics
        precisions, recalls, f1s = [], [], []
        for label in self.LABEL_SET:
            p = tp[label] / pred_total[label] if pred_total[label] > 0 else 0.0
            r = tp[label] / support[label] if support[label] > 0 else 0.0
            f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        # Macro metrics
        macro_p = sum(precisions) / len(self.LABEL_SET)
        macro_r = sum(recalls) / len(self.LABEL_SET)
        macro_f1 = sum(f1s) / len(self.LABEL_SET)

        # Micro metrics
        total_tp = sum(tp.values())
        total_fp = sum(pred_total[label] - tp[label] for label in self.LABEL_SET)
        total_fn = sum(support[label] - tp[label] for label in self.LABEL_SET)

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

        # Accuracy
        accuracy = total_tp / len(valid_pairs)

        # Regression metrics (MAE, RMSE)
        mae = np.mean([abs(gt - pred) for gt, pred in valid_pairs])
        rmse = np.sqrt(np.mean([(gt - pred)**2 for gt, pred in valid_pairs]))

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
            "mae": mae,
            "rmse": rmse,
            "coverage": coverage,
            "total_samples": len(predictions),
            "valid_predictions": len(valid_pairs),
            "confusion_matrix": dict(cm)
        }
