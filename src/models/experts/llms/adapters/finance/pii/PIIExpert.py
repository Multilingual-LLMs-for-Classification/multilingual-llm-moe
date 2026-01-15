"""
PIIExpert - Output cleaner for PII (Personally Identifiable Information) extraction task.

This expert processes raw LLM outputs that should contain JSON arrays of PII entities.
It handles malformed JSON, repairs truncated outputs, and validates entity structure.
"""

import json
import re
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict


class PIIExpert:
    """
    Task-specific expert for PII (Personally Identifiable Information) extraction.

    Handles:
    - Input data preparation (extracting text fields)
    - Output cleaning (JSON extraction and validation)

    Expects raw output to be a JSON array of entities, where each entity has:
    - text: The PII text span
    - label: The PII category (person_name, date, location, etc.)
    - occurrence: The occurrence number (integer)
    """

    # Valid PII labels
    VALID_LABELS = {
        "person_name", "date", "location", "organization", "contact_info",
        "government_id", "financial_account", "payment_card",
        "user_identifier", "secret", "ip_address"
    }

    def __init__(self):
        """Initialize the PIIExpert."""
        pass

    def prepare_input(self, input_data: dict) -> tuple:
        """
        Extract and prepare input fields for PII task.

        Args:
            input_data: Dictionary with task data

        Returns:
            Tuple of (main_text, title_text)
        """
        # PII task uses 'generated_text' or 'text' field, no title
        main_text = input_data.get("generated_text",
                                   input_data.get("text",
                                   input_data.get("classification_text", "")))
        title_text = ""  # PII doesn't use title
        return main_text, title_text

    def clean_output(self, raw: str) -> str:
        """
        Extract and clean PII entities from raw LLM output.

        Args:
            raw: Raw string output from LLM (expected to contain JSON array)

        Returns:
            JSON string representing list of valid PII entities
        """
        # Extract entities using robust multi-strategy approach
        entities = self._robust_extract(raw)

        # Return as JSON string
        return json.dumps(entities, ensure_ascii=False)

    def _robust_extract(self, output: str) -> List[Dict]:
        """
        Enhanced extraction with multiple fallback strategies and validation.

        Tries multiple strategies in order:
        1. Direct JSON parse
        2. Extract JSON array and parse
        3. Repair truncated JSON
        4. Fix JSON syntax errors
        5. Extract objects individually (last resort)
        """
        if not isinstance(output, str) or not output.strip():
            return []

        # Strategy 1: Direct parse (best case)
        try:
            result = json.loads(output)
            if isinstance(result, list):
                return self._sanitize_entities(result)
        except:
            pass

        # Strategy 2: Extract JSON array and parse
        extracted = self._extract_json_array(output)
        try:
            result = json.loads(extracted)
            if isinstance(result, list):
                return self._sanitize_entities(result)
        except:
            pass

        # Strategy 3: Repair truncated JSON
        repaired = self._repair_truncated_json(extracted)
        try:
            result = json.loads(repaired)
            if isinstance(result, list):
                return self._sanitize_entities(result)
        except:
            pass

        # Strategy 4: Fix syntax errors
        fixed = self._fix_json_syntax(repaired)
        try:
            result = json.loads(fixed)
            if isinstance(result, list):
                return self._sanitize_entities(result)
        except:
            pass

        # Strategy 5: Extract objects individually (last resort)
        objects = self._extract_objects_individually(output)
        if objects:
            return self._sanitize_entities(objects)

        # If all strategies fail, return empty list
        return []

    def _extract_json_array(self, text: str) -> str:
        """Extract JSON array from text (handles markdown code blocks)."""
        if not isinstance(text, str):
            return "[]"

        # Strategy 1: Find complete JSON array
        array_pattern = r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]'
        matches = re.findall(array_pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (most likely to be complete)
            return max(matches, key=len)

        # Strategy 2: Find array boundaries even if nested
        if "[" in text:
            start = text.index("[")
            # Find matching closing bracket
            bracket_count = 0
            for i, char in enumerate(text[start:], start):
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[start:i+1]
            # If no matching bracket, take everything after [
            return text[start:]

        return "[]"

    def _repair_truncated_json(self, text: str) -> str:
        """Repair JSON that was cut off mid-generation."""
        if not text or not isinstance(text, str):
            return "[]"

        text = text.strip()

        # Ensure it starts with [
        if not text.startswith("["):
            text = "[" + text

        # Count brackets to detect truncation
        open_braces = text.count("{")
        close_braces = text.count("}")

        # If we have unmatched opening braces, remove the last incomplete object
        if open_braces > close_braces:
            # Find the last complete object
            last_complete_close = text.rfind("}")
            if last_complete_close > 0:
                text = text[:last_complete_close+1]

        # Ensure it ends with ]
        if not text.endswith("]"):
            text = text.rstrip(",").rstrip() + "]"

        return text

    def _fix_json_syntax(self, text: str) -> str:
        """Fix common JSON syntax errors."""
        if not text or not isinstance(text, str):
            return "[]"

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Fix missing commas between objects
        text = re.sub(r'}\s*{', '},{', text)

        # Fix missing commas between array elements
        text = re.sub(r'\]\s*\[', ',', text)

        # Fix trailing commas before closing brackets
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)

        # Fix single quotes (should be double quotes in JSON)
        text = text.replace("'text'", '"text"')
        text = text.replace("'label'", '"label"')
        text = text.replace("'occurrence'", '"occurrence"')

        # Remove any trailing text after the final ]
        if "]" in text:
            last_bracket = text.rindex("]")
            text = text[:last_bracket+1]

        return text

    def _extract_objects_individually(self, text: str) -> List[Dict]:
        """Last resort: extract individual JSON objects from malformed text."""
        objects = []

        # Find all potential JSON objects
        object_pattern = r'\{[^{}]*"text"[^{}]*"label"[^{}]*"occurrence"[^{}]*\}'
        matches = re.findall(object_pattern, text)

        for match in matches:
            try:
                # Try to parse each object
                obj = json.loads(match)
                if all(key in obj for key in ["text", "label", "occurrence"]):
                    objects.append(obj)
            except:
                # Try with quote fixing
                try:
                    fixed = match.replace("'", '"')
                    obj = json.loads(fixed)
                    if all(key in obj for key in ["text", "label", "occurrence"]):
                        objects.append(obj)
                except:
                    continue

        return objects

    def _validate_entity(self, entity: Dict) -> bool:
        """Validate that an entity has all required fields and valid values."""
        if not isinstance(entity, dict):
            return False

        required_keys = {"text", "label", "occurrence"}
        if not all(key in entity for key in required_keys):
            return False

        # Validate types
        if not isinstance(entity["text"], str) or not entity["text"].strip():
            return False
        if not isinstance(entity["label"], str) or not entity["label"].strip():
            return False
        if not isinstance(entity["occurrence"], (int, float)):
            return False

        # Ensure occurrence is positive integer
        if entity["occurrence"] < 1:
            return False

        # Optional: Validate label against known PII types (can be disabled for flexibility)
        # Uncomment the following lines to enforce strict label validation:
        # if entity["label"] not in self.VALID_LABELS:
        #     return False

        return True

    def _sanitize_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Filter and clean entity list.

        Removes invalid entities and normalizes valid ones.
        """
        if not isinstance(entities, list):
            return []

        valid_entities = []

        for entity in entities:
            if self._validate_entity(entity):
                # Normalize occurrence to integer
                entity["occurrence"] = int(entity["occurrence"])
                # Trim whitespace from text and label
                entity["text"] = entity["text"].strip()
                entity["label"] = entity["label"].strip()
                valid_entities.append(entity)

        return valid_entities

    def get_ground_truth(self, input_data: dict) -> List[Dict]:
        """
        Extract ground truth PII entities from input data.

        Args:
            input_data: Dictionary with task data

        Returns:
            List of ground truth PII entities
        """
        # Try different field names for ground truth
        label_field = input_data.get("label",
                                     input_data.get("pii_entities",
                                     input_data.get("entities",
                                     input_data.get("pii_optionA", []))))

        # If it's a JSON string, parse it
        if isinstance(label_field, str):
            try:
                entities = json.loads(label_field)
            except:
                entities = []
        else:
            entities = label_field if isinstance(label_field, list) else []

        # Sanitize to ensure valid structure
        return self._sanitize_entities(entities)

    def compute_metrics(self, predictions: List[List[Dict]],
                       ground_truths: List[List[Dict]]) -> Dict:
        """
        Compute task-specific evaluation metrics for PII extraction.

        Uses entity-level matching (text, label, occurrence).

        Metrics:
        - Micro F1 (overall across all entities)
        - Macro F1 (average per-sample F1)
        - Precision, Recall
        - Per-label F1 scores
        - Entity counts

        Args:
            predictions: List of predicted entity lists (one per sample)
            ground_truths: List of ground truth entity lists (one per sample)

        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) must have same length")

        # Sample-level metrics for macro averaging
        sample_f1s = []
        sample_precisions = []
        sample_recalls = []

        # Global counters for micro averaging
        global_tp = 0
        global_fp = 0
        global_fn = 0

        # Per-label tracking
        label_tp = defaultdict(int)
        label_fp = defaultdict(int)
        label_fn = defaultdict(int)
        label_support = defaultdict(int)

        for pred_entities, gold_entities in zip(predictions, ground_truths):
            # Sanitize inputs
            pred_entities = self._sanitize_entities(pred_entities) if isinstance(pred_entities, list) else []
            gold_entities = self._sanitize_entities(gold_entities) if isinstance(gold_entities, list) else []

            # Convert to sets for matching (case-insensitive text matching)
            gold_set = {
                (e["text"].lower().strip(), e["label"], e["occurrence"])
                for e in gold_entities
            }
            pred_set = {
                (e["text"].lower().strip(), e["label"], e["occurrence"])
                for e in pred_entities
            }

            # Calculate sample-level metrics
            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            global_tp += tp
            global_fp += fp
            global_fn += fn

            # Sample-level P/R/F1
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            sample_f1s.append(f1)
            sample_precisions.append(p)
            sample_recalls.append(r)

            # Per-label tracking
            for entity in gold_entities:
                label = entity["label"]
                label_support[label] += 1

            # Track TP/FP/FN per label
            matched_gold = gold_set & pred_set
            for text, label, occ in matched_gold:
                label_tp[label] += 1

            for text, label, occ in (pred_set - gold_set):
                label_fp[label] += 1

            for text, label, occ in (gold_set - pred_set):
                label_fn[label] += 1

        # Calculate micro metrics (aggregate all entities)
        micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

        # Calculate macro metrics (average per-sample)
        macro_p = np.mean(sample_precisions) if sample_precisions else 0.0
        macro_r = np.mean(sample_recalls) if sample_recalls else 0.0
        macro_f1 = np.mean(sample_f1s) if sample_f1s else 0.0

        # Per-label F1 scores
        per_label_f1 = {}
        for label in set(list(label_support.keys()) + list(label_tp.keys()) + list(label_fp.keys())):
            tp_l = label_tp[label]
            fp_l = label_fp[label]
            fn_l = label_fn[label]

            p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
            r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
            f1_l = (2 * p_l * r_l / (p_l + r_l)) if (p_l + r_l) > 0 else 0.0

            per_label_f1[label] = {
                "f1": f1_l,
                "precision": p_l,
                "recall": r_l,
                "support": label_support[label],
                "tp": tp_l,
                "fp": fp_l,
                "fn": fn_l
            }

        return {
            # Overall metrics
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,

            # Counts
            "total_samples": len(predictions),
            "total_gold_entities": global_tp + global_fn,
            "total_pred_entities": global_tp + global_fp,
            "total_tp": global_tp,
            "total_fp": global_fp,
            "total_fn": global_fn,

            # Sample-level statistics
            "sample_f1_mean": macro_f1,
            "sample_f1_std": np.std(sample_f1s) if sample_f1s else 0.0,
            "sample_f1_min": np.min(sample_f1s) if sample_f1s else 0.0,
            "sample_f1_max": np.max(sample_f1s) if sample_f1s else 1.0,

            # Per-label breakdown
            "per_label_metrics": per_label_f1,
            "label_distribution": dict(label_support)
        }

    @staticmethod
    def parse_label(label: str) -> List[Dict]:
        """
        Parse a label string (JSON) into a list of entity dictionaries.

        Args:
            label: JSON string or list of PII entities

        Returns:
            List of entity dictionaries
        """
        if isinstance(label, list):
            return label
        if not isinstance(label, str) or not label.strip():
            return []
        try:
            entities = json.loads(label)
            return entities if isinstance(entities, list) else []
        except:
            return []

    def compute_match_score(self, pred_label: str, gt_label: str) -> float:
        """
        Compute entity-level F1 score between predicted and ground truth PII entities.

        This method is used for per-sample accuracy evaluation instead of exact string match.

        Args:
            pred_label: JSON string of predicted PII entities
            gt_label: JSON string of ground truth PII entities

        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_entities = self.parse_label(pred_label)
        gt_entities = self.parse_label(gt_label)

        # Sanitize entities
        pred_entities = self._sanitize_entities(pred_entities)
        gt_entities = self._sanitize_entities(gt_entities)

        # Convert to sets for matching (text, label, occurrence)
        # Use case-insensitive text matching
        pred_set = {
            (e["text"].lower().strip(), e["label"], e["occurrence"])
            for e in pred_entities
        }
        gt_set = {
            (e["text"].lower().strip(), e["label"], e["occurrence"])
            for e in gt_entities
        }

        # Calculate F1
        if not gt_set and not pred_set:
            return 1.0  # Both empty = perfect match
        if not gt_set or not pred_set:
            return 0.0  # One empty, one not

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return f1

    def is_correct(self, pred_label: str, gt_label: str, threshold: float = 0.5) -> bool:
        """
        Check if prediction is correct based on F1 threshold.

        For PII task, we use entity-level F1 score instead of exact string match.

        Args:
            pred_label: JSON string of predicted PII entities
            gt_label: JSON string of ground truth PII entities
            threshold: F1 threshold for considering prediction as correct (default: 0.5)

        Returns:
            True if F1 score >= threshold, False otherwise
        """
        return self.compute_match_score(pred_label, gt_label) >= threshold

    def get_score_category(self, pred_label: str, gt_label: str) -> str:
        """
        Get F1 score category for confusion matrix bucketing.

        Args:
            pred_label: JSON string of predicted PII entities
            gt_label: JSON string of ground truth PII entities

        Returns:
            Category string like "F1_0%", "F1_10%", ..., "F1_100%"
        """
        score = self.compute_match_score(pred_label, gt_label)
        bucket = int(score * 10) * 10  # Bucket into 0%, 10%, ..., 100%
        return f"F1_{bucket}%"
