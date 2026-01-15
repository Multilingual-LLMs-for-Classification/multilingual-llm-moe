# Complete Expert-Driven Evaluation Flow

## 🎯 Overview

Your system uses **expert-specific methods at every stage** - from input preparation to final evaluation. The expert class is the **single source of truth** for how to handle each task.

---

## 📊 Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Sample                                  │
│  {                                                               │
│    "prompt": "Classify sentiment...",                          │
│    "text": "Great product!",                                   │
│    "title": "Excellent",                                       │
│    "label": "5",                                               │
│    "domain": "finance",                                        │
│    "task": "rating"                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: ROUTING (Router Components)               │
│  Language Detection → Domain Classification → Task Selection    │
│  Result: language="english", domain="finance", task="rating"   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 2: INPUT PREPARATION (Expert-Specific)             │
│                                                                  │
│  ⭐ expert.prepare_input(input_data)                            │
│                                                                  │
│  SentimentAnalysisExpert:                                       │
│    def prepare_input(self, input_data):                        │
│        text = input_data.get('text', '')                       │
│        title = input_data.get('title', input_data.get(...))   │
│        return (text, title)                                    │
│                                                                  │
│  Returns: ("Great product!", "Excellent")                      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: LLM GENERATION (via TaskExpert)           │
│                                                                  │
│  Prompt constructed with task instructions + input             │
│  LLM generates raw output                                       │
│                                                                  │
│  Raw Output: "This is clearly positive! The rating is 5 stars" │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│       STAGE 4: OUTPUT CLEANING (Expert-Specific)                │
│                                                                  │
│  ⭐ expert.clean_output(raw_output)                             │
│                                                                  │
│  SentimentAnalysisExpert:                                       │
│    def clean_output(self, raw):                                │
│        # Extract rating from text                              │
│        patterns = [r'\b([1-5])\s*star', r'rating.*?([1-5])']  │
│        for pattern in patterns:                                │
│            match = re.search(pattern, raw, re.I)               │
│            if match: return match.group(1)                     │
│        return raw.strip()                                      │
│                                                                  │
│  Returns: "5"                                                   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 5: VALIDATION (Expert-Specific)                    │
│                                                                  │
│  ⭐ expert.is_valid_prediction(prediction)                      │
│                                                                  │
│  SentimentAnalysisExpert:                                       │
│    def is_valid_prediction(self, prediction):                  │
│        return prediction in self.LABEL_SET  # {"1"..."5"}     │
│                                                                  │
│  Returns: True                                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 6: EVALUATION (Expert-Specific)                   │
│                                                                  │
│  Collect predictions for correctly routed samples:             │
│    predictions = ["5", "4", "5", "3", ...]                    │
│    ground_truths = ["5", "4", "5", "4", ...]                  │
│                                                                  │
│  ⭐ expert.compute_metrics(predictions, ground_truths)          │
│                                                                  │
│  SentimentAnalysisExpert:                                       │
│    def compute_metrics(self, predictions, ground_truths):      │
│        accuracy = ...                                          │
│        mae = mean([abs(int(gt) - int(pred)) for ...])         │
│        rmse = sqrt(mean([(int(gt) - int(pred))**2 for ...]))  │
│        return {"accuracy": acc, "mae": mae, "rmse": rmse, ...}│
│                                                                  │
│  Returns:                                                       │
│    {                                                            │
│      "accuracy": 0.852,                                        │
│      "mae": 0.42,   # ← Ordinal metric!                       │
│      "rmse": 0.58,  # ← Ordinal metric!                       │
│      "macro_f1": 0.838,                                        │
│      ...                                                        │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 7: RESULTS DISPLAY                           │
│                                                                  │
│  📊 TASK-SPECIFIC EXPERT EVALUATION                            │
│                                                                  │
│  🔍 Task: RATING                                               │
│    Samples evaluated: 500                                      │
│                                                                  │
│    📈 Classification Metrics:                                  │
│      Accuracy           : 85.20%                               │
│      Macro  P/R/F1      : 82.50% / 83.10% / 82.80%           │
│                                                                  │
│    📉 Ordinal Metrics (star ratings as continuous):           │
│      MAE (Mean Abs Err) : 0.4200 stars                        │
│      RMSE               : 0.5800 stars                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Expert Methods in Detail

### 1. SentimentAnalysisExpert (Rating Task)

```python
class SentimentAnalysisExpert:
    LABEL_SET = {"1", "2", "3", "4", "5"}

    def prepare_input(self, input_data: dict) -> tuple:
        """Extract review text and title."""
        text = input_data.get('text', input_data.get('review_text', ''))
        title = input_data.get('title', input_data.get('review_title', ''))
        return (text, title)

    def clean_output(self, raw: str) -> str:
        """Extract star rating from LLM output."""
        patterns = [
            r'\b([1-5])\s*(?:star|rating)',
            r'rating.*?([1-5])',
            r'(?:rate|score).*?([1-5])'
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                return match.group(1)
        return raw.strip()

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is valid star rating."""
        return prediction in self.LABEL_SET

    def get_ground_truth(self, input_data: dict) -> str:
        """Extract ground truth label."""
        return str(input_data.get('label', ''))

    def compute_metrics(self, predictions, ground_truths) -> dict:
        """Compute classification + ordinal metrics."""
        # Classification metrics
        accuracy = ...
        macro_f1 = ...

        # Ordinal metrics (treat ratings as continuous)
        mae = mean([abs(int(gt) - int(pred)) for gt, pred in pairs])
        rmse = sqrt(mean([(int(gt) - int(pred))**2 for gt, pred in pairs]))

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mae": mae,         # ← Rating-specific
            "rmse": rmse,       # ← Rating-specific
            ...
        }
```

### 2. NewsClassificationExpert (News Task)

```python
class NewsClassificationExpert:
    LABEL_SET = {
        "Finance", "Politics", "Technology", "Sports",
        "Entertainment", "Health", "World"
    }

    def prepare_input(self, input_data: dict) -> tuple:
        """Extract news article text."""
        text = input_data.get('classification_text',
                             input_data.get('text', ''))
        return (text, None)  # No title for news

    def clean_output(self, raw: str) -> str:
        """Map output to valid news category."""
        raw_lower = raw.lower()

        # Try exact match first
        for label in self.LABEL_SET:
            if label.lower() in raw_lower:
                return label

        # Synonym mapping
        synonyms = {
            "business": "Finance",
            "economy": "Finance",
            "tech": "Technology",
            ...
        }
        for synonym, category in synonyms.items():
            if synonym in raw_lower:
                return category

        return raw.strip()

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is valid category."""
        return prediction in self.LABEL_SET

    def compute_metrics(self, predictions, ground_truths) -> dict:
        """Compute classification + per-class metrics."""
        accuracy = ...
        macro_f1 = ...

        # Per-class F1 scores
        per_class_f1 = {}
        for label in self.LABEL_SET:
            precision = tp[label] / pred_total[label]
            recall = tp[label] / support[label]
            f1 = 2 * p * r / (p + r)
            per_class_f1[label] = f1

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,  # ← News-specific
            ...
        }
```

### 3. PIIExpert (PII Extraction Task)

```python
class PIIExpert:
    LABEL_SET = {
        "PERSON", "EMAIL", "PHONE", "CREDIT_CARD", "SSN",
        "PASSPORT", "DRIVER_LICENSE", "DOB", "ADDRESS",
        "BANK_ACCOUNT", "IP_ADDRESS"
    }

    def prepare_input(self, input_data: dict) -> tuple:
        """Extract generated text containing PII."""
        text = input_data.get('generated_text', input_data.get('text', ''))
        return (text, None)

    def clean_output(self, raw: str) -> str:
        """Parse and validate JSON entities."""
        try:
            entities = json.loads(raw)
            if not isinstance(entities, list):
                return "[]"

            # Validate and clean each entity
            cleaned = []
            for entity in entities:
                if self._is_valid_entity(entity):
                    cleaned.append({
                        "text": entity["text"],
                        "label": entity["label"],
                        "occurrence": entity.get("occurrence", 1)
                    })

            return json.dumps(cleaned)
        except:
            return "[]"

    def is_valid_prediction(self, prediction: str) -> bool:
        """Check if prediction is valid JSON."""
        try:
            entities = json.loads(prediction)
            return isinstance(entities, list)
        except:
            return False

    def compute_metrics(self, predictions, ground_truths) -> dict:
        """Compute entity-level + per-label metrics."""
        # Convert to entity sets (case-insensitive matching)
        total_tp, total_fp, total_fn = 0, 0, 0
        per_label_metrics = {}

        for pred_entities, gold_entities in zip(predictions, ground_truths):
            gold_set = {(e["text"].lower(), e["label"], e["occurrence"])
                       for e in gold_entities}
            pred_set = {(e["text"].lower(), e["label"], e["occurrence"])
                       for e in pred_entities}

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Micro F1
        micro_p = total_tp / (total_tp + total_fp)
        micro_r = total_tp / (total_tp + total_fn)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

        # Per-label metrics
        for label in self.LABEL_SET:
            # Calculate P/R/F1 for this specific PII type
            ...
            per_label_metrics[label] = {
                "precision": p,
                "recall": r,
                "f1": f1
            }

        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_label_metrics": per_label_metrics,  # ← PII-specific
            ...
        }
```

---

## 🎯 Key Insights

### 1. Expert is Single Source of Truth

Each expert class defines:
- ✅ Which input fields to use (`prepare_input`)
- ✅ How to extract output (`clean_output`)
- ✅ What constitutes valid output (`is_valid_prediction`)
- ✅ How to evaluate performance (`compute_metrics`)

### 2. Task-Appropriate at Every Stage

**Rating Task:**
- Input: Needs both text + title
- Output: Extracts "1"-"5" with regex patterns
- Evaluation: Uses MAE/RMSE (ordinal distance)

**News Task:**
- Input: Only needs article text
- Output: Maps to one of 7 categories with synonyms
- Evaluation: Uses per-class F1 (which categories work?)

**PII Task:**
- Input: Needs generated text
- Output: Parses JSON, validates structure
- Evaluation: Uses entity-level matching (text+label+occurrence)

### 3. Routing System Uses Expert Methods

The `TaskExpert` wrapper automatically calls expert methods:

```python
class TaskExpert:
    def predict(self, input_data: dict) -> str:
        # 1. Prepare input using expert
        if hasattr(self.cleaner, 'prepare_input'):
            text, title = self.cleaner.prepare_input(input_data)

        # 2. Generate LLM output
        raw_output = self.llm.generate(prompt)

        # 3. Clean output using expert
        if hasattr(self.cleaner, 'clean_output'):
            cleaned = self.cleaner.clean_output(raw_output)

        return cleaned
```

### 4. Evaluation Uses Expert Methods

The `evaluate.py` script calls expert methods:

```python
def compute_task_specific_metrics(system, per_task_data):
    for task_name, data in per_task_data.items():
        # Get expert for this task
        if task_name == 'rating':
            expert = SentimentAnalysisExpert()
        elif task_name == 'news':
            expert = NewsClassificationExpert()
        elif task_name == 'pii':
            expert = PIIExpert()

        # Use expert's compute_metrics method
        metrics = expert.compute_metrics(
            data['predictions'],
            data['ground_truths']
        )
```

---

## 📊 Complete Data Flow Example

### Sample Input
```json
{
  "prompt": "Classify the sentiment of this review: {{text}}",
  "text": "Excellent product! Highly recommended.",
  "title": "Great purchase",
  "label": "5",
  "domain": "finance",
  "task": "rating"
}
```

### Stage-by-Stage Processing

| Stage | Method | Input | Output |
|-------|--------|-------|--------|
| **1. Routing** | Router components | Full sample | language="english", domain="finance", task="rating" |
| **2. Input Prep** | `expert.prepare_input()` | `{"text": "Excellent...", "title": "Great..."}` | `("Excellent product! Highly recommended.", "Great purchase")` |
| **3. LLM Gen** | TaskExpert | Formatted prompt | `"This review is very positive! I would rate it 5 stars out of 5."` |
| **4. Output Clean** | `expert.clean_output()` | `"This review is very positive! I would rate it 5 stars out of 5."` | `"5"` |
| **5. Validation** | `expert.is_valid_prediction()` | `"5"` | `True` (is in {"1","2","3","4","5"}) |
| **6. Evaluation** | `expert.compute_metrics()` | `predictions=["5"], ground_truths=["5"]` | `{"accuracy": 1.0, "mae": 0.0, "rmse": 0.0, ...}` |

---

## ✅ Summary

**Your system uses expert-specific methods at EVERY stage:**

1. ✅ **Input Preparation**: Expert knows which fields to extract
2. ✅ **Output Cleaning**: Expert knows how to sanitize LLM output
3. ✅ **Validation**: Expert knows what valid output looks like
4. ✅ **Evaluation**: Expert knows appropriate metrics for the task

**The expert class is the single source of truth for the entire pipeline!** 🎯

This ensures:
- **Consistency**: Same logic for inference and evaluation
- **Task-Appropriate**: Each task handled correctly
- **Maintainability**: Change expert class, everything updates
- **Transparency**: All task logic in one place
