# Expert-Specific Evaluation Implementation

## Overview

The evaluation system uses **expert-specific methods throughout the entire pipeline**. Each expert defines how to handle its task from input to evaluation:

- **Input Preparation**: `expert.prepare_input()` - Maps input fields to task format
- **Output Cleaning**: `expert.clean_output()` - Sanitizes and validates LLM output
- **Evaluation**: `expert.compute_metrics()` - Task-appropriate metrics

### Task-Specific Methods

- **Rating Task (Sentiment Analysis)**:
  - Prepares review text + title
  - Cleans to star rating (1-5)
  - Evaluates with MAE, RMSE for ordinal ratings

- **News Task (Classification)**:
  - Prepares news article text
  - Cleans to category label
  - Evaluates with per-class F1 scores and weighted metrics

- **PII Task (Entity Extraction)**:
  - Prepares generated text
  - Cleans to structured JSON entities
  - Evaluates with entity-level matching and per-label metrics

---

## 🎯 Problem Addressed

### Before: Generic Evaluation
```python
# Simple string comparison - same for all tasks
is_correct = (pred_label == gt_label)
```

**Issues:**
- ❌ Doesn't capture ordinal nature of star ratings (1-5)
- ❌ Doesn't handle PII entity-level matching
- ❌ Doesn't provide per-class metrics for news categories
- ❌ One-size-fits-all approach ignores task differences

### After: Expert-Specific Evaluation
```python
# Each expert provides task-appropriate evaluation
expert.compute_metrics(predictions, ground_truths)
```

**Benefits:**
- ✅ **Rating**: MAE/RMSE for ordinal distance (predicting "4" when truth is "5" is better than predicting "1")
- ✅ **News**: Per-class F1 scores show which categories work well
- ✅ **PII**: Entity-level matching with occurrence tracking
- ✅ Task-appropriate metrics for each problem type

---

## 📊 Evaluation Metrics by Task

### 1. Rating Task (Sentiment Analysis)

**Expert**: `SentimentAnalysisExpert`

**Metrics Computed:**
```python
{
    # Classification metrics
    "accuracy": 0.85,              # Exact match rate
    "macro_f1": 0.83,              # Average F1 across classes
    "micro_f1": 0.85,              # Overall F1

    # Ordinal metrics (ratings as continuous values)
    "mae": 0.42,                   # Mean Absolute Error (in stars)
    "rmse": 0.58,                  # Root Mean Squared Error

    # Quality metrics
    "coverage": 0.98,              # % of valid predictions
    "total_samples": 500,
    "valid_predictions": 490
}
```

**Why MAE/RMSE Matter:**
- Predicting "4" when truth is "5" → MAE = 1
- Predicting "1" when truth is "5" → MAE = 4
- Shows how "close" predictions are to ground truth

**Example Output:**
```
📊 TASK-SPECIFIC EXPERT EVALUATION
================================================================================

🔍 Task: RATING
--------------------------------------------------------------------------------
  Samples evaluated: 500

  📈 Classification Metrics:
    Accuracy           : 85.20%
    Macro  P/R/F1      : 82.50% / 83.10% / 82.80%
    Micro  P/R/F1      : 85.20% / 85.20% / 85.20%

  📉 Ordinal Metrics (star ratings as continuous):
    MAE (Mean Abs Err) : 0.4200 stars
    RMSE               : 0.5800 stars
    Coverage           : 98.00%
```

---

### 2. News Task (Classification)

**Expert**: `NewsClassificationExpert`

**Metrics Computed:**
```python
{
    # Overall metrics
    "accuracy": 0.88,
    "macro_f1": 0.86,
    "weighted_f1": 0.87,
    "coverage": 0.99,

    # Per-class F1 scores
    "per_class_f1": {
        "Finance": 0.92,
        "Politics": 0.85,
        "Technology": 0.90,
        "Sports": 0.88,
        "Entertainment": 0.82,
        "Health": 0.79,
        "World": 0.84
    }
}
```

**Why Per-Class Metrics Matter:**
- Shows which categories are easy/hard
- Identifies class imbalance issues
- Helps debug category-specific problems

**Example Output:**
```
🔍 Task: NEWS
--------------------------------------------------------------------------------
  Samples evaluated: 350

  📈 Classification Metrics:
    Accuracy           : 88.00%
    Macro F1           : 86.00%
    Weighted F1        : 87.00%
    Coverage           : 99.00%

  📊 Per-Class F1 Scores:
    Finance              : 92.00%
    Technology           : 90.00%
    Sports               : 88.00%
    Politics             : 85.00%
    World                : 84.00%
    Entertainment        : 82.00%
    Health               : 79.00%
```

---

### 3. PII Task (Entity Extraction)

**Expert**: `PIIExpert`

**Metrics Computed:**
```python
{
    # Entity-level metrics
    "micro_f1": 0.85,
    "macro_f1": 0.82,
    "total_gold_entities": 450,
    "total_predicted_entities": 470,

    # Per-label metrics (for each PII type)
    "per_label_metrics": {
        "PERSON": {"precision": 0.92, "recall": 0.89, "f1": 0.90},
        "EMAIL": {"precision": 0.95, "recall": 0.93, "f1": 0.94},
        "PHONE": {"precision": 0.88, "recall": 0.85, "f1": 0.86},
        "CREDIT_CARD": {"precision": 0.90, "recall": 0.87, "f1": 0.88},
        "SSN": {"precision": 0.85, "recall": 0.82, "f1": 0.83},
        # ... other PII types
    }
}
```

**Why Entity-Level Matching Matters:**
- Matches entities by (text, label, occurrence)
- Handles multiple occurrences of same entity
- Case-insensitive matching for robustness

**Example Output:**
```
🔍 Task: PII
--------------------------------------------------------------------------------
  Samples evaluated: 200

  📈 Entity-Level Metrics:
    Micro F1           : 85.00%
    Macro F1           : 82.00%
    Total gold entities: 450
    Total pred entities: 470

  📊 Per-Label Metrics:
    Label                | Precision  |     Recall |         F1
    --------------------------------------------------------
    EMAIL                |     95.00% |     93.00% |     94.00%
    PERSON               |     92.00% |     89.00% |     90.00%
    CREDIT_CARD          |     90.00% |     87.00% |     88.00%
    PHONE                |     88.00% |     85.00% |     86.00%
    SSN                  |     85.00% |     82.00% |     83.00%
```

---

## 🔧 Implementation Details

### 1. Data Collection

The evaluation loop collects task-specific predictions:

```python
# Per-task evaluation data (for expert-specific metrics)
per_task_data = {
    'rating': {'predictions': [], 'ground_truths': []},
    'news': {'predictions': [], 'ground_truths': []},
    'pii': {'predictions': [], 'ground_truths': []}
}

# In evaluation loop:
# Only collect if routing was correct (otherwise task output is meaningless)
if both_ok and gt_task in per_task_data:
    per_task_data[gt_task]['predictions'].append(pred_label)
    per_task_data[gt_task]['ground_truths'].append(gt_label)
```

**Why "both_ok" check?**
- If routing is wrong (wrong domain or task), the expert output is meaningless
- Only evaluate task performance on correctly routed samples
- This isolates routing errors from task performance errors

---

### 2. Expert-Specific Metrics Computation

```python
def compute_task_specific_metrics(system: PromptRoutingSystem,
                                   per_task_data: Dict[str, Dict]) -> Dict:
    """
    Compute expert-specific evaluation metrics for each task.
    """
    task_specific_metrics = {}

    for task_name, data in per_task_data.items():
        predictions = data['predictions']
        ground_truths = data['ground_truths']

        # Get the appropriate expert
        if task_name == 'rating':
            from src.models.experts.llms.adapters.finance.sentiment_analysis.SentimentAnalysisExpert import SentimentAnalysisExpert
            expert = SentimentAnalysisExpert()
        elif task_name == 'news':
            from src.models.experts.llms.adapters.finance.news_classification.NewsClassificationExpert import NewsClassificationExpert
            expert = NewsClassificationExpert()
        elif task_name == 'pii':
            from src.models.experts.llms.adapters.finance.pii.PIIExpert import PIIExpert
            expert = PIIExpert()

        # Compute task-specific metrics using expert's method
        metrics = expert.compute_metrics(predictions, ground_truths)
        task_specific_metrics[task_name] = {
            'status': 'success',
            'metrics': metrics,
            'num_samples': len(predictions)
        }

    return task_specific_metrics
```

---

### 3. Metric Display

The `print_task_specific_metrics()` function formats output appropriately for each task type:

```python
if task_name == 'rating':
    # Show both classification AND ordinal metrics
    print("Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print()
    print("Ordinal Metrics:")
    print(f"  MAE: {metrics['mae']:.4f} stars")
    print(f"  RMSE: {metrics['rmse']:.4f} stars")

elif task_name == 'news':
    # Show overall + per-class breakdown
    print("Classification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print()
    print("Per-Class F1 Scores:")
    for label, f1 in sorted(metrics['per_class_f1'].items()):
        print(f"  {label}: {f1}")

elif task_name == 'pii':
    # Show entity-level metrics + per-label breakdown
    print("Entity-Level Metrics:")
    print(f"  Micro F1: {metrics['micro_f1']}")
    print()
    print("Per-Label Metrics:")
    for label, label_metrics in metrics['per_label_metrics'].items():
        print(f"  {label}: P={label_metrics['precision']}, R={label_metrics['recall']}, F1={label_metrics['f1']}")
```

---

## 📈 Complete Expert-Driven Evaluation Flow

```
1. Load test data
   ↓
2. For each sample:
   a) Route to correct expert (Language → Domain → Task)

   b) Expert prepares input:
      ⭐ expert.prepare_input(input_data)
      - Rating: Extracts "text" + "title" fields
      - News: Extracts "classification_text" field
      - PII: Extracts "generated_text" field

   c) LLM generates output

   d) Expert cleans output:
      ⭐ expert.clean_output(raw_output)
      - Rating: Extracts "1"-"5" from text
      - News: Maps to one of 7 categories
      - PII: Parses JSON, validates entities

   e) Track routing accuracy and collect predictions
   ↓
3. Compute routing metrics:
   - Domain classification accuracy
   - Task classification accuracy
   - Overall expert output accuracy (generic)
   ↓
4. Compute task-specific metrics:
   ⭐ expert.compute_metrics(predictions, ground_truths)
   - Rating: MAE, RMSE, F1
   - News: Per-class F1, Weighted F1
   - PII: Entity-level F1, Per-label metrics
   ↓
5. Print results:
   - Routing metrics (Language → Domain → Task)
   - Generic expert output metrics
   - ⭐ Task-specific expert metrics (from expert.compute_metrics)
   - Confusion matrices
   - Per-language breakdown
   - Per-expert analysis
   ↓
6. Save CSV with all predictions
```

### Expert Methods Used Throughout

| Stage | Method | Purpose | Example (Rating Task) |
|-------|--------|---------|----------------------|
| **Input Prep** | `expert.prepare_input()` | Extract relevant fields | Gets "text" + "title" from input_data |
| **Output Clean** | `expert.clean_output()` | Sanitize LLM output | Extracts "3" from "The rating is 3 stars" |
| **Validation** | `expert.is_valid_prediction()` | Check if output is valid | Checks if result is in ["1", "2", "3", "4", "5"] |
| **Ground Truth** | `expert.get_ground_truth()` | Extract expected label | Gets "label" field from input_data |
| **Evaluation** | `expert.compute_metrics()` | Calculate metrics | Returns accuracy, MAE, RMSE, F1 |

**Key Insight**: The expert class is the **single source of truth** for how to handle each task!

---

## 🎯 Key Benefits

### 1. **Separation of Concerns**
- **Routing evaluation**: How well does the router select the right expert?
- **Task evaluation**: How well does the expert perform its task?

### 2. **Task-Appropriate Metrics**
- Rating task gets ordinal metrics (MAE/RMSE)
- News task gets per-class analysis
- PII task gets entity-level matching

### 3. **Debugging Capabilities**
```
Example scenario:
- Domain accuracy: 95% ✅
- Task accuracy: 88% ✅
- Rating MAE: 0.42 stars ✅
- Rating RMSE: 0.58 stars ✅

Conclusion: Router works well, expert is accurate within 0.5 stars on average
```

vs

```
Example scenario:
- Domain accuracy: 60% ❌
- Task accuracy: 45% ❌
- Rating MAE: 2.5 stars ❌
- Rating RMSE: 3.2 stars ❌

Conclusion: Router is failing - expert can't perform well on wrong tasks
```

### 4. **Expert-Driven Evaluation**
Each expert defines what "good performance" means for its task:
- Sentiment expert knows ratings are ordinal
- News expert knows which categories exist
- PII expert knows how to match entities

---

## 📊 Example Complete Output

```
================================================================================
ROUTING SYSTEM EVALUATION RESULTS
================================================================================

📊 Domain Classification:
  Accuracy           : 95.23%
  Macro  P/R/F1      : 94.50% / 95.00% / 94.75%

📊 Task Classification:
  Accuracy           : 88.45%
  Macro  P/R/F1      : 87.20% / 88.10% / 87.65%

📊 Expert Output Classification:
  Accuracy           : 82.15%
  Macro  P/R/F1      : 80.30% / 81.50% / 80.90%

================================================================================
📊 TASK-SPECIFIC EXPERT EVALUATION
================================================================================

🔍 Task: RATING
--------------------------------------------------------------------------------
  Samples evaluated: 500

  📈 Classification Metrics:
    Accuracy           : 85.20%
    Macro  P/R/F1      : 82.50% / 83.10% / 82.80%
    Micro  P/R/F1      : 85.20% / 85.20% / 85.20%

  📉 Ordinal Metrics (star ratings as continuous):
    MAE (Mean Abs Err) : 0.4200 stars
    RMSE               : 0.5800 stars
    Coverage           : 98.00%

🔍 Task: NEWS
--------------------------------------------------------------------------------
  Samples evaluated: 350

  📈 Classification Metrics:
    Accuracy           : 88.00%
    Macro F1           : 86.00%
    Weighted F1        : 87.00%
    Coverage           : 99.00%

  📊 Per-Class F1 Scores:
    Finance              : 92.00%
    Technology           : 90.00%
    Sports               : 88.00%
    Politics             : 85.00%
    World                : 84.00%
    Entertainment        : 82.00%
    Health               : 79.00%

🔍 Task: PII
--------------------------------------------------------------------------------
  Samples evaluated: 200

  📈 Entity-Level Metrics:
    Micro F1           : 85.00%
    Macro F1           : 82.00%
    Total gold entities: 450
    Total pred entities: 470

  📊 Per-Label Metrics:
    Label                | Precision  |     Recall |         F1
    --------------------------------------------------------
    EMAIL                |     95.00% |     93.00% |     94.00%
    PERSON               |     92.00% |     89.00% |     90.00%
    CREDIT_CARD          |     90.00% |     87.00% |     88.00%
    PHONE                |     88.00% |     85.00% |     86.00%
    SSN                  |     85.00% |     82.00% |     83.00%

[... rest of evaluation output ...]
```

---

## 🔄 Migration Path

### Old Code (Generic Evaluation)
```python
is_correct = (pred_label == gt_label)
accuracy = sum(is_correct) / len(predictions)
```

### New Code (Expert-Specific Evaluation)
```python
# Still compute generic accuracy for overall view
is_correct = (pred_label == gt_label)

# PLUS collect task-specific data
if both_ok and task in per_task_data:
    per_task_data[task]['predictions'].append(pred)
    per_task_data[task]['ground_truths'].append(gt)

# Then compute expert-specific metrics
expert = get_expert_for_task(task)
task_metrics = expert.compute_metrics(predictions, ground_truths)
```

---

## 📝 Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Evaluation Method** | Generic string comparison | Expert-specific metrics |
| **Rating Metrics** | Accuracy only | Accuracy + MAE + RMSE |
| **News Metrics** | Overall accuracy | Accuracy + per-class F1 |
| **PII Metrics** | String match | Entity-level matching + per-label metrics |
| **Debugging** | Hard to identify issues | Clear separation: routing vs task performance |
| **Task-Appropriate** | No | Yes - each expert defines success |

**The evaluation system now provides task-appropriate metrics that match how each expert actually performs its work!** ✅
