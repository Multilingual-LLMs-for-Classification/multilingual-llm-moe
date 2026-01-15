# Dataset Preparation Summary

## Overview

Combined three task-specific datasets into unified format for router training and evaluation.

---

## Source Datasets

1. **Ratings Task**: `/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output/ratings_test.json`
   - Samples: 1,020
   - Languages: de, en, es, fr, ja, zh (6 languages)
   - Task: Sentiment analysis (1-5 star ratings)

2. **News Task**: `/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output/generated_prompts_news.json`
   - Samples: 3,200
   - Languages: Danish, English, Polish, Spanish, Turkish (5 languages)
   - Task: News classification (6 categories)

3. **PII Task**: `/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output/pii_prompts_generated.json`
   - Samples: 8,348
   - Languages: Dutch, English, French, German, Italian, Spanish, Swedish (7 languages)
   - Task: PII entity extraction

**Total**: 12,568 samples

---

## Unified Format

All samples converted to consistent structure:

```json
{
  "prompt": "Classification prompt with instructions",
  "classification_text": "Text to classify" OR "generated_text": "Text with PII",
  "language": "en",
  "task": "rating" | "news" | "pii",
  "domain": "finance",
  "label": "5" | "Finance" | '[{"text": "John", "label": "person_name", ...}]'
}
```

### Field Mapping

**Rating Task**:
- `prompt` ← template prompt
- `classification_text` ← review_text
- `language` ← language code (de, en, es, fr, ja, zh)
- `task` ← "rating"
- `domain` ← "finance"
- `label` ← stars (converted to string: "1"-"5")

**News Task**:
- `prompt` ← template prompt
- `classification_text` ← review_text (headline)
- `language` ← language code (da, en, es, po, tu)
- `task` ← "news"
- `domain` ← "finance"
- `label` ← generic_label (category name)

**PII Task**:
- `prompt` ← template prompt
- `generated_text` ← generated_text (document with PII)
- `language` ← language code (nl, en, fr, de, it, es, sv)
- `task` ← "pii"
- `domain` ← "finance"
- `label` ← pii_json (JSON string with entities)

---

## Language Code Standardization

Language names normalized to 2-letter codes:

| Original | Normalized |
|----------|------------|
| Danish | da |
| Dutch | nl |
| English | en |
| France/French | fr |
| German | de |
| Italian | it |
| Japanese | ja |
| Polish | po |
| Spanish | es |
| Swedish | sv |
| Turkish | tu |
| Chinese | zh |

---

## Dataset Split

**Split Ratio**: 80% train / 20% test
**Strategy**: Stratified by task to maintain balance

### Training Set (train_combined.json)
- **Total**: 10,054 samples
- **Rating**: 816 samples (8.1%)
  - de: 141, en: 129, es: 135, fr: 133, ja: 142, zh: 136
- **News**: 2,560 samples (25.5%)
  - da: 505, en: 509, es: 513, po: 504, tu: 529
- **PII**: 6,678 samples (66.4%)
  - de: 932, en: 1,104, es: 956, fr: 924, it: 919, nl: 911, sv: 932

### Test Set (test_combined.json)
- **Total**: 2,514 samples
- **Rating**: 204 samples (8.1%)
  - de: 29, en: 41, es: 35, fr: 37, ja: 28, zh: 34
- **News**: 640 samples (25.5%)
  - da: 135, en: 131, es: 127, po: 136, tu: 111
- **PII**: 1,670 samples (66.4%)
  - de: 230, en: 274, es: 210, fr: 232, it: 223, nl: 244, sv: 257

---

## Output Files

All files saved to: `/home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router/`

1. **train_combined.json** (10,054 samples)
   - For training domain classifier and Q-learning routers

2. **test_combined.json** (2,514 samples)
   - For evaluation

3. **full_combined.json** (12,568 samples)
   - Complete dataset for reference

---

## Configuration Update

Updated `router_config.json`:

```json
{
  "training": {
    "data_path": "train_combined.json",  // Changed from train1.json
    "enable_training": true
  },
  "evaluation": {
    "test_data_path": "test_combined.json",  // Changed from test2_grouped_languages_flat.json
    "test_n": null,  // Test all samples
    "output_path": "predictions_with_raw_responses.csv",
    "enable_evaluation": true
  }
}
```

---

## Task Distribution Analysis

### Imbalance Note
PII task dominates the dataset (66.4%), while rating task is underrepresented (8.1%).

**Implications**:
- Domain classifier will see balanced distribution (all tasks are finance domain)
- Task classifier needs to handle imbalanced classes
- Q-learning router will have more PII training samples

**Mitigation**:
- Class weighting enabled in domain_config
- Stratified splitting maintains proportions
- Each task has sufficient samples for training (min: 816 for rating)

---

## Verification Commands

```bash
# Check training set structure
cd /home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router
python -c "import json; data = json.load(open('train_combined.json')); print(f'Train: {len(data)} samples')"

# Check test set structure
python -c "import json; data = json.load(open('test_combined.json')); print(f'Test: {len(data)} samples')"

# Verify sample structure
python -c "
import json
data = json.load(open('train_combined.json'))
for task in ['rating', 'news', 'pii']:
    sample = next(s for s in data if s['task'] == task)
    print(f'{task}: {list(sample.keys())}')
"
```

---

## Next Steps

1. ✅ Datasets combined and split
2. ✅ Configuration updated
3. ⏭️  Train the system: `python main.py --mode train`
4. ⏭️  Evaluate: `python main.py --mode eval`

---

## Script Location

Dataset combination script: `/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output/combine_datasets.py`

To regenerate datasets:
```bash
cd /home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output
python combine_datasets.py
```
