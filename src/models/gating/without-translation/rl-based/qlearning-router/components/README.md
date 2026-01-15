# Routing System Components

This directory contains the modular components of the hierarchical routing system.

## Component Overview

### 1. `language_detector.py` (265 lines)
**Purpose:** FastText-based language identification

**Key Class:** `LanguageDetector`

**Capabilities:**
- Detects 176 languages using FastText model
- Dynamic language loading from expert registry
- Fallback detection using Unicode patterns and keywords
- Supports all ISO 639-1 language codes

**Usage:**
```python
from components import LanguageDetector

detector = LanguageDetector(registry_path="path/to/experts_registry.json")
language = detector.detect_language("Hello world")
# Returns: "english"
```

---

### 2. `domain_classifier.py` (393 lines)
**Purpose:** XLM-RoBERTa-based domain classification

**Key Classes:**
- `_DomainDataset` - PyTorch Dataset for domain classification
- `DomainClassifier` - Main classifier (nn.Module)

**Capabilities:**
- Multilingual domain classification using XLM-RoBERTa
- Prototype ensembling for stable predictions
- Optional encoder freezing for fast training
- Class weighting for imbalanced datasets
- Model persistence (save/load)

**Usage:**
```python
from components import DomainClassifier

classifier = DomainClassifier(model_name="xlm-roberta-base")

# Training
classifier.fit_from_labeled_prompts(
    data=[{"prompt": "...", "domain": "finance"}, ...],
    epochs=3,
    batch_size=32
)

# Inference
domain = classifier.classify_domain("What is the stock price?")
# Returns: "finance"
```

---

### 3. `q_learning_router.py` (290 lines)
**Purpose:** Q-learning based task selection

**Key Classes:**
- `TransformersEncoder` - XLM-RoBERTa encoder wrapper
- `QRouter` - Q-network (nn.Module)
- `DomainTaskDataset` - PyTorch Dataset
- `QLearningTaskClassifier` - Main Q-learning coordinator

**Capabilities:**
- Reinforcement learning for task selection
- Per-domain task routers
- Epsilon-greedy exploration
- Validation split during training

**Usage:**
```python
from components import QLearningTaskClassifier

classifier = QLearningTaskClassifier(encoder_name="xlm-roberta-base")

# Training
classifier.train_task_routers(
    data=[{"prompt": "...", "domain": "finance", "task": "rating"}, ...],
    val_split=0.1
)

# Inference
task, q_values = classifier.select_task(
    prompt="Rate this product",
    domain="finance",
    epsilon=0.1
)
```

---

### 4. `routing_system.py` (420 lines)
**Purpose:** Main routing orchestrator

**Key Class:** `PromptRoutingSystem`

**Capabilities:**
- Coordinates all routing components
- Hierarchical routing pipeline:
  1. Language Detection
  2. Domain Classification
  3. Task Selection (Q-learning)
  4. Expert Execution
- Manages expert pool (LLMAdapterPool)
- Loads domain/task configurations

**Usage:**
```python
from components import PromptRoutingSystem

system = PromptRoutingSystem()

# Training
system.train_domain_classifier(training_data, epochs=3)
system.train_task_routers(training_data, val_split=0.1)

# Inference
result = system.route_prompt(
    prompt="Classify the sentiment",
    input_data={"text": "Great product!", "title": "Excellent"}
)

print(result)
# {
#     "language": "english",
#     "domain": "finance",
#     "task": "rating",
#     "result": "5",
#     "confidence": 0.95,
#     ...
# }
```

---

## Import Patterns

### Recommended (New Code)
```python
# Import from components package
from components import (
    LanguageDetector,
    DomainClassifier,
    QLearningTaskClassifier,
    PromptRoutingSystem
)
```

### Individual Imports
```python
from components.language_detector import LanguageDetector
from components.domain_classifier import DomainClassifier
from components.q_learning_router import QLearningTaskClassifier
from components.routing_system import PromptRoutingSystem
```

### Backward Compatible (Still Supported)
```python
# Old imports still work via router1.py compatibility layer
from router1 import PromptRoutingSystem, LanguageDetector
```

---

## Dependencies

### Common Dependencies
- `torch` - Neural network framework
- `transformers` - XLM-RoBERTa models
- `numpy` - Numerical operations

### Component-Specific
- **language_detector.py**: `fasttext`, `requests`
- **domain_classifier.py**: `torch.utils.data`, `transformers.AutoModel`
- **q_learning_router.py**: Same as domain_classifier
- **routing_system.py**: All above + expert pool modules

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  PromptRoutingSystem                     │
│                  (routing_system.py)                     │
└──────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Language     │  │ Domain       │  │ Q-Learning       │
│ Detector     │  │ Classifier   │  │ Task Classifier  │
│              │  │              │  │                  │
│ (language_   │  │ (domain_     │  │ (q_learning_     │
│  detector)   │  │  classifier) │  │  router)         │
└──────────────┘  └──────────────┘  └──────────────────┘
```

---

## Testing

Each component can be tested independently:

```python
# Test language detection
def test_language_detector():
    detector = LanguageDetector()
    assert detector.detect_language("Hello") == "english"
    assert detector.detect_language("Bonjour") == "french"

# Test domain classification
def test_domain_classifier():
    classifier = DomainClassifier()
    classifier.load_model("checkpoint.pt")
    assert classifier.classify_domain("stock price") == "finance"

# Test Q-learning router
def test_q_router():
    router = QLearningTaskClassifier()
    router.load_models("models/")
    task, _ = router.select_task("Rate this", "finance", epsilon=0.0)
    assert task in ["rating", "news", "pii"]
```

---

## File Metrics

| File | Lines | Classes | Key Responsibility |
|------|-------|---------|-------------------|
| `__init__.py` | 40 | - | Component exports |
| `language_detector.py` | 265 | 1 | Language identification |
| `domain_classifier.py` | 393 | 2 | Domain classification |
| `q_learning_router.py` | 290 | 4 | Task selection (Q-learning) |
| `routing_system.py` | 420 | 1 | Routing orchestration |
| **Total** | **1,408** | **8** | **Complete routing system** |

---

## Migration from Monolith

The original `router1.py` (1286 lines) has been split into these components.

**Benefits:**
- ✅ Smaller files (265-420 lines vs 1286 lines)
- ✅ Single responsibility per module
- ✅ Easier to test and maintain
- ✅ Clear dependency structure
- ✅ Better code organization

**Backward Compatibility:**
- Old imports via `router1.py` still work
- No breaking changes
- Gradual migration path available

---

## Related Files

- `../evaluation_metrics.py` - Evaluation utilities (Phase 1 refactoring)
- `../router_config.py` - Configuration system (Phase 2 refactoring)
- `../train.py` - Training orchestration (Phase 3 refactoring)
- `../evaluate.py` - Evaluation orchestration (Phase 3 refactoring)
- `../main.py` - CLI entry point (Phase 3 refactoring)

---

## Documentation

For complete refactoring documentation, see:
- `../ROUTER_COMPONENT_SPLIT.md` - Component split details (Phase 4)
- `../PHASE3_REFACTORING_COMPLETE.md` - Train/eval split details (Phase 3)
- `../ROUTER_REFACTORING_SUMMARY.md` - Phases 1 & 2 details
- `../EXPERT_REFACTORING_SUMMARY.md` - Expert refactoring details
