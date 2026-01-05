# Language-Grouped LLM Assignment Implementation

## Overview

This implementation adds language-aware expert selection for the `finance/rating` (sentiment analysis) task based on experimental findings from the research paper. The system now routes European languages (EN, DE, ES, FR) to one LLM and Asian languages (JA, ZH) to another LLM using a flexible configuration-based approach.

## Implementation Date

January 2, 2026

## Changes Made

### 1. Configuration Changes ([experts_registry.json](src/models/experts/config/experts_registry.json))

#### Added New Base Model
```json
"aya-23": {
  "hf_name": "CohereForAI/aya-23-8B",
  "load_in_4bit": true,
  "device_map": "auto"
}
```

#### Added Language Mapping to finance/rating Task
```json
"finance/rating": {
  "base_model_key": "llama-2-7b-hf",
  "adapter_name": "fin-sentiment",
  "adapter_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/llama-2-7b-hf/",
  "expert_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/SentimentAnalysisExpert",
  "template_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/template.json",
  "label_set": ["1","2","3","4","5"],
  "strict_label_decoding": true,
  "generation": { "max_new_tokens": 4, "temperature": 0.0, "top_p": 1.0 },
  "language_mapping": {
    "european": {
      "languages": ["english", "german", "spanish", "french"],
      "base_model_key": "llama-2-7b-hf",
      "adapter_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/llama-2-7b-hf/"
    },
    "asian": {
      "languages": ["japanese", "chinese"],
      "base_model_key": "aya-23",
      "adapter_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/aya-23/"
    }
  }
}
```

### 2. Code Changes ([expert_pool.py](src/models/experts/llms/expert_pool.py))

#### Added Language Resolution Method
New method `_resolve_base_model_for_language()` (lines 122-153):
- Resolves which base model and adapter to use based on language mapping
- Returns (base_model_key, adapter_path) tuple
- Fallback to default model if language not in mapping

#### Modified ensure_task_ready() Method
Updated signature (line 155):
```python
def ensure_task_ready(self, task_key: str, language: Optional[str] = None)
```

Updated implementation (lines 165-166):
- Calls `_resolve_base_model_for_language()` to get language-specific model
- Uses resolved adapter_path instead of static config

#### Modified generate() Method
Updated line 211:
```python
model, tok = self.ensure_task_ready(task_key, language=language)
```
- Passes language parameter to `ensure_task_ready()` for model selection

### 3. Directory Structure

Created adapter directory:
```
src/models/experts/llms/adapters/finance/sentiment_analysis/
├── llama-2-7b-hf/               ✅ Existing (European languages)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
├── aya-23/                       ✅ Created (Asian languages)
│   └── (awaiting adapter files)
├── SentimentAnalysisExpert.py
└── template.json
```

## Language Routing Logic

The system now routes prompts based on detected language:

```
Input Prompt
    ↓
Language Detection (FastText)
    ↓
┌────────────────────────────────────┐
│  Language Mapping Resolution       │
├────────────────────────────────────┤
│  European (en, de, es, fr)        │
│  └─→ llama-2-7b-hf                │
│      └─→ llama-2-7b-hf/ adapter   │
│                                    │
│  Asian (ja, zh)                   │
│  └─→ aya-23                       │
│      └─→ aya-23/ adapter          │
│                                    │
│  Unknown/None                     │
│  └─→ llama-2-7b-hf (default)     │
└────────────────────────────────────┘
    ↓
Expert Prediction
```

## Experimental Justification

Based on MARC dataset experiments (Table 4 from research paper):

### European Languages (Best: Llama-2-7b-hf)
- **English**: 68.8% F1 score
- **German**: 62.0% F1 score
- **Spanish**: 59.8% F1 score
- **French**: 60.4% F1 score

### Asian Languages (Best: aya-23)
- **Japanese**: 57.2% F1 score
- **Chinese**: 55.7% F1 score

## Files Modified

1. **[src/models/experts/config/experts_registry.json](src/models/experts/config/experts_registry.json)**
   - Lines 19-23: Added aya-23 base model
   - Lines 35-46: Added language_mapping to finance/rating

2. **[src/models/experts/llms/expert_pool.py](src/models/experts/llms/expert_pool.py)**
   - Lines 122-153: Added `_resolve_base_model_for_language()` method
   - Line 155: Updated `ensure_task_ready()` signature
   - Lines 165-166: Updated `ensure_task_ready()` implementation
   - Line 211: Updated `generate()` to pass language parameter

3. **Adapter directories**
   - Created: `src/models/experts/llms/adapters/finance/sentiment_analysis/aya-23/`

## Next Steps

### 1. Place aya-23 Adapter Files

Copy your pre-trained aya-23 adapters to:
```
src/models/experts/llms/adapters/finance/sentiment_analysis/aya-23/
```

Required files:
- `adapter_model.safetensors`
- `adapter_config.json`
- (Optional) tokenizer files

### 2. Test the Implementation

Run the router with sample prompts:

```python
from src.models.gating.without-translation.rl-based.qlearning-router.router1 import PromptRoutingSystem

system = PromptRoutingSystem()

# Test English (should route to llama-2-7b-hf)
result = system.route_prompt(
    "This product is excellent!",
    "This product is excellent!"
)
print(f"Language: {result['language']}, Domain: {result['domain']}, Task: {result['task']}")
# Expected: language='english', uses llama-2-7b-hf

# Test Japanese (should route to aya-23)
result = system.route_prompt(
    "この製品は素晴らしい",
    "この製品は素晴らしい"
)
print(f"Language: {result['language']}, Domain: {result['domain']}, Task: {result['task']}")
# Expected: language='japanese', uses aya-23
```

### 3. Validate Performance

Run evaluation on MARC test set to confirm F1 scores match experimental findings:

```bash
# Run the router1.py evaluation
python src/models/gating/without-translation/rl-based/qlearning-router/router1.py
```

Expected improvements:
- European languages: Maintain ~60-68% F1
- Asian languages: Improve to ~55-57% F1 (with aya-23)

### 4. Monitor and Tune

- Check logs for language → model routing decisions
- Look for `[LLMAdapterPool]` log messages showing routing decisions
- Adjust language_mapping if needed (e.g., add Bloomz for Chinese if aya-23 underperforms)

## Extensibility

To add more language groups or models:

1. Add new base model to `experts_registry.json`:
```json
"new-model": {
  "hf_name": "org/model-name",
  "load_in_4bit": true,
  "device_map": "auto"
}
```

2. Add new group to `language_mapping`:
```json
"new_group": {
  "languages": ["lang1", "lang2"],
  "base_model_key": "new-model",
  "adapter_path": "path/to/adapter/"
}
```

3. Create adapter directory and place trained adapters

4. No code changes needed!

## Configuration Flexibility

The implementation allows easy modification of language groupings:

- **More granular**: Separate each language
- **Broader groupings**: Combine all into 2-3 groups
- **Model swapping**: Change `base_model_key` to try different LLMs
- **Task extension**: Apply same pattern to `finance/news` or other tasks

Example: Use Bloomz for Chinese instead of aya-23:
```json
"asian": {
  "languages": ["japanese", "chinese"],
  "base_model_key": "bloomz-7b1",  // Change this
  "adapter_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/bloomz-7b1/"
}
```

## Backward Compatibility

- Existing code without language parameter continues to work (uses default model)
- Tasks without `language_mapping` fall back to `base_model_key`
- No breaking changes to router or expert interfaces

## Troubleshooting

### Issue: Model not switching based on language

**Check:**
1. Language detection is working: Check logs for detected language
2. Language mapping exists in config: Verify `language_mapping` field
3. Language name matches exactly: Use lowercase (e.g., "english", not "English")

**Debug:**
```python
# Check what language is detected
language = system.language_detector.detect_language(prompt)
print(f"Detected language: {language}")

# Check language resolution
base, path = pool._resolve_base_model_for_language("finance/rating", language)
print(f"Resolved to model: {base}")
```

### Issue: Adapter not loading

**Check:**
1. Adapter directory exists and has correct structure
2. Adapter files are present: `adapter_model.safetensors`, `adapter_config.json`
3. Path in config matches actual directory path

### Issue: Out of memory

**Solution:**
- Only 2 base models loaded simultaneously (Llama-2 + aya-23)
- Both are loaded in 4-bit quantization (~7GB total)
- If memory limited, load models on-demand or use smaller variants

## References

- Research paper: `Second_Paper.pdf` (Table 4, pages 8-9)
- Implementation plan: `/home/cse/.claude/plans/witty-bubbling-pixel.md`
- Verification script: `/home/cse/Desktop/verify_config.py`

## Authors

- Implementation: Claude Code (Anthropic)
- Based on research by: Chamod Neluhena, Adam Moraes, Dulitha Hasith, Nirodha Thisum, et al.
- Date: January 2, 2026


### Remove aya temporary
- When you run your tests:
```
Only Llama-2 will load (no Aya-23)
~8GB memory savings (Aya model won't be loaded)
Faster initialization (no Aya download/loading time)
All languages supported (Japanese and Chinese will use Llama-2 adapter)
Same routing logic (language detection and task classification unchanged)
```

- When you want to re-enable Aya:
`cp src/models/experts/config/experts_registry.json.backup src/models/experts/config/experts_registry.json
`
