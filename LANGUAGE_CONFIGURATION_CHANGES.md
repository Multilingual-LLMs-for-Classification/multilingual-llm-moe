# Language Configuration Changes

## Overview

Moved language definitions from hardcoded values in `LanguageDetector` to task-specific configurations in `experts_registry.json`. This allows each task to specify which languages it supports, making the system more flexible and maintainable.

---

## Changes Made

### 1. Updated `experts_registry.json`

**Location**: `/home/cse/Desktop/multilingual-llm-moe/src/models/experts/config/experts_registry.json`

Added `supported_languages` field to each task:

```json
{
  "tasks": {
    "finance/rating": {
      "supported_languages": ["de", "en", "es", "fr", "ja", "zh"],
      ...
    },
    "finance/pii": {
      "supported_languages": ["de", "en", "es", "fr", "ja", "zh"],
      ...
    },
    "finance/news": {
      "supported_languages": ["de", "en", "es", "fr", "ja", "zh"],
      ...
    },
    "general/text_summarization": {
      "supported_languages": ["en"],
      ...
    }
  }
}
```

**Language codes used**:
- `de` = German
- `en` = English
- `es` = Spanish
- `fr` = French
- `ja` = Japanese
- `zh` = Chinese

### 2. Updated `LanguageDetector` Class

**Location**: `/home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router/router1.py`

**Changes**:

#### Before:
```python
class LanguageDetector:
    def __init__(self):
        self.language_mapping = {
            '__label__de': 'german',
            '__label__en': 'english',
            '__label__es': 'spanish',
            '__label__fr': 'french',
            '__label__ja': 'japanese',
            '__label__zh': 'chinese'
        }
```

#### After:
```python
class LanguageDetector:
    def __init__(self, registry_path: str | Path = None):
        # Default language mapping (FastText label -> full name)
        self._default_language_mapping = {
            '__label__de': 'german',
            '__label__en': 'english',
            '__label__es': 'spanish',
            '__label__fr': 'french',
            '__label__ja': 'japanese',
            '__label__zh': 'chinese'
        }

        # Load supported languages from registry if provided
        self.registry_path = registry_path
        self.supported_languages_by_task = {}
        self.all_supported_languages = set()

        if registry_path:
            self._load_languages_from_registry()
```

**New Methods Added**:

1. **`_load_languages_from_registry()`**
   - Reads the experts registry JSON file
   - Extracts `supported_languages` from each task
   - Builds a mapping of task → supported languages
   - Collects all unique languages across all tasks

2. **`_code_to_full_name(code: str) -> str`**
   - Converts short language codes to full names
   - Examples: `'en'` → `'english'`, `'ja'` → `'japanese'`

3. **`get_supported_languages_for_task(domain: str, task: str) -> list[str]`**
   - Returns list of supported languages for a specific task
   - Task key format: `"domain/task"` (e.g., `"finance/rating"`)
   - Falls back to all supported languages if task not found

### 3. Updated `PromptRoutingSystem`

**Location**: Same file as `LanguageDetector`

**Changes**:

#### Before:
```python
class PromptRoutingSystem:
    def __init__(self):
        self.language_detector = LanguageDetector()
```

#### After:
```python
class PromptRoutingSystem:
    def __init__(self):
        config_path = Path(__file__).parents[4] / "experts" / "config"
        self.expert_registry_path = config_path / "experts_registry.json"

        # Initialize language detector with registry path
        self.language_detector = LanguageDetector(registry_path=self.expert_registry_path)
```

### 4. Enhanced `get_system_stats()` Method

**Changes**:

#### Before:
```python
def get_system_stats(self):
    supported_languages = len(self.language_detector.language_mapping)
    return {
        'supported_languages': supported_languages,
        ...
    }
```

#### After:
```python
def get_system_stats(self):
    all_languages = self.language_detector.all_supported_languages
    return {
        'supported_languages': len(all_languages),
        'all_languages': sorted(all_languages),
        'languages_by_task': self.language_detector.supported_languages_by_task,
        ...
    }
```

Now shows:
- Total number of supported languages
- List of all supported languages
- Languages supported by each task

---

## Benefits

### 1. **Task-Specific Language Support**
- Each task can now declare which languages it supports
- Example: `general/text_summarization` only supports English
- Example: `finance/rating` supports 6 languages

### 2. **Better Maintainability**
- Languages are configured in one place (`experts_registry.json`)
- No need to modify code to add/remove language support
- Easy to see which languages each task supports

### 3. **Runtime Validation**
- System can check if a language is supported for a specific task
- Can warn users if they try to use unsupported language/task combinations

### 4. **Better Logging**
- System initialization now prints:
  ```
  ✅ Loaded language support from registry:
     finance/rating: ['de', 'en', 'es', 'fr', 'ja', 'zh']
     finance/pii: ['de', 'en', 'es', 'fr', 'ja', 'zh']
     finance/news: ['de', 'en', 'es', 'fr', 'ja', 'zh']
     general/text_summarization: ['en']
     All supported languages: ['chinese', 'english', 'french', 'german', 'japanese', 'spanish']
  ```

### 5. **Backward Compatibility**
- If no registry path is provided, falls back to default mapping
- Existing code continues to work without modification

---

## Usage Examples

### Example 1: Check Supported Languages for a Task

```python
system = PromptRoutingSystem()

# Get languages supported by rating task
rating_langs = system.language_detector.get_supported_languages_for_task("finance", "rating")
print(rating_langs)  # Output: ['de', 'en', 'es', 'fr', 'ja', 'zh']

# Get languages supported by text summarization
summary_langs = system.language_detector.get_supported_languages_for_task("general", "text_summarization")
print(summary_langs)  # Output: ['en']
```

### Example 2: Get System Statistics

```python
stats = system.get_system_stats()
print(f"Total supported languages: {stats['supported_languages']}")
print(f"All languages: {stats['all_languages']}")
print(f"Languages by task: {stats['languages_by_task']}")
```

Output:
```
Total supported languages: 6
All languages: ['chinese', 'english', 'french', 'german', 'japanese', 'spanish']
Languages by task: {
  'finance/rating': ['de', 'en', 'es', 'fr', 'ja', 'zh'],
  'finance/pii': ['de', 'en', 'es', 'fr', 'ja', 'zh'],
  'finance/news': ['de', 'en', 'es', 'fr', 'ja', 'zh'],
  'general/text_summarization': ['en']
}
```

---

## Adding Support for New Languages

To add a new language:

### Step 1: Add to `experts_registry.json`

```json
{
  "tasks": {
    "finance/rating": {
      "supported_languages": ["de", "en", "es", "fr", "ja", "zh", "ko"],  // Added 'ko' for Korean
      ...
    }
  }
}
```

### Step 2: Add Language Code Mapping (if needed)

If using a new language code, add it to `_code_to_full_name()` in `LanguageDetector`:

```python
def _code_to_full_name(self, code: str) -> str:
    code_mapping = {
        'de': 'german',
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'ja': 'japanese',
        'zh': 'chinese',
        'ko': 'korean'  # Add new language
    }
    return code_mapping.get(code.lower(), code)
```

### Step 3: Add FastText Label Mapping (if needed)

If FastText uses a different label, add it to `_default_language_mapping`:

```python
self._default_language_mapping = {
    '__label__de': 'german',
    '__label__en': 'english',
    '__label__es': 'spanish',
    '__label__fr': 'french',
    '__label__ja': 'japanese',
    '__label__zh': 'chinese',
    '__label__ko': 'korean'  # Add new FastText label
}
```

### Step 4: Add Language-Specific Model Configuration

Update the task's `language_mapping` section:

```json
{
  "tasks": {
    "finance/rating": {
      "language_mapping": {
        "korean": {
          "base_model_key": "aya-23",
          "adapter_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/aya-23/",
          "template_path": "src/models/experts/llms/adapters/finance/sentiment_analysis/aya-23/template.json"
        }
      }
    }
  }
}
```

That's it! The system will automatically load the new language configuration on next initialization.

---

## Migration Notes

### For Existing Code

No changes needed! The `LanguageDetector` is backward compatible:
- If no registry path is provided, uses default language mapping
- Existing code that doesn't pass `registry_path` continues to work

### For New Code

Recommended to pass registry path:
```python
detector = LanguageDetector(registry_path="path/to/experts_registry.json")
```

This enables:
- Task-specific language support
- Dynamic language loading
- Better error messages

---

## Testing

### Verify Configuration

Run the router with test data:
```bash
cd /home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router/
python router1.py
```

Expected output should include:
```
✅ Loaded language support from registry:
   finance/rating: ['de', 'en', 'es', 'fr', 'ja', 'zh']
   finance/pii: ['de', 'en', 'es', 'fr', 'ja', 'zh']
   finance/news: ['de', 'en', 'es', 'fr', 'ja', 'zh']
   general/text_summarization: ['en']
   All supported languages: ['chinese', 'english', 'french', 'german', 'japanese', 'spanish']
```

---

## Files Modified

1. `/home/cse/Desktop/multilingual-llm-moe/src/models/experts/config/experts_registry.json`
   - Added `supported_languages` to all tasks

2. `/home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router/router1.py`
   - Updated `LanguageDetector` class
   - Updated `PromptRoutingSystem` class
   - Enhanced `get_system_stats()` method

---

## Future Enhancements

1. **Language Validation**
   - Add method to validate if language is supported before routing
   - Throw warning if unsupported language is detected

2. **Language Fallback**
   - If detected language not supported, fall back to English
   - Log fallback events for monitoring

3. **Dynamic Language Addition**
   - Support hot-reloading of registry without restart
   - Add API endpoint to query supported languages per task

4. **Language Coverage Metrics**
   - Track which languages are most/least used
   - Identify tasks that need more language support

---

## Summary

✅ **Completed**:
- Moved language configuration from code to JSON registry
- Made language support task-specific
- Added backward compatibility
- Enhanced system statistics with language information
- Created helper methods for language querying

✅ **Benefits**:
- More flexible and maintainable
- Easy to add new languages
- Task-specific language support
- Better visibility into language coverage

✅ **No Breaking Changes**:
- Existing code continues to work
- Backward compatible fallback to default mapping
