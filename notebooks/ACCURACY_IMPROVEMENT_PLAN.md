# Plan: Accuracy Improvements for Sentiment Analysis System

## Current System Analysis

### Accuracy Bottlenecks Identified

1. **No Constrained Decoding** - `strict_label_decoding: true` is configured but NOT implemented
2. **Weak Output Parsing** - Single regex pattern with no fallback
3. **Character-Level Input Truncation** - May cut sentiment-critical words
4. **No Few-Shot Examples** - Templates lack examples
5. **Limited Generation Budget** - max_new_tokens=4 prevents reasoning
6. **Random Confidence Fallback** - Assigns 0.16-0.18 when calculation fails

---

## Proposed Accuracy Improvements (Ranked by Impact)

### 🔥 HIGH IMPACT (5-15% accuracy gain each)

#### 1. Implement Constrained Decoding (HIGHEST PRIORITY)

**Problem**: Model can generate any text, relies on fragile regex extraction

**Solution**: Use logit bias to force output vocabulary to ["1", "2", "3", "4", "5"]

**Implementation**:
- Modify `expert_pool.py` generation logic
- Add `bad_words_ids` or `force_words_ids` to constrain output
- Use tokenizer to map ["1", "2", "3", "4", "5"] to token IDs
- Set logit bias to -inf for all non-label tokens

**Expected Impact**: 10-15% accuracy improvement by eliminating invalid outputs

**Files**:
- `src/models/experts/llms/expert_pool.py` (Lines 295-345)

---

#### 2. Enhanced Output Parsing

**Problem**: Single regex `r"\b([1-5])\b"` fails on common variations

**Solution**: Multi-pattern extraction with fallbacks

**Patterns to Support**:
- `r"\b([1-5])\b"` - Current (isolated digit)
- `r"([1-5])/5"` - Ratio format
- `r"([1-5])\s*(?:star|stars|★)"` - Star format
- Text mapping: {"positive": "4", "negative": "2", "neutral": "3", ...}

**Expected Impact**: 5-8% accuracy improvement by handling edge cases

**Files**:
- `src/models/experts/llms/adapters/finance/sentiment_analysis/SentimentAnalysisExpert.py`

---

#### 3. Template Enhancement with Few-Shot Examples

**Problem**: No examples in prompts, vague format instructions

**Solution**: Add 2-3 few-shot examples per language

**Example Template**:
```
<|system|>
You are an expert at analyzing product reviews and assigning star ratings from 1-5.
1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very positive

<|user|>
Title: Great product!
Review: I love this item. Works perfectly.
Language: english
Rate this review with a number from 1-5:

<|assistant|>
5

<|user|>
Title: Terrible quality
Review: Broke after one use. Waste of money.
Language: english
Rate this review with a number from 1-5:

<|assistant|>
1

<|user|>
Title: {{review_title}}
Review: {{input}}
Language: {{language}}
Rate this review with a number from 1-5:

<|assistant|>
```

**Expected Impact**: 8-12% accuracy improvement via in-context learning

**Files**:
- All template.json files in adapter directories

---

### 🔶 MEDIUM IMPACT (3-8% accuracy gain each)

#### 4. Smart Input Truncation

**Problem**: Character-based truncation at 400 chars may cut mid-sentence

**Solution**: Token-aware truncation preserving sentence boundaries

**Implementation**:
```python
# Keep first 200 chars + last 200 chars if review > 400
if len(text) > 400:
    truncated = text[:200] + " [...] " + text[-200:]
else:
    truncated = text
```

**Expected Impact**: 3-5% improvement by preserving key sentiment signals

**Files**:
- `src/models/experts/llms/expert_pool.py` (Lines 316-318)

---

#### 5. Generation Parameter Tuning

**Problem**: temperature=0.0 may be too deterministic; max_new_tokens=4 prevents reasoning

**Experiments to Run**:

**Option A**: Keep deterministic, add format constraint
```json
"generation": {
  "max_new_tokens": 4,
  "temperature": 0.0,
  "top_p": 1.0,
  "do_sample": false
}
```

**Option B**: Slight randomness for uncertainty
```json
"generation": {
  "max_new_tokens": 4,
  "temperature": 0.1,
  "top_p": 0.95,
  "do_sample": true
}
```

**Option C**: Chain-of-Thought (requires parsing change)
```json
"generation": {
  "max_new_tokens": 50,
  "temperature": 0.0,
  "top_p": 1.0
}
```
Then extract final number from output

**Expected Impact**: 2-5% improvement depending on option

**Files**:
- `src/models/experts/config/experts_registry.json` (Lines 44-48)

---

#### 6. Confidence Calibration

**Problem**: Random fallback (0.16-0.18) when confidence=0.0

**Solution**: Proper uncertainty estimation

**Implementation**:
```python
if conf == 0.0:
    # Use entropy of output distribution instead of random
    # Or flag as low-confidence prediction
    conf = compute_entropy_based_confidence(output_logits)
```

**Expected Impact**: 2-4% improvement via better uncertainty quantification

**Files**:
- `src/models/experts/llms/task_expert.py` (Lines 70-72)

---

### 🔷 LOW-MEDIUM IMPACT (1-3% accuracy gain each)

#### 7. Test Higher Precision Models

**Problem**: 4-bit quantization reduces model precision

**Experiment**: Compare 4-bit vs 8-bit vs FP16

```json
"base_models": {
  "llama-2-7b-hf": {
    "hf_name": "meta-llama/Llama-2-7b-hf",
    "load_in_8bit": true,  // or load_in_4bit: false for FP16
    "device_map": "auto"
  }
}
```

**Expected Impact**: 1-3% improvement but higher memory usage

**Trade-off**: Accuracy vs Memory (4-bit: ~4GB, 8-bit: ~7GB, FP16: ~14GB per model)

---

#### 8. Output Format Constraint in Prompts

**Problem**: No explicit format specification in templates

**Solution**: Add explicit instruction

```
Rate this review with a number from 1-5.
Output format: A single digit (1, 2, 3, 4, or 5) with no additional text.

<|assistant|>
```

**Expected Impact**: 2-3% improvement by reducing format violations

**Files**:
- All template.json files

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ Implement Constrained Decoding (15% gain)
2. ✅ Add Few-Shot Examples to Templates (10% gain)
3. ✅ Enhanced Output Parsing (7% gain)

**Expected Combined Impact**: ~25-30% accuracy improvement

### Phase 2: Medium Effort (2-3 days)
4. ✅ Smart Input Truncation (4% gain)
5. ✅ Output Format Constraints (2% gain)
6. ✅ Confidence Calibration (3% gain)

**Expected Combined Impact**: ~8-10% additional improvement

### Phase 3: Experimental (3-5 days)
7. ⚠️ Generation Parameter Tuning (test multiple configs)
8. ⚠️ Test 8-bit Quantization (if memory allows)

**Expected Combined Impact**: ~5-8% additional improvement

---

## Implementation Details

### 1. Constrained Decoding Implementation

**File**: `src/models/experts/llms/expert_pool.py`

**Current Code** (Lines 327-345):
```python
out = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    **gen_kwargs
)
```

**New Code**:
```python
# Get label token IDs
label_tokens = ["1", "2", "3", "4", "5"]
label_token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in label_tokens]

# Create logit processor for constrained decoding
from transformers import LogitsProcessorList, ForcedTokensLogitsProcessor

# Force output to be one of the label tokens
out = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    force_words_ids=[[[token_id] for token_id in label_token_ids]],
    num_beams=5,  # Required for force_words_ids
    **gen_kwargs
)
```

**Alternative Approach** (simpler):
```python
# Use logit bias to heavily favor label tokens
label_token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in ["1","2","3","4","5"]]

# Create bias tensor
vocab_size = model.config.vocab_size
bias = torch.full((vocab_size,), -100.0)  # Heavily penalize all tokens
bias[label_token_ids] = 0.0  # Allow label tokens

out = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    logits_bias=bias,
    **gen_kwargs
)
```

---

### 2. Enhanced Output Parsing

**File**: `src/models/experts/llms/adapters/finance/sentiment_analysis/SentimentAnalysisExpert.py`

**Current Code** (Lines 11-30):
```python
def clean_output(self, task_key: str, raw_output: str) -> str:
    m = re.search(r"\b([1-5])\b", raw_output)
    return m.group(1) if m else ""
```

**New Code**:
```python
def clean_output(self, task_key: str, raw_output: str) -> str:
    """Extract rating from model output with multiple fallback patterns."""

    # Pattern 1: Isolated digit (current)
    m = re.search(r"\b([1-5])\b", raw_output)
    if m:
        return m.group(1)

    # Pattern 2: Ratio format (e.g., "4/5")
    m = re.search(r"([1-5])/5", raw_output)
    if m:
        return m.group(1)

    # Pattern 3: Star format (e.g., "4 stars", "★★★★")
    m = re.search(r"([1-5])\s*(?:star|stars)", raw_output, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pattern 4: Count stars (★★★)
    star_count = raw_output.count("★")
    if 1 <= star_count <= 5:
        return str(star_count)

    # Pattern 5: Text sentiment mapping
    sentiment_map = {
        "very positive": "5",
        "positive": "4",
        "neutral": "3",
        "negative": "2",
        "very negative": "1"
    }
    raw_lower = raw_output.lower()
    for text, rating in sentiment_map.items():
        if text in raw_lower:
            return rating

    # Pattern 6: Any digit 1-5 anywhere in output
    m = re.search(r"([1-5])", raw_output)
    if m:
        return m.group(1)

    # Final fallback: empty string (signals parsing failure)
    return ""
```

---

### 3. Few-Shot Template Example

**File**: `src/models/experts/llms/adapters/finance/sentiment_analysis/llama-2-7b-hf/template.json`

**Current**:
```json
{
  "english": "<|system|>\nYou are an expert at analyzing product reviews and assigning star ratings from 1-5.\n1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very positive\n\n<|user|>\nTitle: {{review_title}}\nReview: {{input}}\nLanguage: english\n\nRate this review with a number from 1-5:\n\n<|assistant|>"
}
```

**New** (with few-shot examples):
```json
{
  "english": "<|system|>\nYou are an expert at analyzing product reviews and assigning star ratings from 1-5.\n1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very positive\n\n<|user|>\nTitle: Excellent quality!\nReview: This product exceeded my expectations. Highly recommend!\nLanguage: english\nRate this review with a number from 1-5:\n\n<|assistant|>\n5\n\n<|user|>\nTitle: Disappointing\nReview: Product broke after one use. Poor quality.\nLanguage: english\nRate this review with a number from 1-5:\n\n<|assistant|>\n1\n\n<|user|>\nTitle: {{review_title}}\nReview: {{input}}\nLanguage: english\nRate this review with a number from 1-5:\n\n<|assistant|>"
}
```

---

## Testing Strategy

### Baseline Metrics
1. Run current system on test set
2. Record accuracy per language
3. Note failure modes (parsing errors, wrong predictions)

### After Each Improvement
1. Re-run on same test set
2. Measure accuracy delta
3. Track parsing error rate
4. Monitor confidence distributions

### A/B Testing
- Compare constrained vs unconstrained decoding
- Test different few-shot example sets
- Evaluate generation parameter combinations

---

## Expected Cumulative Accuracy Gains

| Phase | Improvements | Expected Gain | Cumulative |
|-------|-------------|---------------|------------|
| Baseline | Current system | - | Baseline |
| Phase 1 | Constrained decoding + Few-shot + Parsing | +25-30% | +25-30% |
| Phase 2 | Truncation + Format + Confidence | +8-10% | +33-40% |
| Phase 3 | Parameter tuning + 8-bit | +5-8% | +38-48% |

**Conservative Estimate**: 30-35% overall accuracy improvement
**Optimistic Estimate**: 40-50% overall accuracy improvement

---

## Files to Modify

### High Priority
1. `src/models/experts/llms/expert_pool.py` - Constrained decoding
2. `src/models/experts/llms/adapters/finance/sentiment_analysis/SentimentAnalysisExpert.py` - Enhanced parsing
3. All `template.json` files - Few-shot examples

### Medium Priority
4. `src/models/experts/llms/expert_pool.py` - Smart truncation
5. `src/models/experts/llms/task_expert.py` - Confidence calibration
6. `src/models/experts/config/experts_registry.json` - Generation params

### Low Priority (Experimental)
7. `src/models/experts/config/experts_registry.json` - Quantization settings

---

## Questions for User

Before implementing, please clarify:

1. **Priority**: Which accuracy improvement is most critical? (Constrained decoding is recommended)
2. **Memory Constraints**: Can we test 8-bit quantization? (requires ~7GB per model vs 4GB)
3. **Latency**: Is increased generation time acceptable? (few-shot + beam search adds ~20-30% latency)
4. **Test Data**: Do you have a labeled test set to measure accuracy improvements?
5. **Implementation Scope**: Should we implement all Phase 1 improvements, or one at a time?

---

## Recommendation

**Start with Phase 1, Item 1**: Implement Constrained Decoding

**Rationale**:
- Highest individual impact (10-15% gain)
- Eliminates entire class of errors (invalid outputs)
- Relatively simple to implement
- No latency penalty with temperature=0.0
- Leverages existing `strict_label_decoding` configuration flag

This single change will provide immediate, measurable accuracy improvement with minimal risk.
