# Sentiment Analysis Adapters - Per-Language Configuration

## Current Setup

### Language-to-Model Assignment

| Language | Base Model | Adapter Location |
|----------|------------|------------------|
| English | Llama-2-7b-hf | `llama-2-7b-hf/` ✓ |
| German | Deepseek-llm-7B-chat | `deepseek-llm-7b-chat/` ⚠️ |
| Spanish | aya-23 | `aya-23/` ⚠️ |
| French | Deepseek-llm-7B-chat | `deepseek-llm-7b-chat/` ⚠️ |
| Japanese | Llama-2-7b-hf | `llama-2-7b-hf/` ✓ |
| Chinese | Bloomz-7b1 | `bloomz-7b1/` ⚠️ |

**Legend**:
- ✓ = Adapter files complete
- ⚠️ = Adapter files needed

---

## Required Adapter Files

Each adapter directory should contain:

```
adapter_directory/
├── adapter_config.json          # LoRA configuration (rank, alpha, etc.)
├── adapter_model.safetensors    # Trained adapter weights (or .bin)
├── tokenizer.json               # Tokenizer vocabulary
├── tokenizer_config.json        # Tokenizer settings
├── special_tokens_map.json      # Special token mappings
└── template.json                # Language-specific prompt templates
```

---

## Example: Deepseek Adapter Structure

**Directory**: `deepseek-llm-7b-chat/`

**adapter_config.json** example:
```json
{
  "base_model_name_or_path": "deepseek-ai/deepseek-llm-7b-chat",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

**template.json**:
```json
{
  "german": "<YOUR_DEEPSEEK_PROMPT_FORMAT_FOR_GERMAN>",
  "french": "<YOUR_DEEPSEEK_PROMPT_FORMAT_FOR_FRENCH>"
}
```

---

## Example: Bloomz Adapter Structure

**Directory**: `bloomz-7b1/`

**adapter_config.json** example:
```json
{
  "base_model_name_or_path": "bigscience/bloomz-7b1",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "target_modules": [
    "query_key_value"
  ],
  "task_type": "CAUSAL_LM"
}
```

**template.json**:
```json
{
  "chinese": "<YOUR_BLOOMZ_PROMPT_FORMAT_FOR_CHINESE>"
}
```

---

## Aya-23 Template Update

**Directory**: `aya-23/`

**Current template.json**:
```json
{
  "spanish": "PLACEHOLDER: Add Aya-23 template for Spanish here"
}
```

**Update with your actual Spanish prompt template**

---

## Training Your Adapters

If you haven't trained these adapters yet, you'll need to:

1. **Fine-tune LoRA adapters** for each model on sentiment analysis task
2. **Save adapter weights** using PEFT library
3. **Copy tokenizer files** from base model
4. **Create template.json** with appropriate prompt format for each model

Example training script structure:
```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Create PEFT model
model = get_peft_model(base_model, peft_config)

# ... train on sentiment analysis data ...

# Save adapter
model.save_pretrained("./deepseek-llm-7b-chat/")
```

---

## Quick Start

1. **If adapters are ready**: Copy them into respective directories
2. **Update template.json files** with actual prompts
3. **Test**: Run router1.py with test samples in each language
4. **Verify**: Check that correct models are loaded for each language

The system will automatically handle model loading and routing!
