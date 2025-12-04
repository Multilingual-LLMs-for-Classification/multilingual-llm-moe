# LLM Experts & Adapters

This module houses the task-specific mixture-of-experts (MoE) logic used by the
router. Each “expert” consists of:

1. A base foundation model managed by `LLMAdapterPool`.
2. A LoRA/QLoRA adapter stored under `src/models/experts/llms/adapters/<domain>/<task>/<adapter-name>`.
3. An adapter-aware cleaner that normalises the raw LLM response into the format
   expected by downstream evaluators.

The registry located at `src/models/experts/config/experts_registry.json`
connects these pieces. Entries are keyed by `domain/task` and define:

- The base model (`base_model_key`).
- The adapter (`adapter_name`, `adapter_path`).
- A prompt template (`template_path`).
- The cleaning logic (`expert_path`).

## Adapter-aware cleaning

Different LoRA adapters can emit subtly different text (extra prefixes,
structured answers, emoji, etc.). Clean-up functions therefore need to be
adapter-specific rather than task-wide. Both existing tasks follow this model:

- `finance/news` *(news classification)* – `NewsClassificationExpert` now chooses
  a cleaner based on the adapter name (`gemma-7b`, `llama-2-7b-hf`, …) and falls
  back to a default if no override exists.
- `finance/rating` *(sentiment analysis)* – `SentimentAnalysisExpert` applies
  adapter-specific post-processing (e.g., stripping `⭐` before parsing the rating).

The `TaskExpert` loader inspects the registry entry and passes the adapter name,
adapter path, and task metadata to the expert constructor. Experts can opt-in to
new parameters simply by listing them in their `__init__` signature.

### Adding a new adapter

1. **Copy the adapter files** into a new directory,
   `src/models/experts/llms/adapters/<domain>/<task>/<adapter-id>/`.
2. **Provide a prompt template** if your adapter requires a different prompt.
   Reference it via `template_path` in the registry entry.
3. **Update `experts_registry.json`**:
   ```json
   "<domain>/<task>": {
     "base_model_key": "llama-2-7b-hf",
     "adapter_name": "my-new-adapter",
     "adapter_path": "src/models/experts/llms/adapters/<domain>/<task>/my-new-adapter/",
     "expert_path": "src/models/experts/llms/adapters/<domain>/<task>/<TaskExpertClass>",
     "template_path": "src/models/experts/llms/adapters/<domain>/<task>/template.txt"
   }
   ```
4. **Implement adapter-specific cleaning** by updating the expert class. Add a
   cleaner entry keyed by `adapter_name` and provide a sensible default for new
   adapters.
5. **Test** by invoking `TaskExpert` for the task/adapter combination to ensure
   the cleaner runs without errors.

## Helper utilities

- `LLMAdapterPool` loads base models and attaches adapters on demand.
- `TaskExpert` orchestrates prompt generation, model invocation, and calls the
  adapter-specific cleaner.

When contributing new experts or adapters, keep responses minimal (single tokens
where possible) and add deterministic cleaners to avoid introducing variability
into downstream evaluations.
