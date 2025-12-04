# Multilingual LLM Mixture-of-Experts Router

This repository contains research code for building a multilingual mixture-of-experts (MoE) routing
stack that can steer large language models toward the most appropriate expert without translating the
incoming text.  The current focus is a reinforcement-learning based router (`router1.py`) that
performs four sequential routing decisions – language detection, domain classification, task
selection, and final expert dispatch – to deliver high-quality responses for diverse languages and
domains.

## Repository layout

- `src/` – model implementations, including routing, experts, and utility modules.
  - `models/gating/without-translation/rl-based/qlearning-router/router1.py` – end-to-end routing
    pipeline that combines FastText language identification, transformer-based domain classification,
    a Q-learning task router, and expert selection.
  - `models/experts` – expert pool definitions and utilities for loading specialist LLM adapters.
  - `training/` – scripts and helpers for training the individual components.
- `data/` – canonical data sets for supervised training and evaluation (not tracked in VCS).
- `experiments/` – experiment configurations and logs.
- `tests/` – unit and integration tests.
- `notebooks/` – analysis notebooks illustrating experiments and ablations.
- `results/` – generated reports/metrics from runs.

## Routing pipeline (router1.py)

`router1.py` stitches together the complete MoE gating pipeline. Each stage is designed to operate on
native text, preserving language-specific signals throughout the routing process.

### 1. Language detection

- Uses Facebook's FastText `lid.176` model to predict the most likely language from raw text.
- Falls back to rule-based heuristics when the FastText model is unavailable or input is too short.
- Maps FastText labels to canonical language names (e.g., `__label__de → german`).

### 2. Domain classification

- Employs a multilingual transformer (`xlm-roberta-base`) with a lightweight classification head.
- Supports fine-tuning on prompt/domain pairs via `fit_from_labeled_prompts` with optional encoder
  freezing and class weighting.
- Maintains per-domain prototype embeddings to smooth predictions through prototype ensembling at
  inference time.
- Persists models to disk (`domain_cls.pt`) with version-aware checkpoints.

### 3. Task selection via Q-learning

- Trains per-domain Q-networks that map transformer embeddings to task actions.
- Uses an epsilon-greedy exploration strategy with experience buffers and optional double Q-learning
  target updates.
- Supports offline replay, validation-driven checkpointing, and persistence of both encoder and
  router weights (`task_routers_qlearning`).

### 4. Expert selection and execution

- Loads expert metadata from `experts/config/` via `ModelLoader` and `DomainTaskLoader`.
- Instantiates task experts through `LLMAdapterPool` and executes the predicted expert to produce the
  final classification or generation result.
- Returns a detailed routing trace containing the detected language, domain probabilities, chosen
  task, expert confidence, and the final output.

### Additional utilities

- Aggregated training APIs (`train_domain_classifier`, `train_q_routers`) for supervised fine-tuning.
- Evaluation harness that reports confusion matrices, per-language breakdowns, and metrics such as
  macro/micro F1 and Cohen's kappa.
- Model persistence helpers (`save_all_models`, `load_models`) for resuming experiments.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**

   - Populate `experts/config/` with `model_config.json`, `domain_tasks.json`, and
     `experts_registry.json` describing available experts and task mappings.
   - Provide labeled training data (CSV/JSON) with `prompt`, `classification_text`, `domain`, `task`,
     and `label` fields for supervised training.

3. **Train components**

   ```python
   from src.models.gating.without_translation.rl_based.qlearning_router.router1 import PromptRoutingSystem

   system = PromptRoutingSystem()
   system.train_domain_classifier(training_data, epochs=3, batch_size=32, lr=2e-5, freeze_encoder=True)
   system.train_q_routers(training_data)
   system.save_all_models()
   ```

4. **Route prompts**

   ```python
   result = system.route_prompt(prompt=text, classification_text=text)
   print(result["routing_path"], result["result"], result["expert_confidence"])
   ```

5. **Evaluate**

   - Use the script section in `router1.py` (`python router1.py`) with appropriate JSON/CSV inputs to
     reproduce evaluation metrics and confusion matrices.

### Adapter & dataset synchronization

The repository does not store QLoRA adapters or large datasets directly. Use
the helper scripts to fetch or publish artifacts based on configuration files
inside `config/`:

```bash
# Download adapters defined in config/adapter_storage.json
bash scripts/download_adapters.sh

# Upload local adapters using the same config
bash scripts/upload_adapters.sh

# Override the config path if needed
bash scripts/download_adapters.sh path/to/custom_config.json

# Download datasets into the ./data folder
bash scripts/download_datasets.sh config/dataset_storage.json

# Upload datasets back to remote storage
bash scripts/upload_datasets.sh config/dataset_storage.json
```

## Roadmap

- Expand language mappings to additional locales and improve fallback heuristics.
- Incorporate reinforcement learning signals from expert quality scores.
- Add experiment tracking integrations for reproducibility.
- Extend tests to cover training reloads and expert selection edge cases.

## Contributing

Please open an issue or submit a pull request if you would like to improve the routing pipeline,
expand expert coverage, or add new evaluation datasets.

## License

This project is released under the Apache 2.0 License. See `LICENSE` for details.
