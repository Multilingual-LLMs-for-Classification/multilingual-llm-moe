# New Sections for Progress Evaluation

---

## 3.4.1 Static (Methodology — Stage 3: Expert Selection)

In the static routing approach, expert selection is performed using a fixed, pre-defined mapping rather than a learned or adaptive policy. Once the language and task have been identified by the preceding routing stages, the system consults a configuration-driven lookup table to deterministically select the appropriate expert for the given input.

The core idea behind the static strategy is that, for each (task, language) pair, a single best-performing expert can be identified through offline evaluation and then hard-coded into the system configuration. This avoids the overhead and instability associated with online learning methods such as bandit-based or reinforcement-learning-based routers, and provides a reliable, interpretable baseline for expert selection.

**Design Decisions.** The static routing strategy was motivated by several key considerations:

1. **Deterministic and reproducible routing.** Given the same input characteristics (detected language and identified task), the static router always selects the same expert. This eliminates variance introduced by exploration policies (e.g., epsilon-greedy) and ensures that evaluation results are fully reproducible across runs.

2. **Offline expert benchmarking as the selection criterion.** The mapping from (task, language) to expert is established by independently evaluating all candidate models on each task–language combination during a prior benchmarking phase. The model achieving the highest task-specific metric (e.g., macro F1 for classification, token-level accuracy for extraction) is selected as the designated expert for that combination. This decouples the model selection process from the routing pipeline itself.

3. **No bandit or RL components.** Unlike the dynamic routing variant, the static approach does not employ multi-armed bandits, Q-learning, or any form of online reward-based optimization. The routing decision is purely a table lookup, making it computationally inexpensive at inference time and straightforward to debug and maintain.

4. **Configuration-driven architecture.** The entire mapping is stored in a centralized JSON registry (`experts_registry.json`), which specifies, for each task, the default base model key, per-language overrides (including the base model, adapter path, and prompt template), and the supported language set. This design allows the mapping to be updated without modifying any code — only the configuration file needs to change when a better expert is identified or a new language is added.

5. **Language-group-aware model assignment.** Rather than assigning experts at the individual language level in an ad-hoc manner, the static mapping reflects language-group-level decisions informed by the benchmarking results. For example, if a single model consistently outperforms alternatives across all European languages for a given task, it is assigned to the entire group, with per-language overrides applied only where a different model provides a measurable advantage. This reduces configuration complexity while preserving performance.

Overall, the static expert selection strategy serves as a strong, interpretable baseline within the three-stage routing framework. It demonstrates that a well-chosen fixed mapping, grounded in systematic offline evaluation, can achieve competitive classification performance without requiring any adaptive routing mechanism. The dynamic routing variants (Section 3.4.2) build upon this baseline by introducing learned selection policies that can adapt to distributional shifts or exploit fine-grained input features beyond language and task identity.

---

## 3.5 Expert Design (Methodology — Registry, Language-to-Expert Mapping, Adapter Pool)

The expert layer of the proposed Mixture of Experts framework is built around three interconnected components: a centralized expert registry, a language-to-expert mapping scheme, and a shared adapter pool. Together, these components enable efficient, modular management of multiple specialized LLM experts while minimizing memory consumption and configuration overhead.

### Expert Registry

All expert metadata is maintained in a single JSON registry file (`experts_registry.json`). For each task (e.g., `finance/rating`, `finance/pii`, `finance/news`, `finance/esci`), the registry stores:

- **Supported languages**: The set of language codes for which the task has been evaluated and for which adapters exist (e.g., `[de, en, es, fr, ja, zh]` for sentiment rating).
- **Label set**: The valid output labels for the task (e.g., `[1, 2, 3, 4, 5]` for star ratings; `[E, S, C, I]` for ESCI relevance).
- **Default base model key**: The base LLM to use when no language-specific override is provided.
- **Language mapping**: A per-language dictionary specifying the base model key, adapter name, and adapter path for each supported language. This allows different languages within the same task to be served by different base models if benchmarking indicates a performance advantage.
- **Generation configuration**: Task-specific inference parameters such as `max_new_tokens`, `temperature`, and `top_p`, which override the global defaults.
- **Expert class**: The Python class implementing the task-specific prediction logic (e.g., `SentimentAnalysisExpert`, `PIIExpert`, `NewsClassificationExpert`, `ESCIExpert`).

This centralized registry design means that adding a new language, swapping an adapter, or changing the assigned base model for a task requires only a configuration change — no code modifications are needed.

### Language-to-Expert Mapping

The mapping from (task, language) to a specific expert is resolved at runtime through a two-level lookup:

1. **Task-level default.** Each task specifies a `base_model_key` that serves as the default model for all languages.
2. **Language-level override.** The `language_mapping` section within each task can override the default for specific languages, pointing to a different base model and a language-specific adapter.

This two-level scheme balances simplicity with flexibility. In practice, the mapping reflects the offline benchmarking results: for each task, every candidate model was evaluated across all supported languages, and the best-performing model for each (task, language) pair was recorded in the registry. The resulting assignments are as follows:

**Sentiment Rating (finance/rating)**:
- English → LLaMA-2-7B; German → DeepSeek-7B-Chat; Spanish → Aya-23-8B; French → DeepSeek-7B-Chat; Japanese → LLaMA-2-7B; Chinese → BloomZ-7B1.

**PII Extraction (finance/pii)**:
- All seven languages (Dutch, English, French, German, Italian, Spanish, Swedish) → Mistral-7B-Instruct-v0.3, with language-specific adapters (and Aya-23-8B adapters used for Dutch and Italian where Mistral showed slightly different adapter requirements).

**Financial News Classification (finance/news)**:
- English, Danish, Spanish, Polish → Mistral-7B-Instruct-v0.3 (unsloth 4-bit variant); Turkish → XGLM-7.5B.

**ESCI Relevance (finance/esci)**:
- English, Japanese → LLaMA-3.1-8B-Instruct; Spanish → Mistral-7B-Instruct-v0.3.

These assignments highlight that no single model dominates across all tasks and languages. Mistral-7B variants are particularly strong for structured extraction (PII) and news classification, while different models (LLaMA-2, DeepSeek, Aya-23, BloomZ) prove superior for specific language–task combinations in sentiment analysis.

### Adapter Pool

Rather than loading a separate full-precision model for each expert, the system uses a shared adapter pool (`LLMAdapterPool`) that manages base model instances and dynamically loads task- and language-specific QLoRA adapters on top of them.

The pool operates as follows:

1. **Base model sharing.** Each unique base model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) is loaded into GPU memory only once, in 4-bit quantized form. Multiple experts that share the same base model reference the same loaded instance.
2. **Adapter hot-swapping.** When an expert is invoked, the pool loads the corresponding QLoRA adapter weights (identified by the adapter path in the registry) on top of the base model. If the adapter is already loaded from a previous invocation, the swap is a no-op. If a different adapter is currently active, the pool unloads it and loads the requested one.
3. **Memory efficiency.** Since QLoRA adapters are small (typically 0.2–0.6% of the base model's parameters), the memory overhead of supporting many experts is minimal. The dominant cost is the base model itself, and sharing base models across experts amortizes this cost.

This adapter pool design is critical for practical deployment: it allows the system to support tens of (task, language)-specific experts while keeping GPU memory usage bounded by the number of unique base models rather than the number of experts.

### Expert Invocation

Each expert is implemented as a `TaskExpert` instance, initialized with a `TaskExpertConfig` that references the registry. At inference time, the expert:

1. Resolves the base model and adapter for the detected language using the registry.
2. Requests the adapter pool to load the appropriate adapter onto the base model.
3. Constructs the task-specific prompt using the designated template.
4. Runs generation with the task-specific inference parameters (temperature, max tokens, etc.).
5. Decodes and validates the output against the task's label set.

This modular design ensures that the routing pipeline and the expert layer remain loosely coupled: the router selects a (task, language) pair, and the expert layer handles all model loading, prompt construction, and output parsing independently.

---

## 5.2 Model Selection and Fine-tuning (Progress)

This section describes the process used to select base models and fine-tune task-specific adapters using QLoRA (Quantized Low-Rank Adaptation). The goal was to produce a set of specialized, memory-efficient experts — one per (task, language) combination — that could be served through the shared adapter pool described in Section 3.5.

### Candidate Models

A pool of multilingual and general-purpose LLMs in the 7–8B parameter range was evaluated as candidate base models. The candidates included:

- **Mistral-7B-Instruct-v0.3** (mistralai) — instruction-tuned, 32k context window
- **LLaMA-2-7B-hf** (Meta) — general-purpose, widely used baseline
- **LLaMA-3-8B-Instruct** and **LLaMA-3.1-8B-Instruct** (Meta) — improved instruction-following variants
- **Aya-23-8B** (Cohere) — multilingual instruction-tuned, trained on 23 languages
- **BloomZ-7B1** (BigScience) — multilingual, instruction-tuned via crosslingual finetuning
- **Gemma-7B** (Google) — general-purpose with strong English performance
- **XGLM-7.5B** (Meta/Facebook) — cross-lingual generative model
- **DeepSeek-7B-Chat** (DeepSeek) — chat-optimized multilingual model
- **Qwen-2.5** and **Qwen-3-8B** (Alibaba) — multilingual with strong CJK support

All candidate models were loaded in 4-bit quantized form to fit within a single-GPU setup (NVIDIA GPU with limited VRAM).

### QLoRA Fine-tuning Configuration

All adapters were trained using QLoRA, which combines 4-bit NormalFloat (NF4) quantization of the base model with low-rank adapter training. This approach reduces memory requirements by approximately 75% compared to full-precision fine-tuning while preserving competitive task performance.

**Quantization settings (universal across all tasks and models):**

| Parameter | Value |
|-----------|-------|
| Quantization precision | 4-bit (`load_in_4bit=True`) |
| Quantization type | NF4 (`bnb_4bit_quant_type="nf4"`) |
| Double quantization | Enabled (`bnb_4bit_use_double_quant=True`) |
| Compute dtype | float16 (`bnb_4bit_compute_dtype=torch.float16`) |

**LoRA adapter settings:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Alpha (α) | 32 | Fixed across all tasks |
| Dropout | 0.1 | Fixed across all tasks |
| Bias | None | No bias adaptation |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All linear projection layers in the attention and MLP blocks |
| Rank (r) | 8–64 | Varied by task complexity (see below) |

**Training hyperparameters (consistent across tasks):**

| Parameter | Value |
|-----------|-------|
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 (effective batch size = 8) |
| Warmup steps | 100 |
| Learning rate scheduler | Cosine decay |
| Optimizer | AdamW (`adamw_torch`) |
| Precision | FP16 |
| Evaluation frequency | Every 200 steps |
| Checkpoint frequency | Every 200 steps |

### Task-Specific Rank Selection and Training

The LoRA rank was the primary hyperparameter varied across tasks, reflecting differences in task complexity and output structure:

**Financial News Classification (MultiFin) — Rank 8:**
This is a straightforward 6-class single-label classification task over short news headlines. The low-dimensional adaptation (rank 8) proved sufficient, yielding the smallest adapter size (~20.9M trainable parameters, 0.29% of the base model). Training was run for 10 epochs with a learning rate of 1×10⁻⁴ and a maximum sequence length of 256 tokens. The best model (Mistral-7B, unsloth 4-bit variant) achieved a macro F1 of 0.8495 across five languages, with per-language accuracy ranging from 81.25% (Danish) to 92.50% (Spanish).

**PII Entity Extraction — Rank 16:**
PII extraction requires generating structured multi-entity outputs, making it more complex than simple classification. Rank 16 was used with a learning rate of 2×10⁻⁴, 3 training epochs, and a maximum sequence length of 2,048 tokens (with inference generation up to 4,096 tokens to accommodate long entity lists). The best model (Mistral-7B-Instruct-v0.3) achieved a token-level accuracy of 76.21% and macro F1 of 0.752 across seven languages. French showed the highest exact-match accuracy (31.03%), while Dutch was most challenging (11.72%).

**Sentiment Rating Prediction (MARC) — Rank 16 to 64:**
Star rating prediction on multilingual product reviews is a classification task with a 5-class ordinal label space. The standard configuration used rank 16, but experiments with Aya-23-8B used rank 64 to accommodate the model's larger parameter space and the need for stronger multilingual adaptation. Training used 3 epochs, a learning rate of 2×10⁻⁴, and a maximum sequence length of 512 tokens. The best-performing model (Gemma-7B at rank 16) achieved a macro F1 of 0.5628 across six languages. English accuracy reached 65.49%, while Chinese was lowest at 50.59%. Cross-language cosine similarity analysis revealed strong alignment for Romance languages (French: 0.9529) but weaker alignment for CJK scripts (Chinese: 0.6428).

**ESCI Query–Product Relevance — Rank 16:**
The 4-class relevance classification task operates over paired query–product inputs, requiring relational reasoning. Rank 16 was used with a learning rate of 1×10⁻⁴, 3 training epochs, and a maximum sequence length of 512 tokens to accommodate the concatenated query and product description. Mistral-7B-Instruct-v0.3 and LLaMA-3.1-8B-Instruct were the top performers across English, Spanish, and Japanese.

### Iterative Model Selection Process

For each task, the model selection followed a systematic benchmarking procedure:

1. **Fine-tune all candidate models** on the task's training split using the QLoRA configuration described above, with the rank and learning rate adjusted per task.
2. **Evaluate each fine-tuned model** on every supported language independently, recording task-specific metrics (accuracy, macro F1, token-level accuracy where applicable).
3. **Rank models per language** based on the primary metric (macro F1 for classification tasks, token-level accuracy for PII extraction).
4. **Select the best model for each (task, language) pair** and record the assignment in the experts registry.

This process revealed that no single model dominated across all tasks and languages:

- **Mistral-7B variants** excelled at structured extraction (PII) and short-text classification (news), winning across all supported languages for these tasks.
- **Gemma-7B** showed the strongest performance for ordinal sentiment classification (star ratings), particularly for European languages.
- **LLaMA-2-7B** and **DeepSeek-7B-Chat** provided the best results for specific language–task combinations in sentiment analysis (e.g., English and Japanese for LLaMA-2; German and French for DeepSeek).
- **Aya-23-8B** performed well for Spanish sentiment rating but underperformed significantly on news classification (F1: 0.3951), illustrating the importance of task-specific evaluation.
- **BloomZ-7B1** showed moderate multilingual capability but failed entirely on PII extraction (producing zero predictions), highlighting that model architecture and instruction-tuning approach matter as much as multilingual pretraining data.
- **XGLM-7.5B** provided the best results for Turkish news classification, a language where other candidates struggled.

These findings validate the MoE architecture's premise: by selecting the best expert per (task, language) combination rather than relying on a single generalist model, the system can exploit each model's specific strengths while mitigating their individual weaknesses.

### Adapter Storage and Deployment

Each fine-tuned adapter is stored under a standardized directory structure:

```
src/models/experts/llms/adapters/{domain}/{task}/{base_model_variant}/
```

For example:
- `finance/rating/sentiment_analysis/llama-2-7b-hf/` — English and Japanese sentiment adapter
- `finance/pii/Mistral-7B-Instruct-v0.3/` — Multilingual PII extraction adapter
- `finance/news_classification/mistral-7b/` — Multilingual news classification adapter
- `finance/esci/LLama-3-8.1B/` — English and Japanese ESCI adapter

Each adapter directory contains the LoRA weight files and an `adapter_config.json` that records the LoRA parameters used during training. At deployment time, the adapter pool reads these files to load and apply the adapters to the corresponding base models.

---

## 5.5.1 Static (Progress — Expert Selector Implementation)

The static expert selector is implemented as a configuration-driven lookup within the `PromptRoutingSystem` class. Unlike the dynamic routing variants, the static approach contains no trainable routing components for expert selection — the mapping from (task, language) to expert is fully determined by the entries in `experts_registry.json`.

### Implementation Overview

The static routing path in the `PromptRoutingSystem` proceeds as follows:

1. **Language detection.** The `LanguageDetector` module identifies the input language using the FastText model (`lid.176.bin`). The detected FastText label (e.g., `__label__en`) is mapped to a canonical language name (e.g., `english`) via a built-in mapping dictionary. If the FastText model is unavailable or the input is too short (fewer than 3 characters), a rule-based fallback using Unicode character ranges and keyword frequency scoring is applied.

2. **Domain classification.** The `DomainClassifier` (an XLM-R-based transformer with a linear head and prototype ensembling) classifies the input into a domain (e.g., `finance`). Since the current experimental setup operates within a single domain, this stage effectively acts as a pass-through but remains in the pipeline to support future multi-domain extensions.

3. **Task classification.** The `QLearningTaskClassifier` identifies the task within the detected domain using per-domain Q-routing networks. Each domain has a separate `QRouter` (a two-layer MLP) that takes the XLM-R CLS embedding as input and produces Q-values over the set of tasks in that domain. At inference time, the task with the highest Q-value is selected (greedy policy, no exploration).

4. **Expert lookup (static).** Once the language and task are known, the system resolves the expert by querying the registry. The `TaskExpert` instance for the identified (domain, task) pair is retrieved from a pre-initialized dictionary (`self.experts[domain][task]`). Internally, the expert uses the `LLMAdapterPool` to resolve the correct base model and adapter for the detected language, following the two-level lookup described in Section 3.5 (task-level default → language-level override).

5. **Expert invocation.** The selected expert constructs a task-specific prompt using the appropriate template, invokes the base model with the loaded QLoRA adapter, and returns the classification result along with a confidence score and the raw model output.

### Configuration-Driven Mapping

The static mapping is entirely encoded in `experts_registry.json`. The relevant resolution logic (mirrored by the `_get_expert_used` helper function used during evaluation) follows these priority levels:

1. **Direct per-language entry.** If the detected language appears as a direct key in the task's `language_mapping` and the entry does not contain a `languages` sub-list, the entry's `base_model_key` is used.
2. **Language group entry.** If the detected language is listed in the `languages` array of a language-group entry within the mapping, the group's `base_model_key` is used.
3. **Task-level default.** If the detected language does not appear in any mapping entry, the task's top-level `base_model_key` is used as a fallback.

This priority-based resolution ensures that the system always selects an expert, even for languages not explicitly listed in the registry, by falling back to the default model.

### Expert Initialization

At system startup, the `PromptRoutingSystem` constructor pre-initializes all experts:

```
for domain, tasks in self.domain_tasks.items():
    self.experts[domain] = {}
    for task in tasks.keys():
        self.experts[domain][task] = TaskExpert(
            TaskExpertConfig(domain=domain, task=task,
                             registry_path=str(self.expert_registry_path)),
            pool=self.expert_pool
        )
```

Each `TaskExpert` is backed by the shared `LLMAdapterPool`, which manages base model loading and adapter swapping. The pool loads each unique base model into GPU memory once (in 4-bit quantized form) and hot-swaps adapters as needed based on the incoming (task, language) pair.

### Evaluation Integration

During evaluation, the static routing path is exercised through `system.route_prompt(prompt, classification_text, review_title)`, which returns a dictionary containing the detected language, classified domain, classified task, the expert's prediction, and the full routing path (e.g., `english → finance → rating`). The evaluation harness additionally determines which base model was used via the `_get_expert_used()` function, enabling performance breakdowns by expert/model and by language group (e.g., European languages served by LLaMA-2 vs. Asian languages served by Aya-23).

### Key Implementation Characteristics

- **No online learning at the expert selection level.** The static selector does not update the mapping based on observed rewards or classification outcomes. The mapping is fixed at deployment time and remains constant throughout inference.
- **Low routing overhead.** The expert lookup is a dictionary access followed by a registry query — negligible compared to the cost of language detection, task classification, or LLM inference.
- **Reproducibility.** Given identical inputs and model weights, the static routing pipeline produces deterministic outputs, facilitating reliable evaluation and comparison with dynamic routing variants.
- **Extensibility.** Adding a new language or replacing an expert for a specific (task, language) pair requires only updating the registry JSON — no retraining of the routing pipeline is needed.
