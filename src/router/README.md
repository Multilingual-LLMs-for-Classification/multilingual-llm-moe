PPO-Based LLM Routing Architecture
==================================

This router uses Proximal Policy Optimization (PPO) to pick the best Hugging Face LLM for multilingual Amazon reviews. The agent routes requests across:

* Qwen/Qwen2.5-3B
* TinyLlama/TinyLlama-1.1B-Chat-v1.0
* CohereLabs/aya-23-8B

Supported languages: English, French, Spanish, German, Chinese, Japanese. Rewards encourage accurate 1–5 star predictions, so the policy learns which LLM is strongest per language/input type.

System Architecture Overview
----------------------------

```
                +----------------------------+
                |       Review Input         |
                | (text + language + label)  |
                +--------------+-------------+
                               |
                               v
                    +---------------------+
                    |   State Builder     |
                    |  (RoutingPipeline)  |
                    +---------------------+
                               |
                               v
        +------------------------------------------------+
        |            PPO Routing Agent                   |
        |  (PPOPolicy + PPOAgent training logic)         |
        +------------------------------------------------+
            |                   |                   |
      (probs over LLMs)   (value estimate)     (log probs)
            |                   |                   |
            +-------------------+-------------------+
                               |
                        sampled action
                               |
                               v
         +-----------------------------------------------+
         |          HuggingFace LLM System               |
         |  - load chosen model (lazy)                   |
         |  - generate rating using language prompt      |
         +-----------------------------------------------+
                               |
                               v
                    +-----------------------+
                    |   Rating Extractor    |
                    +-----------------------+
                               |
                               v
                    +-----------------------+
                    |    Reward Function    |
                    |  reward = 1 - error/4 |
                    +-----------------------+
                               |
                               v
       +--------------------------------------------------------+
       |           PPO Update (training only)                   |
       |  uses: states, actions, log_probs, returns, advantages |
       +--------------------------------------------------------+
```

System-Level Architecture (Dataset Context)
-------------------------------------------

System-Level Diagram
--------------------

```
                ┌─────────────────────────────────────────┐
                │        Amazon Review Dataset            │
                │  (review_body, language, stars)         │
                └─────────────────────────────────────────┘
                                   │ batches
                                   ▼
                      ┌───────────────────────────┐
                      │    RoutingPipeline        │
                      │  (train_step / run_single)│
                      └───────────────────────────┘
                         │ state                 │ reward/logs
                         ▼                       ▲
               ┌────────────────────┐   actions  │
               │     PPOAgent       │────────────┘
               │ (select/computation│
               └────────────────────┘
                         │ chosen LLM id
                         ▼
            ┌───────────────────────────────┐
            │   HuggingFaceLLMSystem        │
            │  (prompts + quantized models) │
            └───────────────────────────────┘
                         │ rating + outputs
                         └─────────────┬─────────────┐
                                       │             │
                                  reward_fn     csv logging
```

* **RoutingPipeline**: Central hub that collects states, calls PPO for actions, invokes the Hugging Face LLM system, computes rewards, and triggers policy updates.
* **PPOAgent**: Learns the routing strategy, determining which LLM is best for a given state.
* **HuggingFaceLLMSystem**: Lazily loads each LLM, builds prompts per language, and extracts ratings.
* **Outputs**: `llm_outputs.csv` captures every interaction (review, language, action, rating, model, prompt).

Full Workflow Description (Training Phase)
------------------------------------------

1. **Input Stage** – Sample `(review_body, language, stars)` mini-batches from `data/Amazon/train_subset_final.csv`.
2. **State Construction** – `RoutingPipeline.get_state` builds `[random noise | one-hot language vector]`. Replace noise with embeddings later if desired.
3. **Routing Decision** – `PPOAgent.select_action` emits LLM probabilities, the sampled action, action log-probability, and critic value estimate.
4. **LLM Execution** – `HuggingFaceLLMSystem.run` lazily loads the selected LLM, injects the strict language-specific prompt, generates text, and extracts the numeric rating.
5. **Reward Calculation** – `reward = 1 - |pred - true| / 4`; perfect prediction yields 1, the worst mismatch returns 0.
6. **Transition Storage** – Save `(state, action, old_log_prob, value, reward, done=False)` plus CSV logging (`review, language, action, rating, prompt, model`).
7. **PPO Training Step** – After the batch, compute discounted returns, advantages, then call `PPOAgent.update` with the clipped PPO objective.
8. **Checkpointing** – After each training round, persist `ppo_router_policy_round{n}.pth`.

Router ↔ PPO Interaction Diagram
--------------------------------

```
               RoutingPipeline.train_step (per batch)
                           │
                           ▼
             ┌─────────────────────────────┐
             │ for review in batch:        │
             │   state = get_state(...)    │
             │   action, logp, value =     │
             │       PPOAgent.select_action│
             │   rating =                  │
             │       HuggingFaceLLMSystem.run|
             │   reward = reward_fn(...)   │
             │   log transition            │
             └─────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │ PPOAgent.compute_returns + PPOAgent.update     │
        │ • consumes states, actions, log_probs, rewards │
        │ • produces updated policy parameters           │
        └────────────────────────────────────────────────┘
```

This loop repeats for each mini-batch and ensures PPO always sees the latest transitions gathered by the router.

PPOAgent Functional Diagram
-----------------------------------

```
 ┌───────────────────────────────────────────────────────────┐
 │                        PPOAgent                           │
 └───────────────────────────────────────────────────────────┘
                 │
                 │ state
                 ▼
     ┌──────────────────────────┐
     │     select_action()      │
     └──────────────────────────┘
         │            │
         │            │
         │            ▼
         │     ┌──────────────────┐
         │     │  policy(state)   │
         │     └──────────────────┘
         │            │
         ▼            ▼
   action/log_prob   value
                 │
                 ▼
 ┌──────────────────────────────────────────────────────────────┐
 │         compute_returns(rewards, values, dones)              │
 └──────────────────────────────────────────────────────────────┘
                 │
                 ▼
      returns, advantages (A_t = R_t − V(s_t))
                 │
                 ▼
 ┌──────────────────────────────────────────────────────────────┐
 │      update(states, actions, old_log_probs,                  │
 │             returns, advantages)                             │
 └──────────────────────────────────────────────────────────────┘
                 │
                 ▼
        Improved routing policy over LLMs
```

Conceptual Flow Diagram (PPO Core)
-----------------------

```
RoutingPipeline.train_step
          │
          ▼
select_action(state)
  • Inputs: state vector (language + noise)
  • Outputs: action, log_prob, value
          │
          ▼
Environment
  • LLM chosen by action
  • Reward = 1 − |pred − true| / 4
          │
          ▼
compute_returns(rewards, values, dones)
  • Returns R_t (discounted rewards)
  • Advantages A_t = R_t − V(s_t)
          │
          ▼
update(states, actions, old_log_probs, returns, advantages)
  • Calculates PPO ratio and clipped objective
  • Adds critic loss
  • Optimizer step → policy improves
```

Detailed Function Explanations
------------------------------

### select_action(state)

* Builds a PyTorch tensor from the incoming NumPy state.
* Runs the actor–critic network to fetch action probabilities and the scalar value estimate.
* Samples an action via `Categorical(probs)` so exploration happens naturally.
* Returns `(action, log_prob, value)` which are stored for PPO’s later ratio calculation and critic training.

### compute_returns(rewards, values, dones)

* Walks through the rewards in reverse, computing the discounted return `R_t = r_t + γ R_{t+1}` with resets on episode boundaries.
* Concatenates the critic values, detaches them, and forms the advantage tensor `A_t = R_t − V(s_t)`.
* Provides both returns and advantages to the update stage; PPO relies on advantages for variance reduction.

### update(states, actions, old_log_probs, returns, advantages, epochs=5)

* Re-runs the policy to get new probabilities and values for the stored states.
* Computes `ratio = exp(new_log_prob − old_log_prob)` and applies PPO clipping `clip(ratio, 1 − ε, 1 + ε)`.
* Actor loss: `−min(ratio * A_t, clipped_ratio * A_t)` averaged over the batch.
* Critic loss: mean-squared error between returns and value predictions.
* Total loss: actor loss + 0.5 × critic loss, followed by `optimizer.step()`.

Full Workflow ASCII Loop
------------------------

```
                   TRAINING LOOP (per review)
┌──────────────────────────────────────────────────────────────────────────────┐
│ Review → State → select_action() → LLM → Extract Rating → Compute Reward     │
│                 ↑                                 ↓                          │
│      update() ← compute_returns() ← store transitions ← reward               │
└──────────────────────────────────────────────────────────────────────────────┘
```

Component Diagram (Modules and Interactions)
--------------------------------------------

```
+-----------------------------+
| RoutingPipeline             |
|-----------------------------|
| get_state()                 |
| train_step()                |
| run_single()                |
+-------------+---------------+
              |
              v
+-----------------------------+
| PPOAgent                    |
|-----------------------------|
| select_action()             |
| compute_returns()           |
| update()                    |
+-------------+---------------+
              |
              v
+-----------------------------+
| PPOPolicy (Neural Net)      |
|-----------------------------|
| actor: LLMSoftmax           |
| critic: ValueEstimator      |
+-------------+---------------+
              |
              v
+-----------------------------+
| HuggingFaceLLMSystem        |
|-----------------------------|
| load models (lazy)          |
| build prompts               |
| generate outputs            |
| extract ratings             |
+-----------------------------+
```

Detailed Step-by-Step Diagram
-----------------------------

```
                          +--------------------+
                          | Amazon Review Data |
                          +----------+---------+
                                     |
                                     v
                         +-----------------------+
                         |  RoutingPipeline      |
                         |    get_state()        |
                         +-----------+-----------+
                                     |
                         state vector built
                                     |
                                     v
                     +-----------------------------+
                     |        PPOAgent             |
                     |     select_action()         |
                     +-----------------------------+
                         |    |            |
                         |    |            |
                action ←-     |            └→ value estimate
                              |
                           log_prob

                                     |
                       chosen LLM ID |
                                     v
                  +----------------------------------+
                  |   HuggingFaceLLMSystem           |
                  | prompt → tokenize → generate     |
                  +----------------------------------+
                                     |
                                     v
                         +-----------------------+
                         |  Output Extraction    |
                         |  (regex-based rating) |
                         +-----------------------+
                                     |
                               predicted rating
                                     |
                                     v
                         +-----------------------+
                         |    Reward Function    |
                         +-----------------------+
                                     |
                                     v
                         +-----------------------+
                         |  store transition     |
                         +-----------------------+

       After batch:
                         +------------------------------+
                         |   compute_returns()          |
                         +------------------------------+
                                     |
                                     v
                         +------------------------------+
                         |   PPOAgent.update()          |
                         +------------------------------+
                                     |
                                     v
                           policy updated
```

Module Responsibilities Summary
-------------------------------

| Module               | Responsibility                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| `PROMPTS`            | Holds strict per-language instructions so LLMs emit only numeric ratings.       |
| `HuggingFaceLLMSystem` | Lazy model loading, prompt construction, generation, rating extraction, logging. |
| `PPOPolicy`          | Actor–critic neural network producing LLM probabilities and value estimates.    |
| `PPOAgent`           | PPO mechanics: sampling actions, computing returns/advantages, updating policy. |
| `RoutingPipeline`    | Bridges PPO with the LLM environment; handles state creation, reward, logging.  |
| Training loop        | Repeats sampling, routing, and PPO updates; saves checkpoints and CSV data.     |

Inference Workflow (No Training)
--------------------------------

```
review → state → PPOPolicy → choose LLM → HuggingFaceLLMSystem.generate()
→ extract rating → return prediction (no reward, no PPO update).
```

How This Fits Into train.py
---------------------------

* `RoutingPipeline.train_step` is the bridge between PPO and the LLM environment. It creates states, calls `select_action`, executes the chosen LLM query, logs the interaction, and pushes rewards to PPO.
* `HuggingFaceLLMSystem` hides tokenizer/model loading, quantization config, prompt generation, inference, and rating extraction. The PPO agent only sees `(state → action → reward)`.
* The main block configures model names, prompts, state dimension, batching parameters, and training rounds. After every round, it saves `ppo_router_policy_round{n}.pth`.
* `test.py` can reuse the same components for evaluation or inference-only routing once a policy checkpoint is loaded.

Next Steps
----------

* Adjust `state_dim` or enrich `get_state` with real features beyond language identifiers.
* Swap in different LLMs in `model_names` or expand the set if you want more routing options.
* Instrument evaluation by using the existing `test.py` or by extending `RoutingPipeline.run_single`.
