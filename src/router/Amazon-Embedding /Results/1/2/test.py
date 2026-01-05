import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -----------------------------
# ✅ Import from train.py
from train import HuggingFaceLLMSystem, PPOAgent, RoutingPipeline, PROMPTS

# -----------------------------
# ✅ CONFIG (MUST MATCH TRAINING)
state_dim = 768 + 6   # mBERT + 6 languages

model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]

num_llms = len(model_names)

test_csv_path = "../../../../../../data/Amazon/test_subset_final.csv"

# ✅ IMPORTANT: This filename MUST match comparison.py
output_csv_test = "test_routing_results.csv"

# ✅ FINAL TRAINED CHECKPOINT
policy_path = "ppo_router_langcond_round5.pth"

# -----------------------------
# ✅ LOAD TEST DATA
test_data = pd.read_csv(
    test_csv_path,
    dtype={'review_body': str, 'language': str, 'stars': int},
    encoding='utf-8',
    on_bad_lines='skip'
)

# -----------------------------
# ✅ INIT SYSTEMS
llm_system = HuggingFaceLLMSystem(model_names, PROMPTS)

num_langs = 6   # en, fr, ja, es, zh, de
agent = PPOAgent(
    state_dim=state_dim,
    num_llms=num_llms,
    num_langs=num_langs
)

agent.policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
agent.policy.eval()

# ✅ LANGUAGE MAP (MUST MATCH TRAINING)
lang_map = {
    "en": 0,
    "fr": 1,
    "ja": 2,
    "es": 3,
    "zh": 4,
    "de": 5
}

pipeline = RoutingPipeline(
    agent,
    llm_system,
    state_dim,
    lang_map
)

# -----------------------------
# ✅ SINGLE LANGUAGE-CONDITIONAL ROUTING STEP
def run_single(pipeline, row):
    # 1. Prepare state
    state = pipeline.get_state(row['review_body'], row['language'])
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # 2. Prepare lang_id
    lang_id = torch.tensor([pipeline.lang_map.get(row['language'], 0)])

    # 3. Move both to same device as policy
    device = next(pipeline.agent.policy.parameters()).device
    state = state.to(device)
    lang_id = lang_id.to(device)

    # 4. PPO forward pass
    with torch.no_grad():
        probs, _ = pipeline.agent.policy(state, lang_id)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

    # 5. Run selected LLM
    pred_rating, decoded_output, prompt_used, llm_name_used = pipeline.llm_system.run(
        action, row['review_body'], row['language']
    )

    return {
        "selected_llm": action,
        "pred_rating": pred_rating,
        "decoded_output": decoded_output,
        "prompt": prompt_used,
        "llm_name": llm_name_used
    }

# -----------------------------
# ✅ RUN FULL TEST SET
all_outputs = []

for _, row in test_data.iterrows():
    result = run_single(pipeline, row)

    all_outputs.append({
        "review_body": row['review_body'],
        "language": row['language'],
        "true_rating": row['stars'],   # ✅ REQUIRED by comparison.py
        "selected_llm": result['selected_llm'],
        "pred_rating": result['pred_rating'],
        "decoded_output": result['decoded_output'],
        "prompt": result['prompt'],
        "llm_name": result['llm_name']
    })

df_outputs = pd.DataFrame(all_outputs)

df_outputs.to_csv(output_csv_test, index=False, encoding='utf-8')
print(f"\n✅ Test results saved to: {output_csv_test}")

# -----------------------------
# ✅ LANGUAGE-WISE METRICS
metrics = []

for lang, group in df_outputs.groupby("language"):
    y_true = group['true_rating']
    y_pred = group['pred_rating']

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics.append({
        "language": lang,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

metrics_df = pd.DataFrame(metrics)

print("\n=== ✅ PPO ROUTER Language-wise Metrics ===")
print(metrics_df)

metrics_df.to_csv("ppo_router_language_wise_metrics.csv", index=False, encoding='utf-8')

print("\n✅ Metrics saved to: ppo_router_language_wise_metrics.csv")
