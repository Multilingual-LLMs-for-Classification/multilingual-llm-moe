import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -----------------------------
# Import your classes
from train import HuggingFaceLLMSystem, PPOAgent, RoutingPipeline, PROMPTS

# -----------------------------
# ✅ Config MUST match training
state_dim = 768 + 6

model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]

num_llms = len(model_names)

test_csv_path = "../../../../../data/Amazon/test_subset_final.csv"
output_csv_test = "ppo_router_test_outputs.csv"

# ✅ Load your FINAL trained checkpoint
policy_path = "ppo_router_mbert_round5.pth"

# -----------------------------
# Load test data
test_data = pd.read_csv(
    test_csv_path,
    dtype={'review_body': str, 'language': str, 'stars': int},
    encoding='utf-8',
    on_bad_lines='skip'
)

# -----------------------------
# Initialize LLM system, agent, and pipeline
llm_system = HuggingFaceLLMSystem(model_names, PROMPTS)

agent = PPOAgent(state_dim=state_dim, num_llms=num_llms)
agent.policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
agent.policy.eval()

pipeline = RoutingPipeline(agent, llm_system, state_dim)

# -----------------------------
# ✅ Single inference using PPO router
def run_single(pipeline, row):
    state = pipeline.get_state(row['review_body'], row['language'])
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probs, _ = pipeline.agent.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

    pred_rating, decoded_output, prompt_used, llm_name_used = pipeline.llm_system.run(
        action, row['review_body'], row['language']
    )

    return {
        "action": action,
        "pred_rating": pred_rating,
        "decoded_output": decoded_output,
        "prompt": prompt_used,
        "llm_name_used": llm_name_used
    }

# -----------------------------
# Run inference on test set
all_outputs = []

for _, row in test_data.iterrows():
    result = run_single(pipeline, row)

    all_outputs.append({
        "review_body": row['review_body'],
        "language": row['language'],
        "true_stars": row['stars'],
        "action": result['action'],
        "pred_rating": result['pred_rating'],
        "decoded_output": result['decoded_output'],
        "prompt": result['prompt'],
        "llm_name_used": result['llm_name_used']
    })

pd.DataFrame(all_outputs).to_csv(output_csv_test, index=False, encoding='utf-8')

# -----------------------------
# Language-wise Metrics for PPO Router
df_outputs = pd.DataFrame(all_outputs)
metrics = []

for lang, group in df_outputs.groupby("language"):
    y_true = group['true_stars']
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

print("\n=== PPO ROUTER Language-wise Metrics ===")
print(metrics_df)

metrics_df.to_csv("ppo_router_language_wise_metrics.csv", index=False, encoding='utf-8')
