import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import numpy as np

# -----------------------------
# Import your classes
# Make sure train.py is in the same folder as test.py
from train import HuggingFaceLLMSystem, PPOAgent, RoutingPipeline, PROMPTS

# -----------------------------
# Config
state_dim = 20

model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]

num_llms = len(model_names)

test_csv_path = "../../data/Amazon/test_subset_final.csv"
output_csv_test = "llm_test_outputs.csv"
policy_path = "ppo_router_policy_round5.pth"

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
agent.policy.load_state_dict(torch.load(policy_path))
agent.policy.eval()

pipeline = RoutingPipeline(agent, llm_system, state_dim)

# -----------------------------
# Add a run_single method using get_state
def run_single(pipeline, row):
    """
    Run inference for a single review using the same state representation as training.
    """
    # 1. Prepare state using the same function as training
    state = pipeline.get_state(row['review_body'], row['language'])  # shape [state_dim]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]

    # 2. PPO agent forward pass
    with torch.no_grad():
        probs, _ = pipeline.agent.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

    # 3. Run LLM
    pred_rating, decoded_output, prompt_used, llm_name_used, cleaned_output = pipeline.llm_system.run(
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
# Run inference on all test data
all_outputs = []
for _, row in test_data.iterrows():
    result = run_single(pipeline, row)

    # Append all info
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

# Save detailed outputs
pd.DataFrame(all_outputs).to_csv(output_csv_test, index=False, encoding='utf-8')

# -----------------------------
# Compute language-wise classification metrics
df_outputs = pd.DataFrame(all_outputs)
metrics = []

for lang, group in df_outputs.groupby("language"):
    y_true = group['true_stars']
    y_pred = group['pred_rating']

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    metrics.append({
        "language": lang,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

metrics_df = pd.DataFrame(metrics)
print("\n=== Language-wise Metrics ===")
print(metrics_df)
metrics_df.to_csv("language_wise_metrics.csv", index=False, encoding='utf-8')
