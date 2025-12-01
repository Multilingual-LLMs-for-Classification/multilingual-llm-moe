import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import numpy as np

# -----------------------------
# Import your classes (all in train.py)
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


test_csv_path = "../fyp_multilingual_Text_classification/data/test_subset_final.csv"
output_csv_test_prefix = "llm_test_outputs"
policy_path = "ppo_router_policy_multilang.pth"

# -----------------------------
# Load test data
test_data = pd.read_csv(
    test_csv_path,
    dtype={'review_body': str, 'language': str, 'stars': int},
    encoding='utf-8',
    on_bad_lines='skip'
)

# -----------------------------
# Initialize LLM system
llm_system = HuggingFaceLLMSystem(model_names, PROMPTS)

# -----------------------------
# Function to run test through a **specific LLM**
def run_all_with_llm(llm_index, llm_name):
    all_outputs = []

    for _, row in test_data.iterrows():
        # Use the same state as training for compatibility
        state = torch.zeros((1, state_dim))  # dummy state; not used since PPO not used here

        # Run the specific LLM
        pred_rating, decoded_output, prompt_used, _ = llm_system.run(llm_index, row['review_body'], row['language'])

        all_outputs.append({
            "review_body": row['review_body'],
            "language": row['language'],
            "true_stars": row['stars'],
            "pred_rating": pred_rating,
            "decoded_output": decoded_output,
            "prompt": prompt_used,
            "llm_name_used": llm_name
        })

    # Save CSV
    output_csv = f"{output_csv_test_prefix}_{llm_name.replace('/', '_')}.csv"
    pd.DataFrame(all_outputs).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved outputs to {output_csv}")

    # Calculate language-wise metrics
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
            "f1_score": f1,
            "num_samples": len(group),
            "llm_name": llm_name
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = f"language_wise_metrics_{llm_name.replace('/', '_')}.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8')
    print(f"Saved metrics to {metrics_csv}")
    print(metrics_df)

# -----------------------------
# Run tests separately for TinyLlama and Qwen
llm_name_map = {
    0: "Qwen/Qwen2.5-3B",
    1: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    2: "CohereLabs/aya-23-8B"
}

for idx, name in llm_name_map.items():
    run_all_with_llm(idx, name)
