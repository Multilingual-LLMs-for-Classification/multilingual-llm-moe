import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# -----------------------------
# Import from train.py
from train import HuggingFaceLLMSystem, PROMPTS

# -----------------------------
# Config
model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]

test_csv_path = "../../../../../data/Amazon/test_subset_final.csv"
output_csv_test_prefix = "llm_test_outputs"

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
# Run all samples using ONE fixed LLM
def run_all_with_llm(llm_index, llm_name):
    all_outputs = []

    for _, row in test_data.iterrows():
        pred_rating, decoded_output, prompt_used, _ = llm_system.run(
            llm_index, row['review_body'], row['language']
        )

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

    # -----------------------------
    # Language-wise metrics
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

    print("\n=== Metrics for", llm_name, "===")
    print(metrics_df)


# -----------------------------
# Run all base LLMs
for idx, name in enumerate(model_names):
    run_all_with_llm(idx, name)
