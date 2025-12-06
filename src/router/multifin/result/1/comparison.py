import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import numpy as np

# NOTE: Assuming the training file is named 'combine_csv.py' and contains all necessary classes/configs.
from train import HuggingFaceLLMSystem, PROMPTS, NEWS_CATEGORIES

# -----------------------------
# Config
# Note: state_dim is technically irrelevant here as PPO agent is not used, 
# but kept for consistency if we adapt state preparation logic later.
state_dim = 20 

model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]

num_llms = len(model_names)

# --- Updated paths and file names for News Classification ---
test_csv_path = "../../data/test.csv" # Placeholder path for news data
output_csv_test_prefix = "llm_news_benchmark_outputs"
# The policy_path is not needed for this script, but kept if you plan to extend.
# policy_path = "ppo_router_policy_news_round5.pth" 
# -----------------------------------------------------------

# -----------------------------
# Load test data (Modified to use 'article_text' and 'category')
try:
    test_data = pd.read_csv(
        test_csv_path,
        # Update column names and expected types for news classification
        dtype={'text': str, 'lang': str, 'label': str},
        encoding='utf-8',
        on_bad_lines='skip'
    )
except FileNotFoundError:
    print(f"Error: Test data file not found at {test_csv_path}. Please check the path.")
    exit(1)

# -----------------------------
# Initialize LLM system
# NOTE: The LLMSystem needs the categories list for its prompt generation and label extraction
llm_system = HuggingFaceLLMSystem(model_names, PROMPTS, NEWS_CATEGORIES)

# -----------------------------
# Function to run test through a **specific LLM**
def run_all_with_llm(llm_index, llm_name):
    print(f"\n=== Running benchmark for LLM: {llm_name} ===")
    all_outputs = []

    for index, row in test_data.iterrows():
        # Run the specific LLM
        # The true label is 'category' (string) not 'stars' (int)
        true_category = row['label'] 
        
        # llm_system.run returns: pred_category (str), decoded_output (str), prompt_used (str), llm_name_used (str)
        pred_category, decoded_output, prompt_used, _,cleaned = llm_system.run(
            llm_index, 
            row['text'], # Use 'article_text'
            row['lang']
        )

        all_outputs.append({
            "article_text": row['text'], # Renamed
            "language": row['lang'],
            "true_category": true_category, # Renamed
            "pred_category": pred_category, # Renamed
            "decoded_output": decoded_output,
            "prompt": prompt_used,
            "llm_name_used": llm_name
        })
        
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} test samples for {llm_name}...")


    # Save CSV
    safe_llm_name = llm_name.replace('/', '_')
    output_csv = f"{output_csv_test_prefix}_{safe_llm_name}.csv"
    pd.DataFrame(all_outputs).to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved outputs to {output_csv}")

    # Calculate language-wise metrics
    df_outputs = pd.DataFrame(all_outputs)
    metrics = []
    
    # Define a list of all possible labels for consistent metric calculation
    # This prevents errors if a language group is missing one category
    valid_labels = NEWS_CATEGORIES 

    for lang, group in df_outputs.groupby("language"):
        y_true = group['true_category']
        y_pred = group['pred_category']

        acc = accuracy_score(y_true, y_pred)
        # Use weighted average for multi-class classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0, labels=valid_labels
        )

        metrics.append({
            "llm_name": llm_name,
            "language": lang,
            "num_samples": len(group),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = f"language_wise_metrics_{safe_llm_name}.csv"
    
    # Calculate overall (macro) metrics for this specific LLM
    y_true_all = df_outputs['true_category']
    y_pred_all = df_outputs['pred_category']
    
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average='macro', zero_division=0, labels=valid_labels
    )
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    
    overall_row = pd.DataFrame([{
        "llm_name": llm_name,
        "language": "Overall (Macro)",
        "num_samples": len(df_outputs),
        "accuracy": overall_acc,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1_score": overall_f1,
    }])
    
    final_metrics_df = pd.concat([metrics_df, overall_row], ignore_index=True)
    
    print("\n--- Benchmark Metrics ---")
    print(final_metrics_df.to_string(index=False))
    
    final_metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8')
    print(f"Saved metrics to {metrics_csv}")

# -----------------------------
# Run tests separately for each LLM
llm_name_map = {
    0: "Qwen/Qwen2.5-3B",
    1: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    2: "CohereLabs/aya-23-8B"
}

if __name__ == '__main__':
    for idx, name in llm_name_map.items():
        run_all_with_llm(idx, name)