import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv
import numpy as np

# NOTE: Assuming the training file is named 'combine_csv.py' and contains all necessary classes/configs.
# If your training file is named 'train.py', please change the import line below.
from train import HuggingFaceLLMSystem, PPOAgent, RoutingPipeline

# -----------------------------
# Config
state_dim = 20

model_names = [
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CohereLabs/aya-23-8B"
]


NEWS_CATEGORIES = [
    "Technology",
    "Industry",
    "Tax & Accounting",
    "Finance",
    "Government & Controls",
    "Business & Management"
]


PROMPTS = {
    "Danis": (
        "Du er en streng nyhedsklassifikator. Læs den danske artikel og udskriv KUN én "
        "kategori fra denne liste: {categories}. Ingen ord. Ingen forklaringer. Ingen tegnsætning.\n"
        "Artikel: \"{text}\"\n"
        "Output (Kun kategori):"
    ),

    "English": (
        "You are a strict news classifier. Read the English article and output ONLY a single "
        "category from this list: {categories}. No words. No explanations. No punctuation.\n"
        "Article: \"{text}\"\n"
        "Output (Category only):"
    ),

    "Spanis": (
        "Eres un clasificador estricto de noticias. Lee el artículo en español y devuelve SOLO una "
        "categoría de esta lista: {categories}. Sin palabras. Sin explicaciones. Sin puntuación.\n"
        "Artículo: \"{text}\"\n"
        "Salida (Categoría solamente):"
    ),

    "Polis": (
        "Jesteś rygorystycznym klasyfikatorem wiadomości. Przeczytaj polski artykuł i zwróć TYLKO jedną "
        "kategorię z tej listy: {categories}. Bez słów. Bez wyjaśnień. Bez znaków interpunkcyjnych.\n"
        "Artykuł: \"{text}\"\n"
        "Wyjście (Tylko kategoria):"
    ),

    "Turkis": (
        "Sen katı bir haber sınıflandırıcısısın. Türkçe makaleyi oku ve SADECE bu listeden tek bir "
        "kategori çıktı: {categories}. Kelime yok. Açıklama yok. Noktalama işareti yok.\n"
        "Makale: \"{text}\"\n"
        "Çıktı (Sadece Kategori):"
    )
}

num_llms = len(model_names)

# --- Updated paths and file names for News Classification ---
test_csv_path = "../../data/test.csv" # Placeholder path for news data
output_csv_test = "llm_news_test_outputs.csv"
policy_path = "ppo_router_policy_news_round1.pth" # Policy name matches the new task
# -----------------------------------------------------------

# -----------------------------
# Load test data (Modified to use 'article_text' and 'category')
try:
    test_data = pd.read_csv(
        test_csv_path,
        dtype={'text': str, 'lang': str, 'label': str},
        encoding='utf-8',
        on_bad_lines='skip'
    )
except FileNotFoundError:
    print(f"Error: Test data file not found at {test_csv_path}. Please check the path.")
    exit(1)

# -----------------------------
# Initialize LLM system, agent, and pipeline
# NOTE: The LLMSystem now needs the categories list
llm_system = HuggingFaceLLMSystem(model_names, PROMPTS, NEWS_CATEGORIES)
agent = PPOAgent(state_dim=state_dim, num_llms=num_llms)

try:
    # Load the trained policy state dictionary
    agent.policy.load_state_dict(torch.load(policy_path))
    agent.policy.eval()
except FileNotFoundError:
    print(f"Error: Policy checkpoint not found at {policy_path}. Please train the model first.")
    exit(1)

pipeline = RoutingPipeline(agent, llm_system, state_dim)

# -----------------------------
# Add a run_single method using get_state
def run_single(pipeline, row):
    """
    Run inference for a single article using the same state representation as training.
    """
    # 1. Prepare state using the same function as training
    # Use 'article_text' instead of 'review_body'
    state = pipeline.get_state(row['text'], row['lang'])  # shape [state_dim]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]

    # 2. PPO agent forward pass
    with torch.no_grad():
        probs, _ = pipeline.agent.policy(state)
        # We need to make a deterministic choice for testing (usually argmax), 
        # but sticking to the training sampling method for consistency.
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

    # 3. Run LLM
    # The return values are now labels (strings) instead of ratings (integers)
    pred_category, decoded_output, prompt_used, llm_name_used, cleaned_output = pipeline.llm_system.run(
        action, row['text'], row['lang']
    )

    return {
        "action": action,
        "pred_category": pred_category, # Renamed
        "decoded_output": decoded_output,
        "prompt": prompt_used,
        "llm_name_used": llm_name_used
    }

# -----------------------------
# Run inference on all test data
print("Starting inference on test data...")
all_outputs = []
for index, row in test_data.iterrows():
    result = run_single(pipeline, row)

    # Append all info (Updated column names)
    all_outputs.append({
        "article_text": row['text'], # Renamed
        "language": row['lang'],
        "true_category": row['label'], # Renamed
        "action": result['action'],
        "pred_category": result['pred_category'], # Renamed
        "decoded_output": result['decoded_output'],
        "prompt": result['prompt'],
        "llm_name_used": result['llm_name_used']
    })
    
    if (index + 1) % 10 == 0:
        print(f"Processed {index + 1} test samples...")


# Save detailed outputs
pd.DataFrame(all_outputs).to_csv(output_csv_test, index=False, encoding='utf-8')
print(f"\nDetailed test outputs saved to: {output_csv_test}")

# -----------------------------
# Compute language-wise classification metrics
df_outputs = pd.DataFrame(all_outputs)
metrics = []

for lang, group in df_outputs.groupby("language"):
    # Use the updated column names for true and predicted labels
    y_true = group['true_category'] 
    y_pred = group['pred_category']

    acc = accuracy_score(y_true, y_pred)
    # The average='weighted' is appropriate for multi-class classification
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, 
        labels=NEWS_CATEGORIES # Ensure metrics are calculated over all expected labels
    )

    metrics.append({
        "language": lang,
        "sample_count": len(group),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

metrics_df = pd.DataFrame(metrics)
print("\n=== Language-wise Classification Metrics ===")
print(metrics_df.to_string(index=False))

# Compute overall (macro) metrics
y_true_all = df_outputs['true_category']
y_pred_all = df_outputs['pred_category']
overall_acc = accuracy_score(y_true_all, y_pred_all)
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
    y_true_all, y_pred_all, average='macro', zero_division=0,
    labels=NEWS_CATEGORIES
)

overall_metrics = {
    "language": "Overall (Macro)",
    "sample_count": len(df_outputs),
    "accuracy": overall_acc,
    "precision": overall_precision,
    "recall": overall_recall,
    "f1_score": overall_f1
}
print("\n=== Overall (Macro) Metrics ===")
print(pd.Series(overall_metrics).to_string())

metrics_df.to_csv("language_wise_metrics.csv", index=False, encoding='utf-8')
print("\nMetrics saved to language_wise_metrics.csv")