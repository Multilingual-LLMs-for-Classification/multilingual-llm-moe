import os
import csv
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ==============================
# 1. Configuration (Must Match Train)
# ==============================
NEWS_CATEGORIES = [
    "Technology", "Industry", "Tax & Accounting", 
    "Finance", "Government & Controls", "Business & Management"
]


MODEL_NAMES = [
    "Qwen/Qwen2.5-3B",
    "CohereLabs/aya-expanse-8b",
    "meta-llama/Llama-3.1-8B"
]

LANG_MAP = {"Danish": 0, "English": 1, "Spanish": 2, "Polish": 3, "Turkish": 4}
NUM_LANGS = len(LANG_MAP)
STATE_DIM = 384 + NUM_LANGS 
HF_TOKEN = os.getenv("HF_TOKEN") or "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"

# ==============================
# 2. Neural Bandit Policy
# ==============================
class NeuralRouter(nn.Module):
    def __init__(self, input_dim, num_models):
        super(NeuralRouter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents overfitting to specific samples
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_models),
            nn.Sigmoid() # Forces output to 0-1 (Expected Accuracy)
        )

    def forward(self, x):
        return self.fc(x)
# ==============================
# 3. Memory-Safe LLM System
# ==============================
class LLMManager:
    def __init__(self, model_names, categories):
        self.model_names = model_names
        self.categories = categories
        self.models = {}
        self.tokenizers = {}
        self.model = None
        self.tokenizer = None

    def load_model(self, name):
        if name in self.models:
            self.model = self.models[name]
            self.tokenizer = self.tokenizers[name]
        else:
            # Clear GPU cache before loading new model
            torch.cuda.empty_cache()
            gc.collect()
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                quantization_config=bnb,
                use_auth_token=HF_TOKEN
            )
            # Cache it for future use
            self.models[name] = self.model
            self.tokenizers[name] = self.tokenizer

    def extract_label(self, decoded):
        prompt_tags = [
            "Output (Category only):", "Output (Kun kategori):", 
            "Salida (Categoría solamente):", "Wyjście (Tylko kategoria):", 
            "Çıktı (Sadece Kategori):"
        ]
        raw_output = decoded
        for tag in prompt_tags:
            if tag in raw_output:
                raw_output = raw_output.split(tag)[-1].strip()
                break
        else:
            raw_output = decoded.strip().split('\n')[-1].strip()

        cleaned_output = raw_output.lower().replace('"', '').replace("'", '').strip()
        sorted_categories = sorted(self.categories, key=len, reverse=True)

        for category in sorted_categories:
            if category.lower() == cleaned_output or category.lower() in cleaned_output:
                return category, category

        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_output)
        string_output = " ".join([t.strip() for t in cleaned_text.split()])
        return string_output, "Unknown"


    def predict(self, model_idx, text, lang):
        self.load_model(self.model_names[model_idx])
        prompt = f"You are a strict news classifier. Read the article headline and output ONLY a single category from this list: {self.categories}. No words. No explanations. No punctuation. Article: {text} Output (Category only):"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=15)
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        _, prediction = self.extract_label(decoded)
        return prediction

# ==============================
# 4. Evaluation Execution
# ==============================
def run_test():
    # Load Data & Model
    test_path = "../../data/test.csv"
    if not os.path.exists(test_path): return print("Test data not found!")
    
    test_df = pd.read_csv(test_path)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Reconstruct Prompts (Ensure this matches your prompt dict)
    llm_manager = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)
    
    router = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    router.load_state_dict(torch.load("neural_bandit_buffer_embedding.pth"))
    router.eval()

    output_csv = "test_detailed_results_buffer2.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["text", "lang", "true_label", "model_used", "pred_label", "is_correct"])

    all_results = []

    print(f"Testing Neural Bandit on {len(test_df)} samples...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Feature Engineering
        emb = embedder.encode(row['text'])
        lang_vec = np.zeros(NUM_LANGS)
        if row['lang'] in LANG_MAP: lang_vec[LANG_MAP[row['lang']]] = 1
        context = torch.tensor(np.concatenate([emb, lang_vec])).float()

        # Selection (Pure Exploitation)
        with torch.no_grad():
            action_idx = torch.argmax(router(context)).item()
        
        # Inference
        pred = llm_manager.predict(action_idx, row['text'], row['lang'])
        is_correct = 1 if pred == row['label'] else 0
        
        all_results.append({"lang": row['lang'], "true": row['label'], "pred": pred})
        
        # Log to CSV
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row['text'], row['lang'], row['label'], MODEL_NAMES[action_idx], pred, is_correct])

    # --- METRICS CALCULATION ---
    results_df = pd.DataFrame(all_results)
    lang_metrics = []

    for lang, group in results_df.groupby('lang'):
        p, r, f1, _ = precision_recall_fscore_support(group['true'], group['pred'], average='macro', zero_division=0)
        acc = accuracy_score(group['true'], group['pred'])
        lang_metrics.append({
            "Language": lang, "Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1, "Count": len(group)
        })

    metrics_df = pd.DataFrame(lang_metrics)

# --- OVERALL METRICS ---
    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        results_df['true'],
        results_df['pred'],
        average='macro',
        zero_division=0
    )
    overall_acc = accuracy_score(results_df['true'], results_df['pred'])

    overall_row = {
        "Language": "OVERALL",
        "Accuracy": overall_acc,
        "Precision": overall_p,
        "Recall": overall_r,
        "F1-Score": overall_f1,
        "Count": len(results_df)
    }

    # Append overall row
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([overall_row])],
        ignore_index=True
    )

    # Save to SAME CSV
    metrics_df.to_csv("language_wise_metrics_buffer.csv", index=False)
        
    print("\n" + "="*50)
    print("FINAL TEST METRICS (LANGUAGE-WISE + OVERALL)")
    print("="*50)
    print(metrics_df)
    print("="*50)

if __name__ == "__main__":
        # Ensure PROMPTS is defined here or imported
    run_test()