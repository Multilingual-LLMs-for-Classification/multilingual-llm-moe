import os
import csv
import re
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================
# 1. Configuration (Must match Training)
# ==============================
NEWS_CATEGORIES = [
    "Technology", "Industry", "Tax & Accounting",
    "Finance", "Government & Controls", "Business & Management"
]

MODEL_NAMES = [
    "CohereLabs/aya-expanse-8b",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-3B"
]

# Ensure your HF Token is set or replace here
HF_TOKEN = os.getenv("HF_TOKEN") or "YOUR_TOKEN_HERE"
STATE_DIM = 384  # MiniLM dimension
WEIGHTS_PATH = "router_weights.npz"
TEST_DATA_PATH = "../../data/test.csv"
OUTPUT_CSV = "test_inference_results.csv"

# ==============================
# 2. Memory-Safe LLM System
# ==============================
class MemorySafeLLMSystem:
    def __init__(self, model_names, categories):
        self.model_names = model_names
        self.categories = categories
        self.current_model = None
        self.tokenizer = None
        self.loaded_model_name = None

    def unload_model(self):
        if self.current_model is not None:
            # Move to CPU to ensure VRAM is fully released
            self.current_model.cpu()
            del self.current_model
            del self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            self.loaded_model_name = None

    def load_model(self, model_name):
        if self.loaded_model_name == model_name:
            return
        self.unload_model()
        
        print(f"\n[System] Loading {model_name} for test evaluation...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.current_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            quantization_config=bnb_cfg, 
            token=HF_TOKEN
        )
        self.loaded_model_name = model_name

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


    def run_inference(self, model_idx, text, lang):
        model_name = self.model_names[model_idx]
        self.load_model(model_name)
        
        prompt = f"You are a strict news classifier. Read the article and output ONLY a single category from this list: {self.categories}. No words. No explanations. No punctuation. Article: {text} Output (Category only):"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.current_model.device)
        
        with torch.no_grad():
            output_tokens = self.current_model.generate(**inputs, max_new_tokens=15)
        
        decoded = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        out, prediction = self.extract_label(decoded)
        return prediction, decoded

# ==============================
# 3. Main Execution
# ==============================
def main():
    # 1. Load trained weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: {WEIGHTS_PATH} not found. Run training first!")
        return
    
    weights = np.load(WEIGHTS_PATH)
    A = weights['A']
    b = weights['b']

    # 2. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # 3. Initialize Tools
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    llm_system = MemorySafeLLMSystem(MODEL_NAMES, NEWS_CATEGORIES)
    
    # Initialize Output CSV with header
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lang", "true_label", "router_choice", "raw_output", "final_pred", "is_correct"])

    print(f"Starting Evaluation on {len(test_df)} samples...")
    correct_count = 0

    # 4. Evaluation Loop
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Get Context Vector
        context = embedder.encode(row['text']).reshape(-1, 1)
        
        # Thompson Sampling / UCB isn't needed here; use Learned Theta (Exploitation)
        scores = np.zeros(len(MODEL_NAMES))
        for a in range(len(MODEL_NAMES)):
            # Theta = A_inv * b
            theta = np.linalg.inv(A[a]) @ b[a]
            scores[a] = theta.T @ context 
        
        best_model_idx = np.argmax(scores)
        model_name = MODEL_NAMES[best_model_idx]
        
        # Execute Inference
        pred, raw_out = llm_system.run_inference(best_model_idx, row['text'], row['lang'])
        
        # Verify Accuracy
        success = 1 if pred == row['label'] else 0
        correct_count += success
        
        # SAVE TO CSV IMMEDIATELY (Safety if script is interrupted)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row['text'], row['lang'], row['label'], model_name, raw_out, pred, success])

    # 5. Summary Statistics
    print("\nCalculating Language-wise Metrics...")
    results_df = pd.read_csv(OUTPUT_CSV)
    metric_rows = []

    # Calculate metrics per language
    for lang, group in results_df.groupby('lang'):
        y_true = group['true_label']
        y_pred = group['final_pred']
        
        acc = accuracy_score(y_true, y_pred)
        # zero_division=0 handles cases where a category was never predicted
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        metric_rows.append({
            "Language": lang,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Sample_Count": len(group)
        })

    # Save Metrics to CSV
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv("language_wise_metrics.csv", index=False)
    
    print("="*30)
    print(metrics_df)
    print("="*30)
    print(f"Detailed logs: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()