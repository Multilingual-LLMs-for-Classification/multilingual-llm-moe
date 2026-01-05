import os
import csv
import re
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
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

EMBED_DIM = 768
STATE_DIM = EMBED_DIM + NUM_LANGS

HF_TOKEN = os.getenv("HF_TOKEN")

# ==============================
# 2. Neural Bandit Policy
# ==============================
class NeuralRouter(nn.Module):
    def __init__(self, input_dim, num_models):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_models)
        )

    def forward(self, x):
        return self.net(x)

# ==============================
# 3. Memory-Safe LLM System
# ==============================
class LLMManager:
    def __init__(self, model_names, categories):
        self.model_names = model_names
        self.categories = categories
        self.models = {}
        self.tokenizers = {}

    def load_model(self, name):
        if name in self.models:
            return self.models[name], self.tokenizers[name]

        torch.cuda.empty_cache()
        gc.collect()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            quantization_config=bnb,
            token=HF_TOKEN
        )

        self.models[name] = model
        self.tokenizers[name] = tokenizer
        return model, tokenizer

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

        cleaned = raw_output.lower().replace('"', '').replace("'", '').strip()
        for cat in sorted(self.categories, key=len, reverse=True):
            if cat.lower() == cleaned or cat.lower() in cleaned:
                return cat, cat

        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned)
        return cleaned_text, "Unknown"

    def predict(self, model_idx, text):
        name = self.model_names[model_idx]
        model, tokenizer = self.load_model(name)

        prompt = (
            f"You are a strict news classifier. Read the article headline and "
            f"output ONLY a single category from this list: {self.categories}. "
            f"No words. No explanations. No punctuation. "
            f"Article: {text} Output (Category only):"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=15)

        decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        _, prediction = self.extract_label(decoded)
        return prediction

# ==============================
# 4. XLM-R Embeddings
# ==============================
class XLMREmbedder:
    def __init__(self, model_name="xlm-roberta-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.squeeze(0).cpu().numpy()

# ==============================
# 5. Evaluation Execution
# ==============================
def run_test():
    test_path = "../../data/test.csv"
    if not os.path.exists(test_path):
        return print("Test data not found!")
    
    test_df = pd.read_csv(test_path)
    
    # Initialize XLM-R embedder
    embedder = XLMREmbedder()
    
    # Reconstruct Prompts (Ensure this matches your prompt dict)
    llm_manager = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)
    
    router = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    router.load_state_dict(torch.load("neural_router_langvec5.pth"))
    router.eval()

    output_csv = "test_detailed_results_langvec5.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["text", "lang", "true_label", "model_used", "pred_label", "is_correct", "values"])

    all_results = []

    print(f"Testing Neural Bandit on {len(test_df)} samples...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Feature Engineering
        emb = embedder.encode(row['text'])  # XLM-R embedding
        lang_vec = np.zeros(NUM_LANGS, dtype=np.float32)
        if row['lang'] in LANG_MAP:
            lang_vec[LANG_MAP[row['lang']]] = 1.0
        state = np.concatenate([emb, lang_vec])
        context = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Selection (Pure Exploitation)
        with torch.no_grad():
            q_vals = router(context)
            action_idx = torch.argmax(q_vals).item()

        # Inference
        pred = llm_manager.predict(action_idx, row['text'])
        is_correct = 1 if pred == row['label'] else 0
        
        all_results.append({"lang": row['lang'], "true": row['label'], "pred": pred})
        q_vals_np = q_vals.squeeze(0).cpu().numpy()  # Convert Q-values to numpy
        q_vals_list = q_vals_np.tolist()
        # Log to CSV
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row['text'], row['lang'], row['label'], MODEL_NAMES[action_idx], pred, is_correct, q_vals_list])

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

    metrics_df = pd.concat([metrics_df, pd.DataFrame([overall_row])], ignore_index=True)

    metrics_df.to_csv("language_wise_metrics_langvec5.csv", index=False)
        
    print("\n" + "="*50)
    print("FINAL TEST METRICS (LANGUAGE-WISE + OVERALL)")
    print("="*50)
    print(metrics_df)
    print("="*50)

if __name__ == "__main__":
    run_test()
