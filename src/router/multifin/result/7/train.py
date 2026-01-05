import os
import csv
import re
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ------------------------------
# 1. Configuration & Constants
# ------------------------------
NEWS_CATEGORIES = [
    "Technology", "Industry", "Tax & Accounting",
    "Finance", "Government & Controls", "Business & Management"
]

MODEL_NAMES = [
    "CohereLabs/aya-expanse-8b",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-3B"
]

HF_TOKEN = os.getenv("HF_TOKEN")
STATE_DIM = 384 # Dimension for 'paraphrase-multilingual-MiniLM-L12-v2'
ALPHA = 1.0     # Exploration parameter

# ------------------------------
# 2. LLM Sanitizer
# ------------------------------
class LLMSanitizer:
    def __init__(self, categories):
        self.categories = categories

    def clean(self, raw_text):
        # Basic cleanup of LLM output
        clean_text = raw_text.split("Output (Category only):")[-1].strip()

        clean_text = clean_text.split('\n')[0].strip()
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        
        for category in self.categories:
            if category.lower() in clean_text.lower():
                return category
        return "Unknown"

# ------------------------------
# 3. Memory-Managed LLM System
# ------------------------------
class LLMManager:
    def __init__(self, categories):
        self.categories = categories
        self.tokenizer = None
        self.model = None
        self.current_model_name = None
        self.sanitizer = LLMSanitizer(categories)

    def load_model(self, model_name):
        if self.current_model_name == model_name:
            return
        
        # --- CRITICAL: CLEAR VRAM ---
        if self.model is not None:
            print(f"\n[Memory] Evicting {self.current_model_name} from GPU...")
            self.model.cpu() # Move to CPU first
            del self.model
            del self.tokenizer
            gc.collect()
            torch.cuda.empty_cache() # Clear PyTorch cache

        print(f"[System] Loading {model_name} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            quantization_config=bnb_config,
            token=HF_TOKEN
        )
        self.current_model_name = model_name
    
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


    def run(self, model_name, text, lang):
        self.load_model(model_name)
        prompt = f"You are a strict news classifier. Read the English article and output ONLY a single category from this list: {self.categories}. No words. No explanations. No punctuation. Article: {text} Output (Category only):"
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_new_tokens=10)
        
        decoded = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        out, prediction = self.extract_label(decoded)
        return prediction, decoded

# ------------------------------
# 4. LinUCB Router
# ------------------------------
class LinUCBRouter:
    def __init__(self, num_arms, dim, alpha):
        self.alpha = alpha
        self.A = [np.identity(dim) for _ in range(num_arms)]
        self.b = [np.zeros((dim, 1)) for _ in range(num_arms)]

    def select_arm(self, x):
        x = x.reshape(-1, 1)
        p = np.zeros(len(self.A))
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
        return np.argmax(p)

    def update(self, arm_idx, x, reward):
        x = x.reshape(-1, 1)
        self.A[arm_idx] += x @ x.T
        self.b[arm_idx] += reward * x

# ------------------------------
# 5. Training Loop
# ------------------------------
def main():
    data = pd.read_csv("../../data/train.csv")
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    llm_manager = LLMManager(NEWS_CATEGORIES)
    router = LinUCBRouter(len(MODEL_NAMES), STATE_DIM, ALPHA)
    
    output_csv = "cmab_training_results.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "lang", "true_label", "model", "pred", "raw", "reward"])

    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    for i, row in tqdm(shuffled_data.iterrows(), total=len(data)):
        context = embedder.encode(row['text'])
        arm_idx = router.select_arm(context)
        model_name = MODEL_NAMES[arm_idx]
        
        pred, raw = llm_manager.run(model_name, row['text'], row['lang'])
        reward = 1.0 if pred == row['label'] else 0.0
        
        router.update(arm_idx, context, reward)
        
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row['text'], row['lang'], row['label'], model_name, pred, raw, reward])

    np.savez("router_weights.npz", A=router.A, b=router.b)
    print("Done! Weights saved.")

if __name__ == "__main__":
    main()