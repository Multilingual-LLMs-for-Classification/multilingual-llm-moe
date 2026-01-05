import os
import csv
import re
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================
# 1. Configuration & Categories
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

# Map language keys in CSV to indices for One-Hot Vector
LANG_MAP = {"Danish": 0, "English": 1, "Spanish": 2, "Polish": 3, "Turkish": 4}
NUM_LANGS = len(LANG_MAP)
EMBED_DIM = 384  # MiniLM dimension
STATE_DIM = EMBED_DIM + NUM_LANGS # 389 features total

HF_TOKEN = os.getenv("HF_TOKEN") or "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"
EPSILON = 0.2  # Exploration rate
LEARNING_RATE = 1e-3

# ==============================
# 2. Neural Bandit Policy
# ==============================
class NeuralRouter(nn.Module):
    def __init__(self, input_dim, num_models):
        super(NeuralRouter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_models) # Predicts binary reward for each LLM
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
        self.model = None
        self.models = {}
        self.tokenizer = None
        self.loaded_name = None
        self.tokenizers = {}

    def load_model(self, name):
        if name in self.models:
            self.model = self.models[name]
            self.tokenizer = self.tokenizers[name]
        else:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                name, device_map="auto", quantization_config=bnb, token=HF_TOKEN
            )
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
        name = self.model_names[model_idx]
        self.load_model(name)
        
        prompt = f"You are a strict news classifier. Read the article headline and output ONLY a single category from this list: {self.categories}. No words. No explanations. No punctuation. Article: {text} Output (Category only):"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_new_tokens=15)
        
        decoded = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        _, prediction = self.extract_label(decoded)
       
        return prediction, decoded

# ==============================
# 4. Main Training Script
# ==============================
def main():
    # 1. Setup Data & Tools
    data_path = "../../data/train.csv"
    if not os.path.exists(data_path):
        print("Error: train.csv not found.")
        return
    
    # Load original data
    original_data = pd.read_csv(data_path)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    llm_manager = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)
    
    # 2. Init Neural Bandit
    router_net = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    optimizer = optim.Adam(router_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- NEW: Training Hyperparameters ---
    ROUNDS = 2  # Set how many iterations you want
    EPSILON_DECAY = 0.8 # Reduce exploration as rounds progress
    
    output_csv = "neural_bandit_train_results3.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["round", "text", "lang", "label", "model_chosen", "pred", "reward"])

    # 

    print(f"Starting Neural Bandit Training: {ROUNDS} Rounds...")

    # 3. Multi-Round Training Loop
    for r in range(ROUNDS):
        print(f"\n>>> ROUND {r+1} / {ROUNDS}")
        
        # Shuffle data at the start of every round
        current_data = original_data.sample(frac=1).reset_index(drop=True)
        current_epsilon = EPSILON * (EPSILON_DECAY ** r) # Explore less over time

        for i, row in tqdm(current_data.iterrows(), total=len(current_data)):
            # --- FEATURE ENGINEERING ---
            emb = embedder.encode(row['text'])
            lang_vec = np.zeros(NUM_LANGS)
            if row['lang'] in LANG_MAP: lang_vec[LANG_MAP[row['lang']]] = 1
            context = torch.tensor(np.concatenate([emb, lang_vec])).float()

            # --- SELECTION ---
            if random.random() < current_epsilon:
                action_idx = random.randint(0, len(MODEL_NAMES) - 1)
            else:
                with torch.no_grad():
                    q_values = router_net(context)
                    action_idx = torch.argmax(q_values).item()

            # --- EXECUTION ---
            pred, raw_out = llm_manager.predict(action_idx, row['text'], row['lang'])
            reward = 1.0 if pred == row['label'] else 0.0

            # --- NEURAL UPDATE (Backpropagation) ---
            predicted_q = router_net(context)
            target_q = predicted_q.clone().detach()
            target_q[action_idx] = reward
            
            loss = criterion(predicted_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([r+1, row['text'], row['lang'], row['label'], MODEL_NAMES[action_idx], pred, reward])

    # 4. SAVE MODEL
    torch.save(router_net.state_dict(), "neural_bandit_router3.pth")
    print(f"\nTraining Complete after {ROUNDS} iterations.")

if __name__ == "__main__":
    # Import PROMPTS here if not defined globally in your environment
    main()