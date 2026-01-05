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
from collections import deque
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================
# 1. Configuration & Categories
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
EMBED_DIM = 384 
STATE_DIM = EMBED_DIM + NUM_LANGS 

HF_TOKEN = os.getenv("HF_TOKEN") or "YOUR_TOKEN_HERE"
EPSILON = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
BUFFER_SIZE = 1000  # How many experiences to keep in memory

# ==============================
# 2. Replay Buffer & Router
# ==============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards).float()

    def __len__(self):
        return len(self.buffer)

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
# 3. LLM Manager (Same as your original)
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
            # Note: Clear cache before loading new model to save VRAM
            torch.cuda.empty_cache()
            gc.collect()
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
    
    
    def compute_reward_optimized(self, pred_label, true_label, category_to_index, category_embeddings, alpha=0.7):
        """
        Reward based on exact match + embedding similarity.
        Uses precomputed embeddings for categories to save computation.
        """
        # Exact match reward
        acc_reward = 1.0 if pred_label == true_label else 0.0

        # Embedding similarity reward
        if pred_label not in category_to_index or true_label not in category_to_index:
            sim_reward = 0.0
        else:
            pred_emb = category_embeddings[category_to_index[pred_label]].reshape(1, -1)
            true_emb = category_embeddings[category_to_index[true_label]].reshape(1, -1)
            sim = cosine_similarity(pred_emb, true_emb)[0][0]
            sim_reward = (sim + 1) / 2  # scale [-1,1] -> [0,1]

        # Combined reward
        reward = alpha * acc_reward + (1 - alpha) * sim_reward
        return reward

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
    data_path = "../../data/train.csv"
    if not os.path.exists(data_path): return
    
    original_data = pd.read_csv(data_path)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    llm_manager = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)
    
    router_net = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    optimizer = optim.Adam(router_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(BUFFER_SIZE)

    ROUNDS = 3
    EPSILON_DECAY = 0.8 

    category_embeddings = embedder.encode(NEWS_CATEGORIES, convert_to_numpy=True)
    category_to_index = {cat: idx for idx, cat in enumerate(NEWS_CATEGORIES)}

    output_csv = "neural_bandit_train_results_buffer2.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["round", "text", "lang", "label", "model_chosen", "pred", "reward"])

    print(f"Starting Training with Replay Buffer...")

    for r in range(ROUNDS):
        current_data = original_data.sample(frac=1).reset_index(drop=True)
        current_epsilon = EPSILON * (EPSILON_DECAY ** r) # epsilon value reduces in every round

        for i, row in tqdm(current_data.iterrows(), total=len(current_data)):
            # 1. Feature Engineering
            emb = embedder.encode(row['text'])
            lang_vec = np.zeros(NUM_LANGS)
            if row['lang'] in LANG_MAP: lang_vec[LANG_MAP[row['lang']]] = 1
            context = torch.tensor(np.concatenate([emb, lang_vec])).float()

            # 2. Selection (Epsilon-Greedy)
            if random.random() < current_epsilon:
                action_idx = random.randint(0, len(MODEL_NAMES) - 1)
            else:
                with torch.no_grad():
                    q_values = router_net(context.unsqueeze(0)) # Add batch dim
                    action_idx = torch.argmax(q_values).item()

            # 3. Execution (LLM Inference)
            pred, raw = llm_manager.predict(action_idx, row['text'], row['lang'])
            reward = llm_manager.compute_reward_optimized(pred, row['label'],category_to_index, category_embeddings, alpha=0.7)

            # 4. Store in Replay Buffer
            memory.push(context, action_idx, reward)

            # 5. Experience Replay Training Step
            if len(memory) >= BATCH_SIZE:
                # Sample a mini-batch
                states, actions, rewards = memory.sample(BATCH_SIZE)
                
                # Predict current Q-values
                predicted_q = router_net(states)
                
                # We only want to update the Q-value for the action taken
                target_q = predicted_q.clone().detach()
                for batch_i in range(BATCH_SIZE):
                    target_q[batch_i, actions[batch_i]] = rewards[batch_i]
                
                # Optimize
                loss = criterion(predicted_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log results
            if i % 10 == 0: # Reduce disk I/O frequency
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([r+1, row['text'], row['lang'], row['label'], MODEL_NAMES[action_idx], pred, reward])

    torch.save(router_net.state_dict(), "neural_bandit_buffer_embedding.pth")
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()