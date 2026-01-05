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
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

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

EMBED_DIM = 768   # 🔴 Changed from 384 → 768 (XLM-R)
STATE_DIM = EMBED_DIM + NUM_LANGS

HF_TOKEN = os.getenv("HF_TOKEN") or "YOUR_TOKEN_HERE"
EPSILON = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
BUFFER_SIZE = 1000

# ==============================
# 2. XLM-R Embedder (ONLY ADDITION)
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
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        return pooled.squeeze(0).cpu().numpy()

# ==============================
# 3. Replay Buffer & Router
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
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_models),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# ==============================
# 4. LLM Manager (UNCHANGED)
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
            torch.cuda.empty_cache()
            gc.collect()
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                device_map="auto",
                quantization_config=bnb,
                token=HF_TOKEN
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

        cleaned = raw_output.lower().replace('"', '').replace("'", '').strip()
        for cat in sorted(self.categories, key=len, reverse=True):
            if cat.lower() == cleaned or cat.lower() in cleaned:
                return cat, cat

        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned)
        return cleaned_text, "Unknown"

    def compute_reward_optimized(self, pred_label, true_label,
                                  category_to_index, category_embeddings, alpha=0.7):
        acc_reward = 1.0 if pred_label == true_label else 0.0

        if pred_label not in category_to_index or true_label not in category_to_index:
            sim_reward = 0.0
        else:
            p = category_embeddings[category_to_index[pred_label]].reshape(1, -1)
            t = category_embeddings[category_to_index[true_label]].reshape(1, -1)
            sim = cosine_similarity(p, t)[0][0]
            sim_reward = (sim + 1) / 2

        return alpha * acc_reward + (1 - alpha) * sim_reward

    def predict(self, model_idx, text, lang):
        name = self.model_names[model_idx]
        self.load_model(name)

        prompt = (
            f"You are a strict news classifier. Read the article headline and "
            f"output ONLY a single category from this list: {self.categories}. "
            f"No words. No explanations. No punctuation. "
            f"Article: {text} Output (Category only):"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            tokens = self.model.generate(**inputs, max_new_tokens=15)

        decoded = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        _, prediction = self.extract_label(decoded)
        return prediction, decoded

# ==============================
# 5. Main Training Script
# ==============================
def main():
    data_path = "../../data/train.csv"
    if not os.path.exists(data_path):
        return

    original_data = pd.read_csv(data_path)

    embedder = XLMREmbedder()  # 🔴 Only embedding change
    llm_manager = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)

    router_net = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    optimizer = optim.Adam(router_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(BUFFER_SIZE)

    ROUNDS = 2
    EPSILON_DECAY = 0.8

    category_embeddings = np.vstack([
        embedder.encode(cat) for cat in NEWS_CATEGORIES
    ])
    category_to_index = {cat: i for i, cat in enumerate(NEWS_CATEGORIES)}

    output_csv = "neural_bandit_train_results_buffer4.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["round", "text", "lang", "label", "model_chosen", "pred", "reward"]
        )

    print("Starting Training with Replay Buffer...")

    for r in range(ROUNDS):
        current_data = original_data.sample(frac=1).reset_index(drop=True)
        current_epsilon = EPSILON * (EPSILON_DECAY ** r)

        for i, row in tqdm(current_data.iterrows(), total=len(current_data)):
            emb = embedder.encode(row["text"])
            lang_vec = np.zeros(NUM_LANGS)
            if row["lang"] in LANG_MAP:
                lang_vec[LANG_MAP[row["lang"]]] = 1

            context = torch.tensor(
                np.concatenate([emb, lang_vec])
            ).float()

            if random.random() < current_epsilon:
                action_idx = random.randint(0, len(MODEL_NAMES) - 1)
            else:
                with torch.no_grad():
                    q_vals = router_net(context.unsqueeze(0))
                    action_idx = torch.argmax(q_vals).item()

            pred, _ = llm_manager.predict(action_idx, row["text"], row["lang"])
            reward = 1.0 if pred == row['label'] else 0.0

            memory.push(context, action_idx, reward)

            if len(memory) >= BATCH_SIZE:
                states, actions, rewards = memory.sample(BATCH_SIZE)
                predicted_q = router_net(states)
                target_q = predicted_q.clone().detach()

                for b in range(BATCH_SIZE):
                    target_q[b, actions[b]] = rewards[b]

                loss = criterion(predicted_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 10 == 0:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        r + 1,
                        row["text"],
                        row["lang"],
                        row["label"],
                        MODEL_NAMES[action_idx],
                        pred,
                        reward
                    ])

    torch.save(router_net.state_dict(), "neural_bandit_buffer_embedding4.pth")
    print("Training Complete.")

if __name__ == "__main__":
    main()
