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
# 1. Configuration
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
UCB_C = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
BUFFER_SIZE = 1000
ROUNDS = 5
LOG_EVERY = 10
OUTPUT_CSV = "neural_bandit_training_langvec5.csv"

# ==============================
# 2. XLM-R Embedder
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
# 3. Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ==============================
# 4. Q-Network (No Sigmoid)
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
# 5. LLM Manager
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

    def compute_reward_optimized(
        self, pred, true, cat2idx, cat_embs, alpha=0.7
    ):
        acc = 1.0 if pred == true else 0.0

        if pred not in cat2idx or true not in cat2idx:
            sim = 0.0
        else:
            p = cat_embs[cat2idx[pred]].reshape(1, -1)
            t = cat_embs[cat2idx[true]].reshape(1, -1)
            sim = (cosine_similarity(p, t)[0][0] + 1) / 2

        return alpha * acc + (1 - alpha) * sim

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
        return prediction, decoded

# ==============================
# 6. Training Loop
# ==============================
def lang_to_vec(lang):
    vec = np.zeros(NUM_LANGS, dtype=np.float32)
    if lang in LANG_MAP:
        vec[LANG_MAP[lang]] = 1.0
    return vec

def main():
    data = pd.read_csv("../../data/train.csv")

    embedder = XLMREmbedder()
    llm = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)

    router = NeuralRouter(STATE_DIM, len(MODEL_NAMES))
    optimizer = optim.Adam(router.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(BUFFER_SIZE)

    cat2idx = {c: i for i, c in enumerate(NEWS_CATEGORIES)}
    cat_embs = np.stack([embedder.encode(c) for c in NEWS_CATEGORIES])

    action_counts = np.zeros(len(MODEL_NAMES))
    total_steps = 0

    # CSV Header
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round", "text", "lang", "true_label",
            "chosen_model", "prediction", "reward", "q_values"
        ])

    print("🚀 Training started...")

    for r in range(ROUNDS):
        data = data.sample(frac=1).reset_index(drop=True)

        for _, row in tqdm(data.iterrows(), total=len(data)):
            total_steps += 1

            emb = embedder.encode(row["text"])
            lang_vec = lang_to_vec(row["lang"])
            state = np.concatenate([emb, lang_vec])
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get Q-values
            with torch.no_grad():
                q_vals = router(state_t).cpu().numpy()[0]

            # UCB exploration
            ucb_scores = q_vals + UCB_C * np.sqrt(
                np.log(total_steps + 1) / (action_counts + 1e-5)
            )

            action = int(np.argmax(ucb_scores))
            action_counts[action] += 1

            # Predict & compute reward
            pred, _ = llm.predict(action, row["text"])
            reward = llm.compute_reward_optimized(pred, row["label"], cat2idx, cat_embs)

            # Add to replay buffer
            memory.push(torch.tensor(state, dtype=torch.float32), action, reward)

            # Train router
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards = memory.sample(BATCH_SIZE)

                q_pred = router(states)
                q_target = q_pred.detach().clone()

                for i in range(BATCH_SIZE):
                    a = actions[i]
                    q_target[i, a] = 0.9 * q_pred[i, a] + 0.1 * rewards[i]

                loss = criterion(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # CSV logging
            if total_steps % LOG_EVERY == 0:
                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        r + 1,
                        row["text"],
                        row["lang"],
                        row["label"],
                        MODEL_NAMES[action],
                        pred,
                        round(float(reward), 4),
                        q_vals.tolist()
                    ])

    torch.save(router.state_dict(), "neural_router_langvec5.pth")
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    main()
