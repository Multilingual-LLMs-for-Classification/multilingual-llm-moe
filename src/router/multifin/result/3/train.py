from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import csv
import re

# ------------------------------
# News Classification Categories
# ------------------------------

NEWS_CATEGORIES = [
    "Technology",
    "Industry",
    "Tax & Accounting",
    "Finance",
    "Government & Controls",
    "Business & Management"
]

HF_TOKEN = os.getenv("HF_TOKEN")
TOKEN = "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"
 # Alias

# ------------------------------
# Language-specific prompts
# ------------------------------

PROMPTS = {
    "Dani": (
        "Du er en streng nyhedsklassifikator. Læs den danske artikel og udskriv KUN én "
        "kategori fra denne liste: {categories}. **Output KUN kategorien på ENGELSK. "
        "Intet andet. Ingen ord. Ingen forklaringer. Ingen tegnsætning.**\n"
        "Artikel: \"{text}\"\n"
        "Output (Kun kategori):"
    ),
    "English": (
        "You are a strict news classifier. Read the English article and output ONLY a single "
        "category from this list: {categories}. No words. No explanations. No punctuation.\n"
        "Article: \"{text}\"\n"
        "Output (Category only):"
    ),
    "Span": (
        "Eres un clasificador estricto de noticias. Lee el artículo en español y devuelve SOLO una "
        "categoría de esta lista: {categories}. **Devuelve SOLAMENTE la categoría en INGLÉS. "
        "Nada más. Sin palabras. Sin explicaciones. Sin puntuación.**\n"
        "Artículo: \"{text}\"\n"
        "Salida (Categoría solamente):"
    ),
    "Pol": (
        "Jesteś rygorystycznym klasyfikatorem wiadomości. Przeczytaj polski artykuł i zwróć TYLKO jedną "
        "kategorię z tej listy: {categories}. **Zwróć TYLKO kategorię w JĘZYKU ANGIELSKIM. "
        "Nic więcej. Bez słów. Bez wyjaśnień. Bez znaków interpunkcyjnych.**\n"
        "Artykuł: \"{text}\"\n"
        "Wyjście (Tylko kategoria):"
    ),
    "Turk": (
        "Sen katı bir haber sınıflandırıcısısın. Türkçe makaleyi oku ve SADECE bu listeden tek bir "
        "kategori çıktı: {categories}. **SADECE kategori adını İNGİLİZCE olarak çıktı. "
        "Başka hiçbir şey. Kelime yok. Açıklama yok. Noktalama işareti yok.**\n"
        "Makale: \"{text}\"\n"
        "Çıktı (Sadece Kategori):"
    )
}

# ------------------------------
# PPO Actor-Critic Network
# ------------------------------

class PPOPolicy(nn.Module):
    def __init__(self, state_dim, num_llms, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_llms),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

# ------------------------------
# PPO Agent
# ------------------------------

class PPOAgent:
    def __init__(self, state_dim, num_llms, lr=4e-4, gamma=0.99, clip_eps=0.1, entropy_weight=0.01):
        self.policy = PPOPolicy(state_dim, num_llms)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_weight = entropy_weight

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs, value = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return min(action.item(), probs.shape[-1]-1), dist.log_prob(action), value

    def compute_returns(self, rewards, values, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(values).squeeze().detach()
        advantages = returns - values
        advantages = advantages.detach()
        return returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages, epochs=5):
        for _ in range(epochs):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = ((returns - values.squeeze()) ** 2).mean()
            entropy = dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - self.entropy_weight * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# ------------------------------
# Hugging Face LLM System
# ------------------------------

class HuggingFaceLLMSystem:
    def __init__(self, model_names, prompts, categories):
        self.model_names = model_names
        self.prompts = prompts
        self.categories = categories
        self.models = [None for _ in model_names]
        self.tokenizers = [None for _ in model_names]

    def get_categories_string(self):
        return ", ".join(self.categories)

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

    def run(self, llm_id, text, language):
        llm_id = min(llm_id, len(self.model_names)-1)
        if self.models[llm_id] is None:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_names[llm_id], trust_remote_code=True, use_auth_token=TOKEN)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_names[llm_id],
                device_map="auto",
                quantization_config=bnb_cfg,
                trust_remote_code=True,
                use_auth_token=TOKEN
            )
            model.eval()
            self.models[llm_id] = model
            self.tokenizers[llm_id] = tokenizer
        else:
            model = self.models[llm_id]
            tokenizer = self.tokenizers[llm_id]

        prompt_template = self.prompts.get(language, self.prompts["English"])
        categories_str = self.get_categories_string()
        prompt = prompt_template.format(text=text, categories=categories_str)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_output, predicted_label = self.extract_label(decoded)
        print(f"Predicted Label: {predicted_label}")
        return predicted_label, decoded, prompt, self.model_names[llm_id], cleaned_output

# ------------------------------
# Routing Pipeline (Multi-model, Language-aware rewards)
# ------------------------------

class RoutingPipeline:
    def __init__(self, agent, llm_system, state_dim):
        self.agent = agent
        self.llm_system = llm_system
        self.state_dim = state_dim
        # Nested dict: lang_history[model_id][language] = deque(maxlen=100)
        self.lang_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))

    def get_state(self, text, language):
        lang_map = {"English":0, "Turkish":1, "Danish":2, "Spanish":3, "Polish":4}
        lang_vec = np.zeros(6)
        lang_key = language
        if language in {"Dani", "Danish"}: lang_key = "Danish"
        elif language in {"Turk", "Turkish"}: lang_key = "Turkish"
        elif language in {"Span", "Spanish"}: lang_key = "Spanish"
        elif language in {"Pol", "Polish"}: lang_key = "Polish"
        elif language in {"English", "en"}: lang_key = "English"
        if lang_key in lang_map:
            lang_vec[lang_map[lang_key]] = 1
        text_len_log = np.log(len(text) + 1)
        padding_size = self.state_dim - 7
        padding = np.zeros(padding_size)
        return np.concatenate([np.array([text_len_log]), padding, lang_vec])

    def reward_fn(self, model_id, pred_label, true_label, lang):
        per_sample = 1.0 if pred_label == true_label else 0.0
        self.lang_history[model_id][lang].append(per_sample)
        lang_accuracy = sum(self.lang_history[model_id][lang]) / len(self.lang_history[model_id][lang])
        final_reward = 0.5 * per_sample + 0.5 * lang_accuracy
        return final_reward, lang_accuracy

    def train_step(self, batch, output_csv=None):
        states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []

        for _, row in batch.iterrows():
            state = self.get_state(row['text'], row['lang'])
            action, log_prob, value = self.agent.select_action(state)

            lang_key_for_prompt = row['lang']  # 'Dani', 'English', etc.
            
            pred_label, decoded_output, prompt_used, llm_name_used, cleaned_output = \
                self.llm_system.run(action, row['text'], lang_key_for_prompt)
            
            true_label = row['label']

            reward, lang_acc = self.reward_fn(action, pred_label, true_label, row['lang'])

            if output_csv is not None:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        row['text'],
                        row['lang'],
                        true_label,
                        cleaned_output,
                        action,
                        pred_label,
                        reward,
                        lang_acc,
                        decoded_output,
                        prompt_used,
                        llm_name_used
                    ])

            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(False)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.stack(old_log_probs).detach()

        returns, advantages = self.agent.compute_returns(rewards, values, dones)
        self.agent.update(states, actions, old_log_probs, returns, advantages)

# ------------------------------
# Main training
# ------------------------------

if __name__ == "__main__":
    state_dim = 20
    batch_size = 4

    model_names = [
        "google/gemma-7b",
        "CohereLabs/aya-expanse-8b",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-3B"
    ]

    num_llms = len(model_names)

    llm_system = HuggingFaceLLMSystem(model_names, PROMPTS, NEWS_CATEGORIES)
    agent = PPOAgent(state_dim, num_llms)
    pipeline = RoutingPipeline(agent, llm_system, state_dim)

    output_csv_path = "llm_classification_outputs.csv"
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "article_text","language","true_category","cleaned_output",
                "action","pred_category","reward","lang_accuracy",
                "decoded_output","prompt","llm_name_used"
            ])

    try:
        data = pd.read_csv("../../data/train.csv",
                           dtype={'text': str, 'lang': str, 'label': str},
                           encoding='utf-8',
                           on_bad_lines='skip')
    except FileNotFoundError:
        print("Error: Training data file not found.")
        exit(1)

    available_langs = set(data['lang'].unique())
    unsupported_langs = available_langs - set(PROMPTS.keys())
    if unsupported_langs:
        print(f"Warning: Unsupported languages in dataset: {unsupported_langs}")

    ROUNDS = 2
    total_steps_per_round = len(data) // batch_size
    print(f"Dataset size: {len(data)}. Batch size: {batch_size}. Steps per round: {total_steps_per_round}.")

    for round in range(ROUNDS):
        print(f"=== ROUND {round+1} / {ROUNDS} ===")
        shuffled_data = data.sample(frac=1).reset_index(drop=True)

        for step in range(total_steps_per_round):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size
            batch = shuffled_data.iloc[start_idx:end_idx]
            pipeline.train_step(batch, output_csv=output_csv_path)

            if step % 25 == 0 and step != 0:
                print(f"Step {step}/{total_steps_per_round} completed")

        torch.save(agent.policy.state_dict(), f"ppo_router_policy_news_round{round+1}.pth")
        print(f"Checkpoint saved after round {round+1}")

    print("Training complete.")
