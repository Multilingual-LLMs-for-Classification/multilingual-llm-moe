import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import random
import os
import csv
import re

# ------------------------------
# Language-specific prompts
# ------------------------------

PROMPTS = {
    "en": (
        "You are a strict rating classifier. Read the English review and output ONLY a single "
        "integer from 1 to 5. No words. No explanations. No punctuation.\n"
        "Review: \"{text}\"\n"
        "Output (1–5 only):"
    ),

    "fr": (
        "Vous êtes un classificateur strict. Lisez l’avis en français et retournez UNIQUEMENT "
        "un entier entre 1 et 5. Aucun mot. Aucune explication. Aucune ponctuation.\n"
        "Avis : \"{text}\"\n"
        "Sortie (1–5 uniquement) :"
    ),

    "es": (
        "Eres un clasificador estricto. Lee la reseña en español y devuelve SOLO un número "
        "entero del 1 al 5. Sin palabras. Sin explicaciones. Sin puntuación.\n"
        "Reseña: \"{text}\"\n"
        "Salida (1–5 solamente):"
    ),

    "de": (
        "Du bist ein strikter Klassifikator. Lies die deutsche Bewertung und gib NUR eine "
        "einzige Zahl von 1 bis 5 zurück. Keine Wörter. Keine Erklärungen. Keine Interpunktion.\n"
        "Bewertung: \"{text}\"\n"
        "Ausgabe (nur 1–5):"
    ),

    "zh": (
        "你是一个严格的分类器。阅读这条中文评价，并且只输出一个 1 到 5 的整数。"
        "不要输出任何文字、解释或标点符号。\n"
        "评论: \"{text}\"\n"
        "输出（只能是 1–5）:"
    ),

    "ja": (
        "あなたは厳密な分類器です。日本語レビューを読み、1〜5 の整数を1つだけ返してください。"
        "言葉・説明・句読点は禁止です。\n"
        "レビュー: \"{text}\"\n"
        "出力（1〜5 のみ）:"
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
    def __init__(self, state_dim, num_llms, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.policy = PPOPolicy(state_dim, num_llms)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

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
            critic_loss = (returns - values.squeeze()) ** 2
            critic_loss = critic_loss.mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# ------------------------------
# Hugging Face LLM System
# ------------------------------

class HuggingFaceLLMSystem:
    def __init__(self, model_names, prompts):
        self.model_names = model_names
        self.prompts = prompts
        self.models = [None for _ in model_names]
        self.tokenizers = [None for _ in model_names]

    def extract_rating(self, decoded):
        matches = re.findall(r'\b([1-5])\b', decoded)
        if not matches:
            return 3
        return int(matches[-1])

    def run(self, llm_id, text, language="en"):
        llm_id = min(llm_id, len(self.model_names)-1)

        if self.models[llm_id] is None:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_names[llm_id], trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_names[llm_id],
                device_map="auto",
                quantization_config=bnb_cfg,
                trust_remote_code=True
            )
            model.eval()
            self.models[llm_id] = model
            self.tokenizers[llm_id] = tokenizer
        else:
            model = self.models[llm_id]
            tokenizer = self.tokenizers[llm_id]

        prompt_template = self.prompts.get(language, self.prompts["en"])
        prompt = prompt_template.format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating = self.extract_rating(decoded)
        print(rating)

        return rating, decoded, prompt, self.model_names[llm_id]


# ------------------------------
# Routing Pipeline
# ------------------------------

class RoutingPipeline:
    def __init__(self, agent, llm_system, state_dim):
        self.agent = agent
        self.llm_system = llm_system
        self.state_dim = state_dim

    def get_state(self, text, language):
        lang_map = {"en":0, "fr":1, "ja":2, "es":3, "zh":4, "de":5}
        lang_vec = np.zeros(6)
        if language in lang_map:
            lang_vec[lang_map[language]] = 1
        return np.concatenate([np.random.randn(self.state_dim-6), lang_vec])

    def reward_fn(self, pred_rating, true_rating):
        return 1 - abs(pred_rating - true_rating)/4

    def train_step(self, batch, output_csv=None):
        states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []

        for _, row in batch.iterrows():
            state = self.get_state(row['review_body'], row['language'])
            action, log_prob, value = self.agent.select_action(state)

            pred_rating, decoded_output, prompt_used, llm_name_used = \
                self.llm_system.run(action, row['review_body'], row['language'])
            print("Actual Star " ,int(row['stars']))
            reward = self.reward_fn(pred_rating, int(row['stars']))

            if output_csv is not None:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        row['review_body'],
                        row['language'],
                        row['stars'],
                        action,
                        pred_rating,
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
        actions = torch.tensor(actions)
        old_log_probs = torch.stack(old_log_probs).detach()

        returns, advantages = self.agent.compute_returns(rewards, values, dones)
        self.agent.update(states, actions, old_log_probs, returns, advantages)

    def run_single(self, row):
        """
        Run inference for a single review without updating the agent.
        `row` should be a dict with 'review_body', 'language', and 'stars'
        """
        # 1. Prepare state (you may already have a function for this)
        state = self.prepare_state(row)  # returns a tensor of shape [1, state_dim]

        # 2. Get action probabilities from agent (no gradient)
        with torch.no_grad():
            probs, _ = self.agent.policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()  # chosen LLM index

        # 3. Run the chosen LLM
        prompt = self.llm_system.get_prompt(row['language'], row['review_body'])
        decoded_output = self.llm_system.generate(action, prompt)

        # 4. Convert decoded output to predicted rating (depends on your logic)
        pred_rating = self.llm_system.extract_rating(decoded_output)

        return {
            "action": action,
            "pred_rating": pred_rating,
            "decoded_output": decoded_output,
            "prompt": prompt,
            "llm_name_used": self.llm_system.model_names[action]
        }


# ------------------------------
# Main training
# ------------------------------

if __name__ == "__main__":
    state_dim = 20
    
    batch_size = 4

    model_names = [
        "Qwen/Qwen2.5-3B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "CohereLabs/aya-23-8B"
    ]

    num_llms = len(model_names)

    llm_system = HuggingFaceLLMSystem(model_names, PROMPTS)
    agent = PPOAgent(state_dim, num_llms)
    pipeline = RoutingPipeline(agent, llm_system, state_dim)

    output_csv_path = "llm_outputs.csv"
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "review_body","language","true_stars",
                "action","pred_rating","decoded_output",
                "prompt","llm_name_used"
            ])

    data = pd.read_csv("../fyp_multilingual_Text_classification/data/train_subset_final.csv",
                       dtype={'review_body': str, 'language': str, 'stars': int},
                       encoding='utf-8',
                       on_bad_lines='skip')

    for step in range(100):
        batch = data.sample(batch_size)
        pipeline.train_step(batch, output_csv=output_csv_path)
        if step % 10 == 0:
            print(f"Step {step} completed")

    torch.save(agent.policy.state_dict(), "ppo_router_policy_multilang.pth")
    print("Router policy saved successfully.")
