import os
import re
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ------------------------------
# Language-Specific Prompts
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
# mBERT STATE ENCODER
# ------------------------------
class MBertStateEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text):
        # handle empty text
        if not isinstance(text, str) or text.strip() == "":
            text = "[EMPTY]"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        outputs = self.model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return sentence_embedding.cpu().numpy()  # (768,)

# ------------------------------
# PPO Actor-Critic Network (language-conditional actor)
# ------------------------------
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, num_llms, num_langs, hidden=256):
        """
        Actor outputs logits of shape (num_langs * num_llms).
        For a sample with language index L, we take the slice logits[L*num_llms : (L+1)*num_llms]
        and softmax it to get probabilities for that language.
        """
        super().__init__()
        self.num_llms = num_llms
        self.num_langs = num_langs

        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        # produce logits for all language-LLM combos
        self.actor_head = nn.Linear(hidden, num_langs * num_llms)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, lang_ids):
        """
        state: tensor [B, state_dim]
        lang_ids: tensor [B] containing language indices (0..num_langs-1)
        returns:
          probs: [B, num_llms] (probabilities for the language of each sample)
          values: [B, 1]
        """
        device = state.device
        B = state.shape[0]
        hidden = self.base(state)  # [B, hidden]
        logits_all = self.actor_head(hidden)  # [B, num_langs * num_llms]
        # reshape to [B, num_langs, num_llms]
        logits_all = logits_all.view(B, self.num_langs, self.num_llms)

        # pick the language-specific logits per sample using advanced indexing
        # lang_ids should be a long tensor
        lang_ids = lang_ids.long().to(device)  # [B]
        batch_idx = torch.arange(B, device=device)
        lang_logits = logits_all[batch_idx, lang_ids, :]  # [B, num_llms]
        probs = torch.softmax(lang_logits, dim=-1)  # [B, num_llms]

        values = self.critic(state)  # [B, 1]
        return probs, values

# ------------------------------
# PPO Agent
# ------------------------------
class PPOAgent:
    def __init__(self, state_dim, num_llms, num_langs, lr=3e-4, gamma=0.99, clip_eps=0.2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(state_dim, num_llms, num_langs).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def select_action(self, state, lang_id):
        """
        state: numpy array shape (state_dim,) or torch tensor (state_dim,)
        lang_id: int
        returns: action (int), log_prob (tensor scalar), value (tensor scalar detached)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device).unsqueeze(0)  # [1, state_dim]
        lang_ids = torch.tensor([lang_id], dtype=torch.long, device=self.device)  # [1]

        with torch.no_grad():
            probs, value = self.policy(state, lang_ids)  # probs: [1, num_llms], value: [1,1]
        probs = probs.squeeze(0)  # [num_llms]
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # scalar tensor on device
        return action.item(), log_prob.detach(), value.detach()  # detach to avoid graph retention

    def compute_returns(self, rewards, values, dones):
        """
        rewards: list of floats
        values: list of torch tensors (each shape [1,1] or [1])
        dones: list of bool
        returns: tensor [T], advantages: tensor [T]
        """
        returns = []
        R = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # values were detached tensors; stack and squeeze
        values_t = torch.cat(values).squeeze().to(self.device)  # [T]
        advantages = returns - values_t
        return returns, advantages.detach()

    def update(self, states, lang_ids, actions, old_log_probs, returns, advantages, epochs=5):
        """
        states: tensor [B, state_dim] on device
        lang_ids: tensor [B] on device (long)
        actions: tensor [B] (long) on device
        old_log_probs: tensor [B] (float) on device
        returns: tensor [B] on device
        advantages: tensor [B] on device
        """
        for _ in range(epochs):
            probs, values = self.policy(states, lang_ids)  # probs: [B, num_llms], values: [B,1]
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)  # [B]
            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

            # actor loss
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            # critic loss (MSE)
            critic_loss = ((returns - values.squeeze()) ** 2).mean()
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
        self.models = [None] * len(model_names)
        self.tokenizers = [None] * len(model_names)

    def extract_rating(self, decoded):
        matches = re.findall(r'\b([1-5])\b', decoded)
        return int(matches[-1]) if matches else 3

    def run(self, llm_id, text, language="en"):
        llm_id = min(llm_id, len(self.model_names) - 1)

        if self.models[llm_id] is None:
            # quantization config (if your environment supports it)
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_names[llm_id])
            model = AutoModelForCausalLM.from_pretrained(
                self.model_names[llm_id],
                device_map="auto",
                quantization_config=bnb_cfg
            )
            model.eval()
            self.models[llm_id] = model
            self.tokenizers[llm_id] = tokenizer

        tokenizer = self.tokenizers[llm_id]
        model = self.models[llm_id]

        prompt = self.prompts.get(language, self.prompts["en"]).format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating = self.extract_rating(decoded)
        return rating, decoded, prompt, self.model_names[llm_id]

# ------------------------------
# Routing Pipeline With mBERT State
# ------------------------------
class RoutingPipeline:
    def __init__(self, agent, llm_system, state_dim, lang_map):
        self.agent = agent
        self.llm_system = llm_system
        self.state_dim = state_dim
        self.encoder = MBertStateEncoder(device=("cuda" if torch.cuda.is_available() else "cpu"))
        self.lang_map = lang_map  # dict language -> index

    def get_state(self, text, language):
        text_embedding = self.encoder.encode(text)  # (768,)
        lang_vec = np.zeros(len(self.lang_map))
        if language in self.lang_map:
            lang_vec[self.lang_map[language]] = 1
        return np.concatenate([text_embedding, lang_vec])

    def reward_fn(self, pred, true):
        # simple normalized reward in [0,1]
        return 1.0 - abs(pred - true) / 4.0

    def train_step(self, batch, output_csv=None):
        states, lang_ids, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], [], []

        for _, row in batch.iterrows():
            text = row['review_body']
            language = row['language']
            true_stars = int(row['stars'])
            state = self.get_state(text, language)  # numpy array

            # language id for this sample
            lang_id = self.lang_map.get(language, 0)

            # select action conditioned on language id
            action, log_prob, value = self.agent.select_action(state, lang_id)

            # run only the selected LLM
            pred_rating, decoded, prompt_used, llm_name = self.llm_system.run(
                action, text, language
            )

            reward = self.reward_fn(pred_rating, true_stars)

            # optionally log
            if output_csv:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        text, language, true_stars,
                        action, pred_rating, decoded, prompt_used, llm_name
                    ])

            # store experience (note: log_prob and value are detached in select_action)
            states.append(state)
            lang_ids.append(lang_id)
            actions.append(action)
            old_log_probs.append(log_prob)  # already detached tensor
            rewards.append(reward)
            values.append(value)  # detached tensor
            dones.append(False)

        # convert to tensors on agent device
        device = self.agent.device
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.stack(old_log_probs).to(device)  # shape [B]
        # values is list of detached tensors shape [1,1] each; compute_returns will cat them
        returns, advantages = self.agent.compute_returns(rewards, values, dones)
        returns = returns.to(device)
        advantages = advantages.to(device)

        # run PPO update (policy.forward will pick language slice internally)
        self.agent.update(states, lang_ids, actions, old_log_probs, returns, advantages)

# ------------------------------
# MAIN TRAINING LOOP
# ------------------------------
if __name__ == "__main__":
    # configuration
    LANG_MAP = {"en": 0, "fr": 1, "ja": 2, "es": 3, "zh": 4, "de": 5}
    NUM_LANGS = len(LANG_MAP)
    STATE_DIM = 768 + NUM_LANGS
    BATCH_SIZE = 4

    model_names = [
        "Qwen/Qwen2.5-3B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "CohereLabs/aya-23-8B"
    ]
    NUM_LLMS = len(model_names)

    llm_system = HuggingFaceLLMSystem(model_names, PROMPTS)
    agent = PPOAgent(state_dim=STATE_DIM, num_llms=NUM_LLMS, num_langs=NUM_LANGS)
    pipeline = RoutingPipeline(agent, llm_system, STATE_DIM, LANG_MAP)

    output_csv_path = "llm_outputs.csv"
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "review_body", "language", "true_stars",
                "action", "pred_rating", "decoded_output",
                "prompt", "llm_name_used"
            ])

    # update the path to your dataset as required
    data = pd.read_csv(
        "../../../../../../data/Amazon/train_subset_final.csv",
        dtype={'review_body': str, 'language': str, 'stars': int},
        encoding='utf-8',
        on_bad_lines='skip'
    )

    ROUNDS = 5
    STEPS_PER_ROUND = 100

    for r in range(ROUNDS):
        print(f"\n=== ROUND {r+1} ===")
        for step in range(STEPS_PER_ROUND):
            batch = data.sample(BATCH_SIZE)
            pipeline.train_step(batch, output_csv_path)
            if step % 10 == 0:
                print(f"Step {step} done")

        torch.save(agent.policy.state_dict(), f"ppo_router_langcond_round{r+1}.pth")
        print(f"Checkpoint saved (Round {r+1})")
