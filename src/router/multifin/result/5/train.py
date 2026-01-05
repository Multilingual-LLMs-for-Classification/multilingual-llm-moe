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
import random
import math

# ------------------------------
# Config / Hyperparameters
# ------------------------------
NEWS_CATEGORIES = [
    "Technology",
    "Industry",
    "Tax & Accounting",
    "Finance",
    "Government & Controls",
    "Business & Management"
]

# The dataset languages (canonical names used for one-hot and tracking)
LANGUAGES = ["English", "Danish", "Spanish", "Polish", "Turkish"]

HF_TOKEN = os.getenv("HF_TOKEN")
TOKEN = HF_TOKEN  # alias for huggingface token if needed

# Use the same English prompt for every sample (as you requested)
ENGLISH_PROMPT_TEMPLATE = (
    "You are a strict news classifier. Read the following article and output ONLY a single "
    "category from this list: {categories}. Output ONLY the single category name (in English). "
    "No explanations, no punctuation.\n\nArticle: \"{text}\"\nOutput (Category only):"
)

# PPO / training hyperparams
STATE_DIM = 20                 # full state dimension (will include language one-hot at front)
BATCH_SIZE = 4                 # training samples pulled per mini-batch iteration (data loading)
UPDATE_BATCH_SIZE = 256        # number of transitions to accumulate before an agent update
ROUNDS = 10                    # number of passes over dataset (increase for longer training)
GAMMA = 0.99
LR = 4e-4
CLIP_EPS = 0.1
ENTROPY_WEIGHT = 0.01
PPO_EPOCHS = 5                 # inner PPO epochs per update
MAX_LANG_HISTORY = 100       # deque length for rolling language accuracy
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------
# Helpers
# ------------------------------
def normalize_lang(lang_raw: str) -> str:
    """Map dataset language strings to canonical LANGUAGES entries."""
    if not isinstance(lang_raw, str):
        return "English"
    k = lang_raw.strip().lower()
    mapping = {
        "en": "English", "english": "English",
        "da": "Danish", "dani": "Danish", "danish": "Danish",
        "es": "Spanish", "span": "Spanish", "spanish": "Spanish",
        "pl": "Polish", "pol": "Polish", "polish": "Polish",
        "tr": "Turkish", "turk": "Turkish", "turkish": "Turkish",
    }
    return mapping.get(k, lang_raw if lang_raw in LANGUAGES else "English")

def category_match(pred: str, gold: str) -> bool:
    """Robust category equality check using canonical list."""
    if not isinstance(pred, str) or not isinstance(gold, str):
        return False
    p = pred.strip().lower()
    g = gold.strip().lower()
    # Try direct match on canonical categories
    for cat in NEWS_CATEGORIES:
        if cat.strip().lower() == p:
            p = cat.strip().lower()
        if cat.strip().lower() == g:
            g = cat.strip().lower()
    return p == g

# ------------------------------
# PPO Policy Network
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
    def __init__(self, state_dim, num_llms, lr=LR, gamma=GAMMA, clip_eps=CLIP_EPS, entropy_weight=ENTROPY_WEIGHT):
        self.policy = PPOPolicy(state_dim, num_llms)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_weight = entropy_weight

    def select_action(self, state_np):
        # state_np: numpy array
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
        probs, value = self.policy(state)  # probs: [1, num_actions], value: [1,1]
        probs = probs.squeeze(0)           # [num_actions]
        value = value.squeeze(0)           # [1]
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0).detach()

    def compute_returns(self, rewards, values, dones):
        # rewards: list of floats
        # values: list of torch tensors (detached) scalar
        returns = []
        R = 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze().detach()  # shape [T]
        advantages = returns - values
        return returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages, epochs=PPO_EPOCHS):
        # states: tensor [N, state_dim]
        # actions: tensor [N]
        # old_log_probs: tensor [N]
        # returns: tensor [N]
        # advantages: tensor [N]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
    def __init__(self, model_names, categories, prompt_template=ENGLISH_PROMPT_TEMPLATE):
        self.model_names = model_names
        self.categories = categories
        self.prompt_template = prompt_template
        self.models = [None for _ in model_names]
        self.tokenizers = [None for _ in model_names]

    def get_categories_string(self):
        return ", ".join(self.categories)

    def extract_label(self, decoded):
        # Extract the final predicted category from decoded text
        # Aggressive cleaning as before
        raw = decoded.strip()
        # take last non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        last = lines[-1] if lines else raw
        cleaned_output = last.lower().replace('"', '').replace("'", '').strip()
        # match canonical categories
        for category in self.categories:
            if category.lower() == cleaned_output or category.lower() in cleaned_output:
                return category, category
        # fallback cleaned
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_output)
        string_output = " ".join([t.strip() for t in cleaned_text.split()])
        return string_output, "Unknown"

    def run(self, llm_id, text, language):
        """
        Run the chosen model. This uses the SAME English prompt for all languages.
        Returns: predicted_label, decoded_output, prompt, model_name, cleaned_output
        """
        llm_id = min(llm_id, len(self.model_names)-1)
        # Lazy load model/tokenizer (be careful with large models)
        if self.models[llm_id] is None:
            try:
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
            except Exception as e:
                # If model loading fails (for quick testing), fallback to returns below
                print(f"Warning: failed to load model {self.model_names[llm_id]}: {e}")
                self.models[llm_id] = None
                self.tokenizers[llm_id] = None

        # Build the English prompt for this sample (same for all languages)
        prompt = self.prompt_template.format(text=text, categories=self.get_categories_string())

        # If model isn't loaded, we provide a random prediction to let the router train
        if self.models[llm_id] is None or self.tokenizers[llm_id] is None:
            # Random selection among canonical categories (for offline testing)
            pred = random.choice(self.categories)
            decoded = pred
            cleaned_output, predicted_label = self.extract_label(decoded)
            return predicted_label, decoded, prompt, self.model_names[llm_id] if llm_id < len(self.model_names) else "unknown", cleaned_output

        model = self.models[llm_id]
        tokenizer = self.tokenizers[llm_id]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_output, predicted_label = self.extract_label(decoded)
        return predicted_label, decoded, prompt, self.model_names[llm_id], cleaned_output

# ------------------------------
# Routing Pipeline (with per-model language history & buffer)
# ------------------------------
class RoutingPipeline:
    def __init__(self, agent, llm_system, state_dim):
        self.agent = agent
        self.llm_system = llm_system
        self.state_dim = state_dim
        # lang_history[model_id][language] = deque(maxlen=MAX_LANG_HISTORY)
        self.lang_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=MAX_LANG_HISTORY)))

        # Replay buffer for PPO updates
        self.buffer = {
            "states": [],
            "actions": [],
            "old_log_probs": [],
            "rewards": [],
            "values": [],
            "dones": []
        }

    def get_state(self, text, language):
        # Normalize language to canonical
        lang_key = normalize_lang(language)
        # one-hot for languages (ordered by LANGUAGES)
        lang_vec = np.zeros(len(LANGUAGES), dtype=np.float32)
        if lang_key in LANGUAGES:
            lang_vec[LANGUAGES.index(lang_key)] = 1.0
        # Feature: log length of text
        text_len_log = float(np.log(len(text) + 1))
        # Compose state: language one-hot first (strong signal), then text_len and padding
        core = np.concatenate([lang_vec, np.array([text_len_log], dtype=np.float32)])
        padding_size = max(0, self.state_dim - len(core))
        padding = np.zeros(padding_size, dtype=np.float32)
        state = np.concatenate([core, padding])
        return state

    def reward_fn(self, model_id, pred_label, true_label, lang):
        # per-sample reward: +1 correct, -1 incorrect
        per_sample = 1.0 if category_match(pred_label, true_label) else -1.0

        # update rolling history for this model & language
        canon_lang = normalize_lang(lang)
        self.lang_history[model_id][canon_lang].append(1.0 if per_sample > 0 else 0.0)

        # compute language accuracy fraction (0..1)
        lang_frac = sum(self.lang_history[model_id][canon_lang]) / len(self.lang_history[model_id][canon_lang])

        # scale language accuracy to [-1, +1]
        lang_scaled = (lang_frac * 2.0) - 1.0

        # combine: give more weight to per-sample (clear signal)
        final_reward = 0.7 * per_sample + 0.3 * lang_scaled

        # Clamp final reward to [-1, +1.0] to keep returns stable
        final_reward = max(-1.0, min(1.0, final_reward))

        return final_reward, lang_frac

    def store_transition(self, state, action, old_log_prob, reward, value, done=False):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["old_log_probs"].append(old_log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["values"].append(value)
        self.buffer["dones"].append(done)

    def flush_and_update(self):
        # Run PPO update if buffer non-empty
        if len(self.buffer["states"]) == 0:
            return
        # convert to tensors
        states = torch.tensor(np.array(self.buffer["states"], dtype=np.float32))
        actions = torch.tensor(self.buffer["actions"], dtype=torch.int64)
        old_log_probs = torch.stack(self.buffer["old_log_probs"]).detach()
        values = [v for v in self.buffer["values"]]  # list of torch scalars
        returns, advantages = self.agent.compute_returns(self.buffer["rewards"], values, self.buffer["dones"])
        # call update
        self.agent.update(states, actions, old_log_probs, returns, advantages)
        # clear buffer
        for k in self.buffer:
            self.buffer[k].clear()

    def train_step(self, batch_df, output_csv=None, round_idx=None, sample_start=None):
        # For each sample in the batch, select action, call model, compute reward, and store transition.
        for i,(_, row) in enumerate(batch_df.iterrows()):
            text = row['text']
            lang = row['lang']
            label = row['label']

            state = self.get_state(text, lang)
            # select action
            action, log_prob, value = self.agent.select_action(state)

            # Use English prompt for all languages
            pred_label, decoded_output, prompt_used, llm_name_used, cleaned_output = \
                self.llm_system.run(action, text, "English")

            reward, lang_frac = self.reward_fn(action, pred_label, label, lang)

            # store and optionally log
            self.store_transition(state, action, log_prob, reward, value, done=False)

            if output_csv is not None:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        text,
                        normalize_lang(lang),
                        label,
                        cleaned_output,
                        action,
                        pred_label,
                        float(reward),
                        float(lang_frac),
                        decoded_output,
                        prompt_used,
                        llm_name_used
                    ])

            global_sample_id = sample_start + i
            print(f"[Round {round_idx}] Sample {global_sample_id} processed")

            # If buffer reached update size, do PPO update
            if len(self.buffer["states"]) >= UPDATE_BATCH_SIZE:
                self.flush_and_update()

# ------------------------------
# Training entrypoint
# ------------------------------
if __name__ == "__main__":

    # Models - user provided; keep as-is (may be large). You can replace with smaller model names for debugging.
    model_names = [
        "CohereLabs/aya-expanse-8b",
        "Qwen/Qwen2.5-3B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]

    num_llms = len(model_names)

    # init systems
    llm_system = HuggingFaceLLMSystem(model_names, NEWS_CATEGORIES, prompt_template=ENGLISH_PROMPT_TEMPLATE)
    agent = PPOAgent(STATE_DIM, num_llms, lr=LR, gamma=GAMMA, clip_eps=CLIP_EPS, entropy_weight=ENTROPY_WEIGHT)
    pipeline = RoutingPipeline(agent, llm_system, STATE_DIM)

    # output csv header
    output_csv_path = "llm_classification_outputs.csv"
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "article_text","language","true_category","cleaned_output",
                "action","pred_category","reward","lang_accuracy",
                "decoded_output","prompt","llm_name_used"
            ])

    # load data - you may change path
    try:
        data = pd.read_csv("../../data/train.csv",
                           dtype={'text': str, 'lang': str, 'label': str},
                           encoding='utf-8',
                           on_bad_lines='skip')
    except FileNotFoundError:
        print("Error: Training data file not found. Please update the path.")
        exit(1)

    # normalize dataset language column in-place so logs use canonical names
    data['lang'] = data['lang'].apply(normalize_lang)

    # Sanity check languages
    available_langs = set(data['lang'].unique())
    unsupported_langs = available_langs - set(LANGUAGES)

    if unsupported_langs:
        print(f"Warning: Dataset contains unexpected languages: {unsupported_langs}. They will be mapped to English.")

    # training loop
    for round_idx in range(ROUNDS):
        print(f"=== ROUND {round_idx+1}/{ROUNDS} ===")
        shuffled_data = data.sample(frac=1, random_state=round_idx).reset_index(drop=True)
        total_steps = len(shuffled_data) // BATCH_SIZE

        # per-round CSV summary file for readability (append to global CSV too)
        round_summary = []
        sample_counter = 0

        for step in range(total_steps):
            start = step * BATCH_SIZE
            end = (step + 1) * BATCH_SIZE
            batch = shuffled_data.iloc[start:end]
            pipeline.train_step(batch, output_csv=output_csv_path,
                            round_idx=round_idx+1,   # pass round
                            sample_start=sample_counter)

            if (step + 1) % 50 == 0:
                print(f"Step {step+1}/{total_steps}")

        # flush remaining buffer updates at end of round
        pipeline.flush_and_update()

        # Compute and print per-model language accuracies (from deques)
        print("Per-model language accuracies after round:")
        for mid in range(num_llms):
            print(f"  Model {mid} ({model_names[mid]}):")
            for lang in LANGUAGES:
                dq = pipeline.lang_history[mid].get(lang, deque())
                if len(dq) == 0:
                    acc = float('nan')
                else:
                    acc = sum(dq) / len(dq)
                print(f"    {lang}: {acc:.3f} (n={len(dq)})")
                round_summary.append([round_idx+1, mid, model_names[mid], lang, acc, len(dq)])

        # optional: save round summary CSV
        with open(f"round_summary_round{round_idx+1}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "model_id", "model_name", "language", "lang_accuracy", "history_len"])
            for row in round_summary:
                writer.writerow(row)

        # save model policy checkpoint
        torch.save(agent.policy.state_dict(), f"ppo_router_policy_round{round_idx+1}.pth")
        print(f"Checkpoint saved after round {round_idx+1}")

    print("Training complete.")
