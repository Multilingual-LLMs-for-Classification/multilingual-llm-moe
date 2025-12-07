from collections import defaultdict, deque
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
TOKEN = HF_TOKEN # Alias for use inside the class

# ------------------------------
# Language-specific prompts
# ------------------------------

# Prompts are adapted to strictly request one category name from the list.
# The list of categories is formatted into the prompt during runtime.

# ------------------------------
# Language-specific prompts (Adjusted for English Output)
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
            
            # PPO Actor Loss (Surrogate Objective)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # PPO Critic Loss (Value Function)
            critic_loss = (returns - values.squeeze()) ** 2
            critic_loss = critic_loss.mean()
            
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
        self.categories = categories # Store categories list
        self.models = [None for _ in model_names]
        self.tokenizers = [None for _ in model_names]

    def get_categories_string(self):
        # Helper to format the categories into a string for the prompt
        return ", ".join(self.categories)

    def extract_label(self, decoded):
        """
        Extracts the predicted category label from the LLM's raw output.
        Aggressively cleans the output and attempts to find a single valid
        English category name within the cleaned text.
        """
        
        # 1. Clean the raw output to isolate the LLM's final response
        # Split by the prompt's final line and take the last part (the predicted label)
        # This handles cases where the LLM repeats the input prompt.
        prompt_tag = "Output (Category only):"
        prompt_tag_dan = "Output (Kun kategori):"
        prompt_tag_span = "Salida (Categoría solamente):"
        prompt_tag_pol = "Wyjście (Tylko kategoria):"
        prompt_tag_turk = "Çıktı (Sadece Kategori):"
        
        raw_output = decoded
        for tag in [prompt_tag, prompt_tag_dan, prompt_tag_span, prompt_tag_pol, prompt_tag_turk]:
            if tag in raw_output:
                raw_output = raw_output.split(tag)[-1].strip()
                break
        else:
            # Fallback for non-English prompts or different LLM behavior
            raw_output = decoded.strip().split('\n')[-1].strip()

        # Aggressive cleaning: lower-case, remove common artifacts like quotes, 
        # and strip whitespace/newlines
        cleaned_output = raw_output.lower().replace('"', '').replace("'", '').strip()

        # 2. Iterate through canonical English categories and check for containment
        # Sort categories by length descending to prioritize multi-word categories 
        # (e.g., 'Tax & Accounting' over just 'Tax')
        sorted_categories = sorted(self.categories, key=len, reverse=True)

        for category in sorted_categories:
            cat_lower = category.lower()
            
            # Check 1: Exact Match (after cleaning)
            if cat_lower == cleaned_output:
                return category, category

            # Check 2: Aggressive Containment
            if cat_lower in cleaned_output:
                # We return the canonical category name
                return category, category
                
        # 3. Fallback: If no valid category is found, prepare a clean "Unknown" output.
        # This ensures string_output is clean for logging.
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_output) # Keep only letters and spaces
        string_output = " ".join([t.strip() for t in cleaned_text.split()])
        
        # For logging purposes: log the best cleaned string, but the predicted label is "Unknown"
        return string_output, "Unknown"

    def run(self, llm_id, text, language="English"):
        llm_id = min(llm_id, len(self.model_names)-1)

        if self.models[llm_id] is None:

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # NOTE: Use the global TOKEN variable set from HF_TOKEN
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
        
        # Format the prompt with the article text AND the categories list
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
# Routing Pipeline (Modified for reward logging only)
# ------------------------------

class RoutingPipeline:
    def __init__(self, agent, llm_system, state_dim):
        self.agent = agent
        self.llm_system = llm_system
        self.state_dim = state_dim
        self.lang_history = defaultdict(lambda: deque(maxlen=100))
        # Precompile helpers for lightweight text featurization
        self._token_pattern = re.compile(r"\b\w+\b", re.UNICODE)
        self._positive_words = {
            "gain","growth","improve","profit","success",
            "strong","upbeat","bullish","stable","optimistic"
        }
        self._negative_words = {
            "loss","decline","drop","risk","weak",
            "downturn","bearish","crisis","debt","uncertain"
        }

    def _tokenize(self, text):
        tokens = self._token_pattern.findall(text.lower())
        return tokens

    def _sentiment_score(self, tokens):
        if not tokens:
            return 0.0
        pos_hits = sum(token in self._positive_words for token in tokens)
        neg_hits = sum(token in self._negative_words for token in tokens)
        return (pos_hits - neg_hits) / len(tokens)

    def get_state(self, text, language):
        # State generation logic remains the same (log of text length + language one-hot)
        lang_map = {"English":0, "Turkish":1, "Danish":2, "Spanish":3, "Polish":4}
        lang_vec = np.zeros(6)
        # Handle cases where language in data is 'Dani', 'Turk', 'Span', 'Pol' and needs mapping
        lang_key = language
        if language in {"Dani", "Danish"}: lang_key = "Danish"
        elif language in {"Turk", "Turkish"}: lang_key = "Turkish"
        elif language in {"Span", "Spanish"}: lang_key = "Spanish"
        elif language in {"Pol", "Polish"}: lang_key = "Polish"
        elif language in {"English", "en"}: lang_key = "English"


        if lang_key in lang_map:
            lang_vec[lang_map[lang_key]] = 1
        
        # Feature 1: Log of Text Length
        text_len_log = np.log(len(text) + 1)
        tokens = self._tokenize(text)
        token_lengths = [len(tok) for tok in tokens]
        avg_word_len = float(np.mean(token_lengths)) if token_lengths else 0.0
        vocab_richness = (len(set(tokens)) / len(tokens)) if tokens else 0.0
        sentiment_score = self._sentiment_score(tokens)
        features = np.array([
            text_len_log,
            avg_word_len,
            vocab_richness,
            sentiment_score
        ])
        
        # Informative state size is now 4 numeric features + 6 (lang_vec) = 10
        informative_dim = len(features) + len(lang_vec)
        if self.state_dim <= informative_dim:
            # Truncate if caller configured a very small state_dim
            return np.concatenate([features, lang_vec])[:self.state_dim]

        # Pad the remaining dimensions with a fixed value (e.g., 0.0)
        padding_size = self.state_dim - informative_dim
        padding = np.zeros(padding_size)
        return np.concatenate([features, padding, lang_vec])

    def reward_fn(self, pred_label, true_label, lang):
        """
        Original reward function: 1.0 if the predicted label 
        exactly matches the true label, 0.0 otherwise.
        """
        per_sample = 1.0 if pred_label == true_label else 0.0
        self.lang_history[lang].append(per_sample)

        # Language-wise accuracy
        lang_accuracy = sum(self.lang_history[lang]) / len(self.lang_history[lang])

        # Combine them
        final_reward = 0.5 * per_sample + 0.5 * lang_accuracy
        return final_reward


    def train_step(self, batch, output_csv=None):
        states, actions, old_log_probs, rewards, values, dones = [], [], [], [], [], []

        for _, row in batch.iterrows():
            state = self.get_state(row['text'], row['lang'])
            action, log_prob, value = self.agent.select_action(state)

            # NOTE: We use the language keys defined in PROMPTS, not the full name
            lang_key_for_prompt = row['lang'] # Assumes row['lang'] is 'Dani', 'English', etc.
            
            pred_label, decoded_output, prompt_used, llm_name_used, cleaned_output = \
                self.llm_system.run(action, row['text'], lang_key_for_prompt)
            
            true_label = row['label'] # The ground truth category
            print(f"True Category: {true_label}")
            
            # Original call to reward_fn (no llm_id passed)
            reward = self.reward_fn(pred_label, true_label, row["lang"])

            if output_csv is not None:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # **MODIFIED to include reward in CSV**
                    writer.writerow([
                        row['text'],
                        row['lang'],
                        true_label,
                        cleaned_output,
                        action,
                        pred_label,
                        reward, # <- ADDED REWARD HERE
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
        # Note: actions need to be tensor of correct type for indexing/dist.log_prob
        actions = torch.tensor(actions, dtype=torch.int64) 
        old_log_probs = torch.stack(old_log_probs).detach()

        returns, advantages = self.agent.compute_returns(rewards, values, dones)
        self.agent.update(states, actions, old_log_probs, returns, advantages)


# ------------------------------
# Main training (Modified CSV Header)
# ------------------------------
if __name__ == "__main__":
    state_dim = 20
    batch_size = 4

    # LLMs remain the same
    model_names = [
        "CohereLabs/aya-expanse-8b",
        "Qwen/Qwen2.5-3B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        
    ]

    num_llms = len(model_names)

    # Pass the categories list to the LLM system
    llm_system = HuggingFaceLLMSystem(model_names, PROMPTS, NEWS_CATEGORIES)
    
    # Using original gamma (0.99) and entropy_weight (0.01)
    agent = PPOAgent(state_dim, num_llms, entropy_weight=0.01) 
    
    pipeline = RoutingPipeline(agent, llm_system, state_dim)

    output_csv_path = "llm_classification_outputs.csv"
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # **MODIFIED CSV header to include 'reward'**
            writer.writerow([
                "article_text","language","true_category","cleaned_output",
                "action","pred_category","reward",
                "decoded_output","prompt","llm_name_used"
            ])

    try:
        # NOTE: Updated column names to match usage in RoutingPipeline: 'text', 'lang', 'label'
        # Please ensure you update the path below to your actual training data path:
        data = pd.read_csv("../../data/train.csv", # Placeholder path
                            dtype={'text': str, 'lang': str, 'label': str},
                            encoding='utf-8',
                            on_bad_lines='skip')
    except FileNotFoundError:
        print("Error: Training data file not found. Please ensure the path is correct and the file exists.")
        exit(1)

    # Ensure all languages in the data have a prompt defined for them
    available_langs = set(data['lang'].unique())
    supported_langs = set(PROMPTS.keys())
    unsupported_langs = available_langs - supported_langs
    
    if unsupported_langs:
        print(f"Warning: The dataset contains unsupported languages: {unsupported_langs}. Falling back to 'en' prompt for these samples.")


    ROUNDS = 1
    
    # Calculate total steps per round for non-repeating batches
    total_steps_per_round = len(data) // batch_size
    print(f"Dataset size: {len(data)}. Batch size: {batch_size}. Steps per round: {total_steps_per_round}.")
    
    for round in range(ROUNDS):
        print(f"=== ROUND {round+1} / {ROUNDS} ===")
        
        # 1. SHUFFLE the entire dataset at the start of the round (NON-REPETITION)
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        
        for step in range(total_steps_per_round):
            # 2. Extract a non-repeating batch using index slicing
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size
            batch = shuffled_data.iloc[start_idx:end_idx]
                
            pipeline.train_step(batch, output_csv=output_csv_path)
            
            if step % 25 == 0 and step != 0:
                print(f"Step {step}/{total_steps_per_round} completed")

        torch.save(agent.policy.state_dict(), f"ppo_router_policy_news_round{round+1}.pth")
        print(f"Checkpoint saved after round {round+1}")
        
    print("Training complete.")
