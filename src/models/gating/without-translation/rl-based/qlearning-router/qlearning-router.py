import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import BertTokenizer, BertModel

# ----------------------
# Global Settings
# ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("../../../../../training/router_prompts/data")
MAX_TOKENS = 128
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 1
VAL_EVERY = 200

EPS_START = 0.2
EPS_END = 0.01
EPS_DECAY = 10000

# ----------------------
# Dataset Preparation
# ----------------------
def prepare_unified_dataset(news_json_path, rating_json_path, esci_json_path, out_csv):
    # Load news dataset
    news_df = pd.read_json(news_json_path)
    news_df = news_df[["prompt", "template_lang"]].rename(
        columns={"prompt": "text", "template_lang": "language"}
    )
    news_df["task"] = "news"

    # Load rating dataset
    rating_df = pd.read_json(rating_json_path)
    rating_df = rating_df[["prompt", "language_column"]].rename(
        columns={"prompt": "text", "language_column": "language"}
    )
    rating_df["task"] = "rating"
    
    # Load esci dataset
    esci_df = pd.read_json(esci_json_path)
    esci_df = esci_df[["prompt", "language_column"]].rename(
        columns={"prompt": "text", "language_column": "language"}
    )
    esci_df["task"] = "esci"

    # Join
    combined = pd.concat([news_df, rating_df, esci_df], ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"Unified dataset saved to {out_csv}, shape={combined.shape}")
    return combined


def split_unified(df, val_ratio=0.1, test_ratio=0.1):
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    n_val = int(len(df) * val_ratio)
    n_test = int(len(df) * test_ratio)

    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx = idx[n_val+n_test:]

    train = df.iloc[train_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, val, test


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=128, task_map=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_map = task_map or {"rating": 0, "news": 1, "esci": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        x = enc["input_ids"].squeeze(0)
        task_id = self.task_map[row["task"]]
        return x, task_id, row["text"], row["language"]

# ----------------------
# Models
# ----------------------
class XLMREncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        # self.model = XLMRobertaModel.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, x):
        out = self.model(input_ids=x, attention_mask=(x != self.model.config.pad_token_id).long())
        cls_emb = out.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_emb


class QRouter(nn.Module):
    def __init__(self, in_dim, num_tasks=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_tasks)
        )

    def forward(self, h):
        return self.net(h)  # Q-values per task

# ----------------------
# Q-learning Trainer
# ----------------------
class QTrainer:
    def __init__(self, enc, router, lr, num_tasks, id2task,
                 e0=EPS_START, e1=EPS_END, decay=EPS_DECAY):
        self.enc, self.router = enc.to(DEVICE), router.to(DEVICE)
        self.opt = optim.Adam(list(enc.parameters()) + list(router.parameters()), lr=lr)
        self.num_tasks = num_tasks
        self.mse = nn.MSELoss()
        self.e0, self.e1, self.dec, self.step = e0, e1, decay, 0
        self.id2task = id2task

    def epsilon(self):
        t = min(self.step / self.dec, 1)
        return self.e0 + (self.e1 - self.e0) * t

    def step_batch(self, batch):
        x, true_task, _, _ = batch
        x, true_task = x.to(DEVICE), true_task.to(DEVICE)

        # Encode
        h = self.enc(x)
        q_values = self.router(h)  # [batch, num_tasks]

        # Epsilon-greedy action
        if random.random() < self.epsilon():
            actions = torch.randint(0, self.num_tasks, (len(x),), device=DEVICE)
        else:
            actions = q_values.argmax(dim=-1)

        # Reward = 1 if action == true_task else 0
        rewards = (actions == true_task).float()

        # Q-values for taken actions
        q_taken = q_values[range(len(x)), actions]

        # Loss = MSE(Q(s,a), reward)
        loss = self.mse(q_taken, rewards)

        # Backprop
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.router.parameters()), 1.0)
        self.opt.step()
        self.step += 1
        return loss.item()

    @torch.no_grad()
    def evaluate_and_save(self, loader, out_file):
        self.enc.eval(); self.router.eval()
        rows = []
        for xb, yb, texts, langs in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            h = self.enc(xb)
            q_values = self.router(h)
            preds = q_values.argmax(-1)
            for true, pred, text, lang in zip(yb.cpu().tolist(), preds.cpu().tolist(), texts, langs):
                rows.append({
                    "text": text,
                    "language": lang,
                    "true_task": self.id2task[true],
                    "pred_task": self.id2task[pred],
                })

        df = pd.DataFrame(rows)
        df.to_csv(out_file, index=False)
        print(f"Predictions saved to {out_file}")

        # Overall accuracy
        overall_acc = (df["true_task"] == df["pred_task"]).mean()
        print(f"\nOverall test accuracy: {overall_acc:.4f}")

        # Language-wise accuracy
        lang_acc = df.groupby("language").apply(
            lambda g: (g["true_task"] == g["pred_task"]).mean()
        )
        print("\nLanguage-wise accuracy:")
        for lang, acc in lang_acc.items():
            print(f"  {lang}: {acc:.4f}")

        # Language + Task-wise accuracy
        print("\nLanguage + Task-wise accuracy:")
        grouped = df.groupby(["language", "true_task"])
        for (lang, task), g in grouped:
            acc = (g["true_task"] == g["pred_task"]).mean()
            print(f"  {lang} - {task}: {acc:.4f}")

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # 1. Prepare unified dataset
    unified_path = "unified.csv"
    combined = prepare_unified_dataset(
        DATA_DIR / "news.json", DATA_DIR / "ratings.json", DATA_DIR / "esci_with_prompts.json", unified_path
    )

    # 2. Split data
    train_df, val_df, test_df = split_unified(combined)

    task_map = {"rating": 0, "news": 1, "esci": 2}
    id2task = {v: k for k, v in task_map.items()}

    train_ds = UnifiedDataset(train_df, tokenizer, MAX_TOKENS, task_map)
    test_ds  = UnifiedDataset("./test.json", tokenizer, MAX_TOKENS, task_map)
    
    # test_ds  = UnifiedDataset(test_df, tokenizer, MAX_TOKENS, task_map)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 3. Models
    enc = XLMREncoder()
    router = QRouter(enc.out_dim, num_tasks=len(task_map))

    trainer = QTrainer(enc, router, LR, num_tasks=len(task_map), id2task=id2task)

    # 4. Training
    print("Training Q-learning Router...")
    it = iter(train_loader)
    for step in range(EPOCHS * len(train_loader)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        loss = trainer.step_batch(batch)

        if step % VAL_EVERY == 0 and step > 0:
            print(f"Step {step}, Q-loss={loss:.4f}")

    # 5. Save predictions + print detailed accuracy
    trainer.evaluate_and_save(test_loader, "predictions_qlearning.csv")