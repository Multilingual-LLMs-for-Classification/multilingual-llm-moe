import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# ----------------------
# Global Settings
# ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Adjust this path to where your training folder is
DATA_DIR = Path("../../../../../training/router_prompts/data")

MAX_TOKENS = 128
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 1
VAL_EVERY = 200

# ----------------------
# Dataset Preparation
# ----------------------
def prepare_unified_dataset(news_json_path, rating_json_path, out_csv):
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

    # Join both
    combined = pd.concat([news_df, rating_df], ignore_index=True)
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
        self.task_map = task_map or {"rating": 0, "news": 1}

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
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, x):
        out = self.model(input_ids=x, attention_mask=(x != self.model.config.pad_token_id).long())
        cls_emb = out.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_emb


class TaskClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        return self.classifier(h)

# ----------------------
# Trainer
# ----------------------
class TaskTrainer:
    def __init__(self, enc, clf, lr, id2task):
        self.enc, self.clf = enc.to(DEVICE), clf.to(DEVICE)
        self.opt = optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=lr)
        self.ce = nn.CrossEntropyLoss()
        self.id2task = id2task

    def step_batch(self, batch):
        x, y, _, _ = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        h = self.enc(x)
        logits = self.clf(h)
        loss = self.ce(logits, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.enc.parameters()) + list(self.clf.parameters()), 1.0)
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, loader):
        self.enc.eval(); self.clf.eval()
        yt, yp = [], []
        for xb, yb, _, _ in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            h = self.enc(xb)
            preds = self.clf(h).argmax(-1)
            yt += yb.cpu().tolist()
            yp += preds.cpu().tolist()
        acc = sum(int(a == b) for a, b in zip(yt, yp)) / len(yt)
        return acc

    @torch.no_grad()
    def evaluate_and_save(self, loader, out_file):
        self.enc.eval(); self.clf.eval()
        rows = []
        for xb, yb, texts, langs in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            h = self.enc(xb)
            preds = self.clf(h).argmax(-1)
            for true, pred, text, lang in zip(yb.cpu().tolist(), preds.cpu().tolist(), texts, langs):
                rows.append({
                    "text": text,
                    "language": lang,
                    "true_task": self.id2task[true],
                    "pred_task": self.id2task[pred],
                })

        # Save predictions
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
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # 1. Prepare unified dataset
    unified_path = DATA_DIR / "unified.csv"
    combined = prepare_unified_dataset(
        DATA_DIR / "news.json", DATA_DIR / "ratings.json", unified_path
    )

    # 2. Split data
    train_df, val_df, test_df = split_unified(combined)

    task_map = {"rating": 0, "news": 1}
    id2task = {v: k for k, v in task_map.items()}

    train_ds = UnifiedDataset(train_df, tokenizer, MAX_TOKENS, task_map)
    val_ds   = UnifiedDataset(val_df, tokenizer, MAX_TOKENS, task_map)
    test_ds  = UnifiedDataset(test_df, tokenizer, MAX_TOKENS, task_map)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 3. Models
    enc = XLMREncoder()
    clf = TaskClassifier(enc.out_dim, num_classes=len(task_map))

    trainer = TaskTrainer(enc, clf, LR, id2task)

    # 4. Training
    print("Training Task Classifier...")
    it = iter(train_loader)
    best = 0
    for step in range(EPOCHS * len(train_loader)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        loss = trainer.step_batch(batch)

        if step % VAL_EVERY == 0 and step > 0:
            acc = trainer.evaluate(val_loader)
            print(f"Step {step}, loss={loss:.4f}, val_acc={acc:.4f}")
            if acc > best:
                best = acc
                torch.save({
                    "enc": enc.state_dict(),
                    "clf": clf.state_dict(),
                }, DATA_DIR / "task_classifier.pt")

    # 5. Save predictions + print detailed accuracy
    trainer.evaluate_and_save(test_loader, "predictions.csv")
