import os
import random
from pathlib import Path
import glob
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from transformers import XLMRobertaTokenizer, XLMRobertaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("data")
SENTIMENT_SINGLE = DATA_DIR / "sentiment.csv"
NEWS_SINGLE = DATA_DIR / "news.csv"
SENTIMENT_PATTERN = str(DATA_DIR / "sentiment_*.csv")
NEWS_PATTERN = str(DATA_DIR / "news_*.csv")

MAX_TOKENS = 128

BATCH_SIZE = 16  # reduce if memory issues
LR = 1e-5
EPOCHS = 1
VAL_EVERY = 200

BANDIT_EPS_START = 0.2
BANDIT_EPS_END = 0.01
BANDIT_EPS_DECAY = 10000

# ----------------------
# Utilities
# ----------------------
def read_task_csvs(single: Path, pattern: str) -> pd.DataFrame:
    files = [str(single)] if single.exists() else glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSVs found for {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
        except pd.errors.ParserError:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
        if {"text","label"}.issubset(df.columns):
            df = df.dropna(subset=["text","label"])
            dfs.append(df)
    if not dfs:
        raise ValueError("No valid CSVs loaded")
    return pd.concat(dfs, ignore_index=True)

def split_df(df: pd.DataFrame, val=0.1, test=0.1):
    n = len(df); idx = np.random.permutation(n)
    nv = int(val*n); nt = int(test*n)
    return (
        df.iloc[idx[nv+nt:]].reset_index(drop=True),
        df.iloc[idx[:nv]].reset_index(drop=True),
        df.iloc[idx[nv:nv+nt]].reset_index(drop=True),
    )

# ----------------------
# Dataset
# ----------------------
class MixedStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, s: pd.DataFrame, n: pd.DataFrame, tokenizer: XLMRobertaTokenizer, max_len=128, p=0.5):
        super().__init__()
        self.s, self.n, self.tokenizer, self.max_len, self.p = (
            s.reset_index(drop=True),
            n.reset_index(drop=True),
            tokenizer,
            max_len,
            p,
        )
    def __iter__(self):
        while True:
            df = self.s if random.random() < self.p else self.n
            task_id = 0 if df is self.s else 1
            row = df.sample(1).iloc[0]
            enc = self.tokenizer(
                row["text"],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            x = enc["input_ids"].squeeze(0)
            y = int(row["label"])
            yield x, task_id, torch.tensor(y, dtype=torch.long)

def make_eval_loader(df: pd.DataFrame, task_id: int, tokenizer: XLMRobertaTokenizer, max_len=128, bs=16):
    xs, ys, tids = [], [], []
    for t,lbl in zip(df["text"], df["label"]):
        enc = tokenizer(t, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        xs.append(enc["input_ids"].squeeze(0))
        ys.append(int(lbl))
        tids.append(task_id)
    ds = torch.utils.data.TensorDataset(torch.stack(xs), torch.tensor(tids), torch.tensor(ys))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

# ----------------------
# Models
# ----------------------
class XLMREncoder(nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, x):
        out = self.model(input_ids=x, attention_mask=(x!=self.model.config.pad_token_id).long())
        cls_emb = out.last_hidden_state[:,0,:]  # [CLS] token
        return cls_emb

class TaskHeads(nn.Module):
    def __init__(self, in_dim, sc, nc):
        super().__init__()
        self.sent = nn.Linear(in_dim, sc)
        self.news = nn.Linear(in_dim, nc)

# ----------------------
# Bandit Router
# ----------------------
class BanditRouter(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,in_dim), nn.ReLU(), nn.Linear(in_dim,2))
    def forward(self, h): return self.net(h)

class BanditRouterTrainer:
    def __init__(self, enc, heads, router, lr, e0, e1, dec):
        self.enc, self.heads, self.router = enc.to(DEVICE), heads.to(DEVICE), router.to(DEVICE)
        self.opt = optim.Adam(list(enc.parameters())+list(heads.parameters())+list(router.parameters()), lr=lr)
        self.e0, self.e1, self.dec, self.step = e0, e1, dec, 0
        self.ce = nn.CrossEntropyLoss()
    def epsilon(self):
        t = min(self.step/self.dec,1)
        return self.e0 + (self.e1-self.e0)*t
    def route_action(self, logits):
        if random.random() < self.epsilon():
            return torch.randint(0,2,(logits.size(0),), device=DEVICE)
        return logits.argmax(-1)
    def step_batch(self, batch):
        x, true_task, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        h = self.enc(x)
        logits = self.router(h)
        a = self.route_action(logits)
        loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        count = 0
        for i in range(a.size(0)):
            if a[i].item() != true_task[i]:
                continue
            hi = h[i:i+1]
            yi = y[i:i+1]
            if a[i]==0:
                out = self.heads.sent(hi)
            else:
                out = self.heads.news(hi)
            l = self.ce(out, yi)
            loss = loss + l if count>0 else l
            count += 1
        if count>1:
            loss = loss / count
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.enc.parameters())+list(self.heads.parameters())+list(self.router.parameters()), 1.0)
        self.opt.step()
        self.step += 1
        return loss.item()
    @torch.no_grad()
    def evaluate(self, loaders):
        self.enc.train(); self.router.train()
        results = {}
        for name, dl in loaders.items():
            yt, yp, tc = [], [], []
            for xb, tb, yb in dl:
                xb = xb.to(DEVICE)
                h = self.enc(xb)
                a = self.router(h).argmax(-1)
                tc += ((a.cpu()==tb).tolist())
                for i in range(len(a)):
                    head = self.heads.sent if a[i]==0 else self.heads.news
                    p = head(h[i:i+1]).argmax(-1).item()
                    yt.append(int(yb[i])); yp.append(p)
            acc = sum(t==p for t,p in zip(yt,yp))/len(yt)
            task_acc = sum(tc)/len(tc)
            results[name] = {"acc":acc, "task_acc":task_acc}
        return results

# ----------------------
# Training
# ----------------------
def train_bandit(mix_loader, val_loaders):
    enc = XLMREncoder()
    heads = TaskHeads(enc.out_dim, 3, 4)
    router = BanditRouter(enc.out_dim)
    trainer = BanditRouterTrainer(enc, heads, router, LR, BANDIT_EPS_START, BANDIT_EPS_END, BANDIT_EPS_DECAY)
    it = iter(mix_loader); best=0
    for step in range(EPOCHS*2000):
        try:
            loss = trainer.step_batch(next(it))
        except Exception as e:
            print("Bandit step error:", e)
            continue
        if step%VAL_EVERY==0 and step>0:
            res = trainer.evaluate(val_loaders)
            metric = (res["sent_val"]["task_acc"]+res["news_val"]["task_acc"])/2
            print(f"[Bandit] {step} loss={loss:.4f} val={res}")
            if metric>best:
                best=metric
                torch.save({"enc":enc.state_dict(),"heads":heads.state_dict(),"router":router.state_dict()},"bandit_xlmr.pt")
    return enc, heads, router

# ----------------------
# Main
# ----------------------
if __name__=="__main__":
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    if not DATA_DIR.exists():
        raise FileNotFoundError("Create data/ folder with CSVs")
    s_all = read_task_csvs(SENTIMENT_SINGLE, SENTIMENT_PATTERN)
    n_all = read_task_csvs(NEWS_SINGLE, NEWS_PATTERN)
    s_tr, s_v, _ = split_df(s_all)
    n_tr, n_v, _ = split_df(n_all)
    print("Sentiment labels:", s_tr["label"].min(), "to", s_tr["label"].max())
    print("News labels:     ", n_tr["label"].min(), "to", n_tr["label"].max())
    
    mix = MixedStreamDataset(s_tr, n_tr, tokenizer, MAX_TOKENS)
    val_loaders = {
        "sent_val": make_eval_loader(s_v, 0, tokenizer, MAX_TOKENS),
        "news_val": make_eval_loader(n_v, 1, tokenizer, MAX_TOKENS),
    }
    print("Training Bandit with XLM-R...")
    train_bandit(torch.utils.data.DataLoader(mix, batch_size=BATCH_SIZE), val_loaders)
