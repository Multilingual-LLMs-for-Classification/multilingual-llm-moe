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

# ----------------------
# Config
# ----------------------
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
MIN_FREQ = 2

BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
LR = 1e-3
EPOCHS = 1
VAL_EVERY = 200

BANDIT_EPS_START = 0.2
BANDIT_EPS_END = 0.01
BANDIT_EPS_DECAY = 10000

DQN_GAMMA = 0.99
DQN_EPS_START = 0.3
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 30000
DQN_TARGET_UPDATE = 1000
REPLAY_CAPACITY = 50000
REPLAY_WARMUP = 1000
DQN_BATCH_SIZE = 64
READ_COST = 0.01

# ----------------------
# Utilities
# ----------------------
def simple_tokenize(text: str) -> List[str]:
    return str(text).lower().split()

def build_vocab(texts: List[str], min_freq=2) -> Dict[str,int]:
    from collections import Counter
    cnt = Counter(tok for t in texts for tok in simple_tokenize(t))
    vocab = {"<pad>":0, "<unk>":1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def encode_text(text: str, vocab: Dict[str,int], max_len=128) -> List[int]:
    toks = simple_tokenize(text)[:max_len]
    ids = [vocab.get(t,1) for t in toks]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

def split_df(df: pd.DataFrame, val=0.1, test=0.1):
    n = len(df); idx = np.random.permutation(n)
    nv = int(val*n); nt = int(test*n)
    return (
        df.iloc[idx[nv+nt:]].reset_index(drop=True),
        df.iloc[idx[:nv]].reset_index(drop=True),
        df.iloc[idx[nv:nv+nt]].reset_index(drop=True),
    )

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

# ----------------------
# Dataset
# ----------------------
class MixedStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, s: pd.DataFrame, n: pd.DataFrame, vocab: Dict[str,int], max_len=128, p=0.5):
        super().__init__()
        self.s, self.n, self.vocab, self.max_len, self.p = (
            s.reset_index(drop=True),
            n.reset_index(drop=True),
            vocab,
            max_len,
            p,
        )
    def __iter__(self):
        while True:
            df = self.s if random.random() < self.p else self.n
            task_id = 0 if df is self.s else 1
            row = df.sample(1).iloc[0]
            x = encode_text(row["text"], self.vocab, self.max_len)
            y = int(row["label"])
            yield torch.tensor(x, dtype=torch.long), task_id, torch.tensor(y, dtype=torch.long)

def make_eval_loader(df: pd.DataFrame, task_id: int, vocab: Dict[str,int], max_len=128, bs=64):
    xs = torch.tensor([encode_text(t, vocab, max_len) for t in df["text"]], dtype=torch.long)
    ys = torch.tensor(df["label"].astype(int).tolist(), dtype=torch.long)
    tids = torch.tensor([task_id]*len(df), dtype=torch.long)
    ds = torch.utils.data.TensorDataset(xs, tids, ys)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)

# ----------------------
# Models
# ----------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, vs, ed=128, hd=128):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm = nn.LSTM(ed, hd, batch_first=True, bidirectional=True)
        self.out_dim = hd*2
    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        mask = (x!=0).float().unsqueeze(-1)
        return (o*mask).sum(1)/mask.sum(1).clamp(min=1)

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
        x, y = x.to(DEVICE), batch[2].to(DEVICE)
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
# DQN and Env
# ----------------------
class DQN(nn.Module):
    def __init__(self, in_dim, na):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,in_dim), nn.ReLU(), nn.Linear(in_dim,na))
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.cap, self.buf, self.idx = cap, [], 0
    def push(self, s,a,r,s2,d):
        if len(self.buf)<self.cap: self.buf.append(None)
        self.buf[self.idx] = (s,a,r,s2,d)
        self.idx = (self.idx+1) % self.cap

    def sample(self, bs):
        idxs = np.random.randint(0, len(self.buf), bs)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idxs])

        # Stack tensors directly, keep their device
        s = torch.stack(s).float()
        s2 = torch.stack(s2).float()
        a = torch.tensor(a, dtype=torch.long, device=s.device)
        r = torch.tensor(r, dtype=torch.float, device=s.device)
        d = torch.tensor(d, dtype=torch.float, device=s.device)

        return s, a, r, s2, d


class SequentialEnv:
    def __init__(self, enc, heads, vocab, max_len=128, chunk=32):
        self.enc = enc
        self.heads = heads
        self.vocab = vocab
        self.max_len = max_len
        self.chunk = chunk

    def reset_with_ids(self, ids, task_id, label):
        # Initialize environment state
        self.ids = list(ids)
        self.task_id = task_id
        self.label = label
        self.ptr = 1
        self.reads = 0

        return self._encode()  # returns a PyTorch tensor

    def _encode(self):
        up = min(self.ptr * self.chunk, self.max_len)
        pid = self.ids[:up] + [0] * (self.max_len - up)

        # Device-agnostic: use the same device as the encoder
        device = next(self.enc.parameters()).device
        x = torch.tensor([pid], dtype=torch.long, device=device)

        # Run encoder, detach to stop gradients
        return self.enc(x).squeeze(0).detach()  # PyTorch tensor on same device

    def step(self, action):
        if action == 0:  # read more tokens
            if self.ptr * self.chunk < self.max_len:
                self.ptr += 1
                self.reads += 1
            return self._encode(), -READ_COST, False

        else:  # classify
            h = self._encode()
            device = next(self.enc.parameters()).device
            x = torch.tensor([self.ids], dtype=torch.long, device=device)
            hf = self.enc(x)

            if action == 1:  # sentiment
                pred = self.heads.sent(hf).argmax(-1).item()
                correct = (pred == self.label and self.task_id == 0)
            else:  # news
                pred = self.heads.news(hf).argmax(-1).item()
                correct = (pred == self.label and self.task_id == 1)

            reward = float(correct) - READ_COST * self.reads
            return h, reward, True


class DQNTrainer:
    def __init__(self, enc, heads, dqn_mod, na, gamma, lr, e0, e1, dec, tu):
        self.enc, self.heads, self.dqn = enc.to(DEVICE), heads.to(DEVICE), dqn_mod.to(DEVICE)
        self.target = DQN(self.dqn.net[0].in_features, na).to(DEVICE)
        self.target.load_state_dict(self.dqn.state_dict()); self.target.eval()
        self.opt = optim.Adam(list(enc.parameters())+list(heads.parameters())+list(dqn_mod.parameters()), lr=lr)
        self.gamma, self.e0, self.e1, self.dec, self.tu = gamma, e0, e1, dec, tu
        self.replay = ReplayBuffer(REPLAY_CAPACITY)
        self.step, self.loss_fn = 0, nn.SmoothL1Loss()
    def epsilon(self):
        t = min(self.step/self.dec,1)
        return self.e0 + (self.e1-self.e0)*t
    def act(self, s):
        if random.random()<self.epsilon(): return random.randrange(3)
        return int(self.dqn(torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)).argmax(-1))
    def optimize(self):
        if len(self.replay.buf)<max(REPLAY_WARMUP, DQN_BATCH_SIZE): return 0.0
        s,a,r,s2,d = self.replay.sample(DQN_BATCH_SIZE)
        q = self.dqn(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.target(s2).max(1)[0]
            tgt = r + (1-d)*self.gamma*q2
        loss = self.loss_fn(q, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()
    @torch.no_grad()
    def evaluate(self, loaders):
        self.enc.train(); self.heads.train(); self.dqn.train()
        res = {}
        for name, dl in loaders.items():
            yt, yp, tc = [], [], []
            for xb, tb, yb in dl:
                for i in range(len(xb)):
                    xi = xb[i:i+1].to(DEVICE)
                    h = self.enc(xi).squeeze(0)
                    a = int(self.dqn(h.unsqueeze(0)).argmax(-1))
                    lbl = int(yb[i])
                    # Skip misrouted
                    if (a==1 and lbl>=self.heads.sent.out_features) or (a==2 and lbl>=self.heads.news.out_features):
                        continue
                    if a==1:
                        p = int(self.heads.sent(h.unsqueeze(0)).argmax(-1))
                        task_pred = 0
                    else:
                        p = int(self.heads.news(h.unsqueeze(0)).argmax(-1))
                        task_pred = 1
                    yt.append(lbl); yp.append(p); tc.append(task_pred==int(tb[i]))
            acc = sum(t==p for t,p in zip(yt,yp))/len(yt) if yt else 0
            tac = sum(tc)/len(tc) if tc else 0
            res[name] = {"acc":acc, "task_acc":tac}
        return res

# ----------------------
# Training
# ----------------------
def train_bandit(mix_loader, val_loaders, vocab_size):
    enc = BiLSTMEncoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
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
                torch.save({"enc":enc.state_dict(),"heads":heads.state_dict(),"router":router.state_dict()},"bandit.pt")
    return enc, heads, router

def train_dqn(mix_loader, val_loaders, vocab, steps=20000):
    enc = BiLSTMEncoder(len(vocab), EMBED_DIM, HIDDEN_DIM)
    heads = TaskHeads(enc.out_dim, 3, 4)
    dqn_mod = DQN(enc.out_dim, 3)
    trainer = DQNTrainer(enc, heads, dqn_mod, 3, DQN_GAMMA, LR, DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY, DQN_TARGET_UPDATE)
    env = SequentialEnv(enc, heads, vocab, MAX_TOKENS, 32)
    it = iter(mix_loader); best=0
    for step in range(steps):
        try:
            xb, tb, yb = next(it)
        except StopIteration:
            it = iter(mix_loader)
            continue
        ids = xb[0].tolist()
        s = env.reset_with_ids(ids, tb[0].item(), yb[0].item())
        done = False
        while not done:
            a = trainer.act(s)
            s2, r, done = env.step(a)
            trainer.replay.push(s, a, r, s2, done)
            s = s2
            _ = trainer.optimize()
            trainer.step += 1
            if trainer.step % trainer.tu == 0:
                trainer.target.load_state_dict(trainer.dqn.state_dict())
        if step%VAL_EVERY==0 and step>0:
            res = trainer.evaluate(val_loaders)
            metric = (res["sent_val"]["task_acc"]+res["news_val"]["task_acc"])/2
            print(f"[DQN] {step} val={res}")
            if metric>best:
                best=metric
                torch.save({"enc":enc.state_dict(),"heads":heads.state_dict(),"dqn":dqn_mod.state_dict()},"dqn.pt")
    return enc, heads, dqn_mod

# ----------------------
# Main
# ----------------------
if __name__=="__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError("Create data/ folder with CSVs")
    s_all = read_task_csvs(SENTIMENT_SINGLE, SENTIMENT_PATTERN)
    n_all = read_task_csvs(NEWS_SINGLE, NEWS_PATTERN)
    s_tr, s_v, _ = split_df(s_all)
    n_tr, n_v, _ = split_df(n_all)
    print("Sentiment labels:", s_tr["label"].min(), "to", s_tr["label"].max())
    print("News labels:     ", n_tr["label"].min(), "to", n_tr["label"].max())
    vocab = build_vocab(list(s_tr["text"])+list(n_tr["text"]), MIN_FREQ)
    mix = MixedStreamDataset(s_tr, n_tr, vocab, MAX_TOKENS)
    val_loaders = {
        "sent_val": make_eval_loader(s_v, 0, vocab, MAX_TOKENS),
        "news_val": make_eval_loader(n_v, 1, vocab, MAX_TOKENS),
    }
    print("Training Bandit...")
    train_bandit(torch.utils.data.DataLoader(mix, batch_size=BATCH_SIZE), val_loaders, len(vocab))
    print("Training DQN...")
    train_dqn(torch.utils.data.DataLoader(mix, batch_size=BATCH_SIZE), val_loaders, vocab)
