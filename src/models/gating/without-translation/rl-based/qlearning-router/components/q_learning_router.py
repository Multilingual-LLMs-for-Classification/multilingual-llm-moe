"""
Q-learning based task routing component.

Implements reinforcement learning approach for task selection within domains.
Includes TransformersEncoder, QRouter, and QLearningTaskClassifier.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Import device from parent
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Get DEVICE constant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TransformersEncoder(nn.Module):
    """
    Shared multilingual encoder. Defaults to 'xlm-roberta-base'.
    """
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # CLS-like token (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb
    
    def tokenize(self, texts: List[str], max_len: int = 128):
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]


class QRouter(nn.Module):
    def __init__(self, in_dim, num_tasks):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_tasks)
        )
    def forward(self, h):
        return self.net(h)  # Q-values


class DomainTaskDataset(Dataset):
    """
    Dataset for Q-learning: sample = (input_ids, attention_mask, true_task_id, text, language)
    Filtered per-domain using given task2id.
    """
    def __init__(self, items: List[Dict], encoder: TransformersEncoder, task2id: Dict[str, int], max_len=128):
        self.items = [it for it in items if it["task"] in task2id]
        self.encoder = encoder
        self.task2id = task2id
        self.max_len = max_len

        texts = [it["prompt"] for it in self.items]
        self.input_ids, self.attn = self.encoder.tokenize(texts, max_len=max_len)
        self.labels = torch.tensor([task2id[it["task"]] for it in self.items], dtype=torch.long)
        self.langs = [it.get("language", "") for it in self.items]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn[idx], self.labels[idx], self.items[idx]["prompt"], self.langs[idx])


class QLearningTaskClassifier:
    """
    Replaces PPO TaskClassifier. Maintains:
      - Shared multilingual encoder
      - Per-domain QRouter (num_tasks = len(tasks in that domain))
    Trains with simple epsilon-greedy Q-learning on labeled (domain, task) prompts.
    """
    def __init__(
        self,
        domain_tasks,
        model_dir: Path = None,
        encoder_name: str = "xlm-roberta-base",
        max_len: int = 128,
        batch_size: int = 16,
        lr: float = 1e-5,
        epochs: int = 1,
        eps_start: float = 0.2,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10000
    ):
        # Normalize domain_tasks (DomainTaskLoader or dict)
        if hasattr(domain_tasks, "domain_tasks"):
            domain_tasks = domain_tasks.domain_tasks
        self.domain_tasks: Dict[str, Dict[str, dict]] = domain_tasks

        self.device = torch.device(DEVICE)
        self.encoder = TransformersEncoder(encoder_name).to(self.device)
        self.encoder_name = encoder_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.e0, self.e1, self.edec = eps_start, eps_end, eps_decay_steps
        self.global_step = 0
        
        # Prepare mappings and routers
        self.task2id: Dict[str, Dict[str, int]] = {}
        self.id2task: Dict[str, Dict[int, str]] = {}
        self.routers: Dict[str, QRouter] = {}
        
        for domain, tasks in self.domain_tasks.items():
            task_names = list(tasks.keys())
            t2i = {t: i for i, t in enumerate(task_names)}
            i2t = {i: t for t, i in t2i.items()}
            self.task2id[domain] = t2i
            self.id2task[domain] = i2t
            self.routers[domain] = QRouter(self.encoder.out_dim, num_tasks=len(task_names)).to(self.device)
        
        # Single optimizer for encoder + all routers
        params = list(self.encoder.parameters()) + [p for r in self.routers.values() for p in r.parameters()]
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.mse = nn.MSELoss()
        
        # Model storage
        self.model_dir = model_dir or (Path(__file__).parent.parent / "models" / "task_routers_qlearning")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def epsilon(self) -> float:
        t = min(self.global_step / max(1, self.edec), 1.0)
        return self.e0 + (self.e1 - self.e0) * t
    
    def _domain_train_loop(self, domain: str, items: List[Dict]):
        if not items:
            print(f"⚠️ No training items for domain '{domain}', skipping.")
            return
        dataset = DomainTaskDataset(items, self.encoder, self.task2id[domain], max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE=="cuda"))
        router = self.routers[domain]
        
        print(f"Training QRouter for domain '{domain}'   | samples={len(dataset)} tasks={len(self.task2id[domain])}")
        for epoch in range(self.epochs):
            router.train(); self.encoder.train()
            for input_ids, attn, labels, _, _ in loader:
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                labels = labels.to(self.device)
                
                # Encode
                h = self.encoder(input_ids, attn)
                q_values = router(h)  # [B, num_tasks]
                
                # ε-greedy selection
                if random.random() < self.epsilon():
                    actions = torch.randint(0, q_values.size(1), (q_values.size(0),), device=self.device)
                else:
                    actions = q_values.argmax(dim=-1)
                
                rewards = (actions == labels).float()
                q_taken = q_values[torch.arange(q_values.size(0), device=self.device), actions]
                
                loss = self.mse(q_taken, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(router.parameters()), 1.0)
                self.optimizer.step()
                self.global_step += 1
            
            print(f"  Epoch {epoch+1}/{self.epochs} finished for domain '{domain}'")
    
    @torch.no_grad()
    def _domain_eval(self, domain: str, items: List[Dict]) -> float:
        if not items:
            return 0.0
        dataset = DomainTaskDataset(items, self.encoder, self.task2id[domain], max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))
        router = self.routers[domain]
        router.eval(); self.encoder.eval()
        correct = 0; total = 0
        for input_ids, attn, labels, _, _ in loader:
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)
            labels = labels.to(self.device)
            h = self.encoder(input_ids, attn)
            q_values = router(h)
            preds = q_values.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(1, total)
        return acc
    
    def train(self, training_data: List[Dict], val_split: float = 0.1):
        """
        training_data: list of dicts with keys {'prompt','domain','task',('language')}
        """
        for domain in self.domain_tasks.keys():
            domain_items = [d for d in training_data if d.get("domain") == domain and d.get("task") in self.task2id[domain]]
            if not domain_items:
                print(f"⚠️ No labeled items for domain '{domain}', skipping training.")
                continue
            idx = np.random.permutation(len(domain_items))
            n_val = int(val_split * len(domain_items))
            val_items = [domain_items[i] for i in idx[:n_val]]
            train_items = [domain_items[i] for i in idx[n_val:]]
            self._domain_train_loop(domain, train_items)
            acc = self._domain_eval(domain, val_items) if val_items else 0.0
            print(f"  → Validation accuracy for domain '{domain}': {acc:.4f}")
    
    @torch.no_grad()
    def classify_task(self, text: str, domain: str) -> str:
        if domain not in self.routers:
            task_names = list(self.domain_tasks[domain].keys())
            return task_names[0] if task_names else "unknown"
        self.encoder.eval(); self.routers[domain].eval()
        input_ids, attn = self.encoder.tokenize([text], max_len=self.max_len)
        input_ids = input_ids.to(self.device); attn = attn.to(self.device)
        h = self.encoder(input_ids, attn)
        q_values = self.routers[domain](h)
        pred_id = int(q_values.argmax(dim=-1).item())
        return self.id2task[domain].get(pred_id, "unknown")
    
    def save_models(self):
        enc_path = self.model_dir / "encoder.pth"
        torch.save(self.encoder.state_dict(), enc_path)
        cfg_path = self.model_dir / "qrouter_config.json"
        with open(cfg_path, "w") as f:
            json.dump({"encoder_name": self.encoder_name}, f)
        for domain, router in self.routers.items():
            rp = self.model_dir / f"router_{domain}.pth"
            torch.save(router.state_dict(), rp)
        print(f"✅ Q-learning routers saved to: {self.model_dir}")
    
    def load_models(self) -> bool:
        cfg_path = self.model_dir / "qrouter_config.json"
        enc_path = self.model_dir / "encoder.pth"
        ok = True
        if cfg_path.exists() and cfg_path.stat().st_size > 0:
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                    enc_name = cfg.get("encoder_name", self.encoder_name)
                    if enc_name != self.encoder_name:
                        print(f"ℹ️ Stored encoder '{enc_name}' differs from requested '{self.encoder_name}'. Using stored name.")
                        self.encoder = TransformersEncoder(enc_name).to(self.device)
                        self.encoder_name = enc_name
            except json.JSONDecodeError:
                print(f"⚠️  Config file {cfg_path} is corrupted, using default encoder")
                ok = False
        if enc_path.exists():
            self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
        else:
            ok = False
        for domain, router in self.routers.items():
            rp = self.model_dir / f"router_{domain}.pth"
            if rp.exists():
                router.load_state_dict(torch.load(rp, map_location=self.device))
            else:
                ok = False
        if ok:
            print(f"✅ Q-learning routers loaded from: {self.model_dir}")
        else:
            print(f"⚠️ Q-learning routers not fully found; will train new ones.")
        return ok

# 4) TASK EXPERTS (unchanged stub)
# class TaskExpert:
#     def __init__(self, task_name):
#         self.task_name = task_name
#     def predict(self, text):
#         confidence = random.uniform(0.1, 0.2)
#         prediction = f"{self.task_name}_result"
#         return prediction, confidence

# 5) COMPLETE PROMPT ROUTING SYSTEM (now uses QLearningTaskClassifier + Transformer Domain CLS)

