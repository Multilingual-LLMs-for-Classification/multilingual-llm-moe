"""
Domain classification component using XLM-RoBERTa.

Transformer-based multilingual domain classifier with prototype ensembling
for stable predictions on short prompts.
"""

from pathlib import Path
from typing import Dict, List
from collections import Counter

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

# 2) DOMAIN CLASSIFICATION (Transformer-based, replaces TF-IDF version)
class _DomainDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int] | None, tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

class DomainClassifier(nn.Module):
    """
    Multilingual domain classifier using XLM-R embeddings + linear head.
    - Encoder frozen by default (fast). Optionally fine-tune with freeze_encoder=False.
    - Prototype ensembling at inference for more stable predictions on short prompts.
    """
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        model_dir: Path | None = None,
        max_len: int = 128,
        alpha_proto: float = 0.30,   # blend weight for prototype distribution
        proto_temp: float = 10.0     # softmax temperature over prototype sims
    ):
        super().__init__()
        self.model_dir = model_dir or (Path(__file__).parents[2] / "models" / "domain_xlmr")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Will update from training data if more domains exist
        self.domains: List[str] = ['finance', 'general']
        self.label2id = {d: i for i, d in enumerate(self.domains)}
        self.id2label = {i: d for d, i in self.label2id.items()}

        self.model_name = model_name
        self.max_len = max_len
        self.alpha_proto = alpha_proto
        self.proto_temp = proto_temp
        self.is_ready = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        hidden = self.encoder.config.hidden_size

        # Simple linear head
        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(hidden, len(self.domains))
        )

        # Prototypes (class means in embedding space)
        self.prototypes = torch.zeros((len(self.domains), hidden), dtype=torch.float32)

        self.device_ = torch.device(DEVICE)
        self.to(self.device_)

    # -------------------- Utilities -------------------- #
    def _freeze_encoder(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def _embed_cls(self, input_ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        CLS-like embedding (token 0) for a batch.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attn)
        return outputs.last_hidden_state[:, 0, :]  # [B, H]

    def _build_loaders(self, texts: List[str], labels: List[int], val_split=0.1, batch_size=32):
        idx = np.random.permutation(len(texts))
        n_val = int(len(texts) * val_split)
        val_idx = idx[:n_val]; tr_idx = idx[n_val:]
        x_tr = [texts[i] for i in tr_idx]; y_tr = [labels[i] for i in tr_idx]
        x_va = [texts[i] for i in val_idx]; y_va = [labels[i] for i in val_idx]

        ds_tr = _DomainDataset(x_tr, y_tr, self.tokenizer, self.max_len)
        ds_va = _DomainDataset(x_va, y_va, self.tokenizer, self.max_len)
        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE=="cuda")),
            DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))
        )

    @torch.no_grad()
    def _compute_prototypes(self, texts: List[str], labels: List[int], batch_size: int = 64):
        self.eval(); self.encoder.eval()
        hidden = self.encoder.config.hidden_size
        sums = torch.zeros((len(self.domains), hidden), device=self.device_)
        counts = torch.zeros((len(self.domains),), device=self.device_)
        loader = DataLoader(_DomainDataset(texts, labels, self.tokenizer, self.max_len),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))
        for batch in loader:
            inp = batch["input_ids"].to(self.device_)
            att = batch["attention_mask"].to(self.device_)
            emb = self._embed_cls(inp, att)  # [B, H]
            ys  = batch["labels"].to(self.device_)
            for c in range(len(self.domains)):
                mask = (ys == c)
                if mask.any():
                    sums[c] += emb[mask].sum(dim=0)
                    counts[c] += mask.sum()
        counts = counts.clamp(min=1.0)
        protos = sums / counts.unsqueeze(1)
        self.prototypes = protos.detach()

    # -------------------- Public API -------------------- #
    def fit_from_labeled_prompts(
        self,
        data: List[Dict],
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        val_split: float = 0.1,
        freeze_encoder: bool = True,
        class_weighting: bool = True
    ):
        """
        Train on [{'prompt','domain',...}, ...].
        """
        # Build labels from data
        doms = sorted({d['domain'] for d in data if 'domain' in d})
        self.domains = doms
        self.label2id = {d: i for i, d in enumerate(self.domains)}
        self.id2label = {i: d for d, i in self.label2id.items()}

        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains))).to(self.device_)
        self.prototypes = torch.zeros((len(self.domains), hidden), device=self.device_)

        texts = [d['prompt'] for d in data]
        labels = [self.label2id[d['domain']] for d in data]

        train_loader, val_loader = self._build_loaders(texts, labels, val_split, batch_size)
        self._freeze_encoder(freeze_encoder)

        # Class weights if imbalanced
        if class_weighting:
            counts = Counter(labels)
            weights = torch.tensor(
                [1.0 / max(1, counts[i]) for i in range(len(self.domains))],
                dtype=torch.float, device=self.device_
            )
            weights = weights / weights.mean()
        else:
            weights = torch.ones(len(self.domains), device=self.device_)

        criterion = nn.CrossEntropyLoss(weight=weights)
        if freeze_encoder:
            params = list(self.classifier.parameters())
        else:
            params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        optimizer = optim.AdamW(params, lr=lr)

        best_acc = 0.0
        for ep in range(1, epochs+1):
            # ---- Train ----
            self.train()
            running = 0.0; n = 0
            for batch in train_loader:
                inp = batch["input_ids"].to(self.device_)
                att = batch["attention_mask"].to(self.device_)
                y   = batch["labels"].to(self.device_)

                h = self._embed_cls(inp, att)
                logits = self.classifier(h)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                running += loss.item() * y.size(0)
                n += y.size(0)

            # ---- Eval ----
            self.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inp = batch["input_ids"].to(self.device_)
                    att = batch["attention_mask"].to(self.device_)
                    y   = batch["labels"].to(self.device_)
                    h = self._embed_cls(inp, att)
                    logits = self.classifier(h)
                    pred = logits.argmax(-1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = correct / max(1, total)
            print(f"[DomainCLS] epoch {ep}/{epochs} | train_loss={(running/max(1,n)):.4f} | val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                # Recompute prototypes on the full training set
                self._compute_prototypes(texts, labels, batch_size=batch_size)

        self.is_ready = True
        print(f"✅ Domain classifier training finished. Best val_acc={best_acc:.4f}")

    @torch.no_grad()
    def _proto_distribution(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B,H] embeddings. Returns proto-based distribution [B, C].
        """
        if self.prototypes is None or self.prototypes.numel() == 0:
            # uniform fallback
            return torch.full((h.size(0), len(self.domains)), 1.0/len(self.domains), device=h.device)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)  # [C,H]
        sims = torch.matmul(h_norm, p_norm.T)          # [B,C]
        return F.softmax(self.proto_temp * sims, dim=-1)

    @torch.no_grad()
    def get_domain_probabilities(self, text: str) -> Dict[str, float]:
        if not self.is_ready:
            # quick rule-based fallback if not trained/loaded
            return self._fallback_domain_probs(text)

        self.eval()
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device_)
        h = self._embed_cls(enc["input_ids"], enc["attention_mask"])
        logits = self.classifier(h)               # [1,C]
        p_head = F.softmax(logits, dim=-1)        # [1,C]
        p_proto = self._proto_distribution(h)     # [1,C]
        p = (1.0 - self.alpha_proto) * p_head + self.alpha_proto * p_proto
        p = p.squeeze(0).detach().cpu().tolist()
        return {self.id2label[i]: float(p[i]) for i in range(len(self.domains))}

    @torch.no_grad()
    def classify_domain(self, text: str) -> str:
        probs = self.get_domain_probabilities(text)
        return max(probs.items(), key=lambda kv: kv[1])[0]

    # --------------- Persistence --------------- #
    # def save_model(self, filepath: str | Path | None = None):
    #     self.model_dir.mkdir(parents=True, exist_ok=True)
    #     state = {
    #         "model_name": self.model_name,
    #         "domains": self.domains,
    #         "classifier": self.classifier.state_dict(),
    #         "prototypes": self.prototypes.detach().cpu().numpy(),
    #         "max_len": self.max_len,
    #         "alpha_proto": self.alpha_proto,
    #         "proto_temp": self.proto_temp,
    #         "is_ready": self.is_ready,
    #     }
    #     path = self.model_dir / "domain_cls.pt"
    #     torch.save(state, path)
    #     print(f"✅ Domain classifier saved to: {path}")
    
    def save_model(self, filepath: str | Path | None = None):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 2,  # new
            "model_name": self.model_name,
            "domains": self.domains,
            "classifier": self.classifier.state_dict(),
            # save as a tensor (no numpy) to avoid pickle needs
            "prototypes_t": self.prototypes.detach().cpu(),
            "max_len": self.max_len,
            "alpha_proto": self.alpha_proto,
            "proto_temp": self.proto_temp,
            "is_ready": self.is_ready,
        }
        path = self.model_dir / "domain_cls.pt"
        torch.save(state, path)
        print(f"✅ Domain classifier saved to: {path}")


    # def load_model(self, filepath: str | Path | None = None) -> bool:
    #     path = (filepath if filepath else self.model_dir / "domain_cls.pt")
    #     if not Path(path).exists():
    #         print(f"⚠️ No saved domain classifier at {path}")
    #         self.is_ready = False
    #         return False
    #     try:
    #         state = torch.load(path, map_location=self.device_, weights_only=True)
    #     except Exception:
    #         # Legacy checkpoints (e.g., with numpy arrays) need this:
    #         state = torch.load(path, map_location=self.device_, weights_only=False)
    #     self.model_name = state["model_name"]
    #     self.domains = state["domains"]
    #     self.label2id = {d: i for i, d in enumerate(self.domains)}
    #     self.id2label = {i: d for d, i in self.label2id.items()}
    #     self.max_len = state.get("max_len", self.max_len)
    #     self.alpha_proto = state.get("alpha_proto", self.alpha_proto)
    #     self.proto_temp = state.get("proto_temp", self.proto_temp)

    #     # Resize head & prototypes
    #     hidden = self.encoder.config.hidden_size
    #     self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains))).to(self.device_)
    #     self.classifier.load_state_dict(state["classifier"])
    #     self.prototypes = torch.tensor(state["prototypes"], dtype=torch.float32, device=self.device_)
    #     self.is_ready = bool(state.get("is_ready", True))
    #     print(f"✅ Domain classifier loaded from: {path}")
    #     return True
    
    def load_model(self, filepath: str | Path | None = None) -> bool:
        path = (filepath if filepath else self.model_dir / "domain_cls.pt")
        if not Path(path).exists():
            print(f"⚠️ No saved domain classifier at {path}")
            self.is_ready = False
            return False

        # PyTorch 2.6 compatibility: try safe first, then legacy
        try:
            state = torch.load(path, map_location=self.device_, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=self.device_, weights_only=False)

        self.model_name = state["model_name"]
        self.domains = state["domains"]
        self.label2id = {d: i for i, d in enumerate(self.domains)}
        self.id2label = {i: d for d, i in self.label2id.items()}
        self.max_len = state.get("max_len", self.max_len)
        self.alpha_proto = state.get("alpha_proto", self.alpha_proto)
        self.proto_temp = state.get("proto_temp", self.proto_temp)

        # Resize head & load weights
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains))).to(self.device_)
        self.classifier.load_state_dict(state["classifier"])

        # Backward compatibility for old checkpoints
        if "prototypes_t" in state:
            self.prototypes = state["prototypes_t"].to(self.device_)
        else:
            # legacy numpy-based
            self.prototypes = torch.as_tensor(state["prototypes"], dtype=torch.float32, device=self.device_)

        self.is_ready = bool(state.get("is_ready", True))
        print(f"✅ Domain classifier loaded from: {path}")
        return True


    # --------------- Simple fallback --------------- #
    def _fallback_domain_probs(self, text: str) -> Dict[str, float]:
        domain_keywords = {
            'finance': ['market','stock','price','investment','trading','portfolio','risk','return',
                        'bank','money','revenue','profit','analysis','economic','financial'],
            'general': ['help','question','what','how','why','when','where','explain','summary']
        }
        text_lower = text.lower()
        scores = {d: sum(1 for kw in kws if kw in text_lower)
                  for d, kws in domain_keywords.items()}
        # normalize
        total = sum(scores.values()) or 1
        return {d: s/total for d, s in scores.items()}

# 3) TASK CLASSIFICATION with Q-LEARNING
