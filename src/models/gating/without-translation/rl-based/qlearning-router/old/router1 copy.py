import os
import sys
import csv
import json
import random
from pathlib import Path
from typing import Dict, List
from collections import Counter

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fasttext
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging

project_root = Path(__file__).parents[6]
sys.path.insert(0, str(project_root))

from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader

hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)

# 1) LANGUAGE DETECTION (FastText-based with dynamic language loading)
class LanguageDetector:
    def __init__(self, registry_path: str | Path = None):
        self.model_path = Path(__file__).parent.parent / "models" / "lid.176.bin"
        self.model = None
        self._load_fasttext_model()

        # Default language mapping (FastText label -> full name)
        self._default_language_mapping = {
            '__label__de': 'german',
            '__label__en': 'english',
            '__label__es': 'spanish',
            '__label__fr': 'french',
            '__label__ja': 'japanese',
            '__label__zh': 'chinese'
        }

        # Load supported languages from registry if provided
        self.registry_path = registry_path
        self.supported_languages_by_task = {}
        self.all_supported_languages = set()

        if registry_path:
            self._load_languages_from_registry()

        # Use default mapping if no registry or no languages loaded
        if not self.all_supported_languages:
            self.all_supported_languages = set(self._default_language_mapping.values())

        self.language_mapping = self._default_language_mapping

    def _load_languages_from_registry(self):
        """Load supported languages from experts registry"""
        try:
            registry_file = Path(self.registry_path)
            if not registry_file.is_absolute():
                registry_file = Path(__file__).parents[4] / self.registry_path

            with open(registry_file, 'r') as f:
                registry = json.load(f)

            # Extract supported languages from each task
            tasks = registry.get("tasks", {})
            for task_key, task_config in tasks.items():
                supported_langs = task_config.get("supported_languages", [])
                self.supported_languages_by_task[task_key] = supported_langs

                # Add to global set (convert short codes to full names)
                for lang_code in supported_langs:
                    lang_full = self._code_to_full_name(lang_code)
                    self.all_supported_languages.add(lang_full)

            print(f"✅ Loaded language support from registry:")
            for task, langs in self.supported_languages_by_task.items():
                print(f"   {task}: {langs}")
            print(f"   All supported languages: {sorted(self.all_supported_languages)}")

        except Exception as e:
            print(f"⚠️ Could not load languages from registry: {e}")
            print(f"   Using default language mapping")

    def _code_to_full_name(self, code: str) -> str:
        """Convert language code to full name (e.g., 'en' -> 'english')"""
        code_mapping = {
            'de': 'german',
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'ja': 'japanese',
            'zh': 'chinese'
        }
        return code_mapping.get(code.lower(), code)

    def get_supported_languages_for_task(self, domain: str, task: str) -> list[str]:
        """Get list of supported languages for a specific task"""
        task_key = f"{domain}/{task}"
        return self.supported_languages_by_task.get(task_key, list(self.all_supported_languages))
    
    def _load_fasttext_model(self):
        try:
            if not self.model_path.exists():
                self._download_fasttext_model()
            self.model = fasttext.load_model(str(self.model_path))
            print("✅ FastText language model loaded")
        except Exception as e:
            print(f"⚠️ FastText model failed, using fallback: {e}")
            self.model = None
    
    def _download_fasttext_model(self):
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading FastText model...")
        response = requests.get(url)
        with open(self.model_path, 'wb') as f:
            f.write(response.content)
        print("✅ FastText model downloaded")
    
    def detect_language(self, text):
        if self.model is None:
            return self._fallback_detection(text)
        try:
            cleaned_text = text.replace('\n', ' ').strip()
            if len(cleaned_text) < 3:
                return 'english'
            labels, scores = self.model.predict(cleaned_text, k=1)
            detected_lang = labels[0]
            mapped_lang = self.language_mapping.get(detected_lang, 'english')
            return mapped_lang
        except Exception:
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text):
        patterns = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'you', 'this'],
            'german': ['der', 'die', 'das', 'und', 'ist', 'ich', 'nicht', 'ein', 'eine', 'zu', 'den', 'von'],
            'spanish': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'por'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans'],
            'japanese': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる'],
            'chinese': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上']
        }
        text_lower = text.lower()
        scores = {}
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            scores['chinese'] = len([char for char in text if '\u4e00' <= char <= '\u9fff'])
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            scores['japanese'] = len([char for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'])
        text_words = text_lower.split()
        for lang, keywords in patterns.items():
            if lang in ['japanese', 'chinese']:
                continue
            scores[lang] = sum(1 for w in text_words if w in keywords)
        return max(scores, key=scores.get) if any(scores.values()) else 'english'

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
        self.model_dir = model_dir or (Path(__file__).parent.parent / "models" / "domain_xlmr")
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
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
                enc_name = cfg.get("encoder_name", self.encoder_name)
                if enc_name != self.encoder_name:
                    print(f"ℹ️ Stored encoder '{enc_name}' differs from requested '{self.encoder_name}'. Using stored name.")
                    self.encoder = TransformersEncoder(enc_name).to(self.device)
                    self.encoder_name = enc_name
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

from src.models.experts.llms.task_expert import TaskExpert, TaskExpertConfig
from src.models.experts.llms.expert_pool import LLMAdapterPool

class PromptRoutingSystem:
    def __init__(self):
        config_path = Path(__file__).parents[4] / "experts" / "config"
        self.expert_registry_path = config_path / "experts_registry.json"

        # Initialize language detector with registry path
        self.language_detector = LanguageDetector(registry_path=self.expert_registry_path)
        self.domain_classifier = DomainClassifier(
            model_name="xlm-roberta-base",
            model_dir=Path(__file__).parent.parent / "models" / "domain_xlmr",
            max_len=128,
            alpha_proto=0.30,
            proto_temp=10.0
        )
        self.model_loader = ModelLoader(config_path / "model_config.json")
        self.domain_tasks_obj = DomainTaskLoader(config_path / "domain_tasks.json")
        self.domain_tasks = self.domain_tasks_obj.domain_tasks if hasattr(self.domain_tasks_obj, "domain_tasks") else self.domain_tasks_obj
        
        # Q-learning task classifier
        self.task_classifier = QLearningTaskClassifier(
            self.domain_tasks,
            model_dir=Path(__file__).parent.parent / "models" / "task_routers_qlearning",
            encoder_name="xlm-roberta-base",
            max_len=128,
            batch_size=16,
            lr=1e-5,
            epochs=1,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10000
        )
        # Try loading existing domain model & QRouters
        self.domain_classifier.load_model()
        self.task_classifier.load_models()
        
        # Download any external models if needed
        print("Checking and downloading models if needed...")
        self.model_loader.download_all_models()

        self.expert_pool = LLMAdapterPool(self.expert_registry_path)
        
        # Instantiate experts per domain/task using the registry
        self.experts = {}
        for domain, tasks in self.domain_tasks.items():
            self.experts[domain] = {}
            for task in tasks.keys():
                self.experts[domain][task] = TaskExpert(
                    TaskExpertConfig(
                        domain=domain,
                        task=task,
                        registry_path=str(self.expert_registry_path),
                        generation=None  # or per-task overrides dict
                    ),
                    pool=self.expert_pool
                )
    
    def save_all_models(self):
        print("💾 Saving all models...")
        self.domain_classifier.save_model()
        self.task_classifier.save_models()
        print("✅ All models saved successfully!")

    def train_domain_classifier(self, training_data: List[Dict], **kwargs):
        """
        Train the transformer domain classifier on labeled prompts.
        kwargs are passed to fit_from_labeled_prompts (epochs, batch_size, lr, freeze_encoder, ...).
        """
        print("Training Domain Classifier (Transformer)...")
        self.domain_classifier.fit_from_labeled_prompts(training_data, **kwargs)
        self.domain_classifier.save_model()
    
    def train_q_routers(self, training_data: List[Dict]):
        """Train per-domain QRouters on labeled (domain, task, prompt) items."""
        self.task_classifier.train(training_data, val_split=0.1)
        self.task_classifier.save_models()
    
    def route_prompt(self, prompt: str, classification_text: str, review_title: str) -> Dict:

        language = self.language_detector.detect_language(prompt)
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        task = self.task_classifier.classify_task(prompt, domain)
        expert = self.experts[domain][task]

        result, expert_confidence, raw_response = expert.predict(
            classification_text,review_title,
            prompt,
            language
        )
        output = {
            'input': prompt,
            'language': language,
            'domain': domain,
            'domain_probabilities': domain_probs,
            'task': task,
            'result': result,
            'expert_confidence': expert_confidence,
            'routing_path': f"{language} → {domain} → {task}",
            'raw_response': raw_response
        }
        return output
    def get_system_stats(self):
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.values())
        all_languages = self.language_detector.all_supported_languages
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': len(all_languages),
            'all_languages': sorted(all_languages),
            'languages_by_task': self.language_detector.supported_languages_by_task,
            'domains': list(self.domain_tasks.keys())
        }

# 7) MAIN: wiring + evaluation (as in your harness)
def _pct(x): return f"{x*100:6.2f}%"

def _compute_prf_bal_kappa(cm: Counter, labels: List[str]):
    support = {c: 0 for c in labels}
    pred_tot = {c: 0 for c in labels}
    tp = {c: cm[(c, c)] for c in labels}
    for g in labels:
        support[g] = sum(cm[(g, p)] for p in labels)
    for p in labels:
        pred_tot[p] = sum(cm[(g, p)] for g in labels)
    total = sum(support.values()) if support else 0
    precisions, recalls, f1s, weights, recalls_only = [], [], [], [], []
    micro_tp = sum(tp.values())
    micro_fp = sum(pred_tot[c] - tp[c] for c in labels)
    micro_fn = sum(support[c] - tp[c] for c in labels)
    for c in labels:
        p = tp[c] / pred_tot[c] if pred_tot[c] > 0 else 0.0
        r = tp[c] / support[c]  if support[c]  > 0 else 0.0
        f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f); recalls_only.append(r)
        weights.append(support[c] / total if total else 0.0)
    macro_p = sum(precisions)/len(labels) if labels else 0.0
    macro_r = sum(recalls)/len(labels)    if labels else 0.0
    macro_f1 = sum(f1s)/len(labels)       if labels else 0.0
    weighted_f1 = sum(w*f for w, f in zip(weights, f1s)) if labels else 0.0
    balanced_acc = sum(recalls_only)/len(labels) if labels else 0.0
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p + micro_r) > 0 else 0.0
    accuracy = micro_tp / total if total else 0.0
    pe = sum((support[c]/total) * (pred_tot[c]/total) for c in labels) if total else 0.0
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1,
        'weighted_f1': weighted_f1, 'balanced_acc': balanced_acc, 'kappa': kappa,
        'support': support, 'pred_tot': pred_tot, 'total': total,
    }

def _print_confusion(cm: Counter, labels: List[str], title: str):
    print(title)
    if not labels:
        print("  (no labels)\n"); return
    header = "      " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    for gt in labels:
        row = [f"{gt:>6}"]
        for pr in labels:
            row.append(f"{cm[(gt, pr)]:>22}")
        print(" ".join(row))
    print()
# Helper function to determine which expert/model was used
def _get_expert_used(language: str, domain: str, task: str, registry_path: str = "experts/config/experts_registry.json") -> str:
    """
    Determine which base model/expert was used based on task+language combination.
    Queries experts_registry.json to find the correct base_model_key.

    This mirrors the logic in LLMAdapterPool._resolve_base_model_for_language()
    to determine which model was actually selected during routing.

    Args:
        language: Detected language (e.g., "english", "japanese")
        domain: Detected domain (e.g., "finance")
        task: Detected task (e.g., "rating", "news")
        registry_path: Path to experts_registry.json

    Returns:
        base_model_key (e.g., "llama-2-7b-hf", "aya-23", "google/gemma-7b")
    """
    # Construct task_key (e.g., "finance/rating")
    task_key = f"{domain}/{task}"

    # Load registry
    try:
        registry_file = Path(registry_path)
        if not registry_file.is_absolute():
            # Make it relative to this file's location
            registry_file = Path(__file__).parents[4] / registry_path

        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load registry from {registry_path}: {e}")
        return "unknown"

    # Get task config
    tcfg = registry.get("tasks", {}).get(task_key)
    if not tcfg:
        return "unknown"

    default_base = tcfg.get("base_model_key", "default")

    # Check if task has language_mapping
    lang_mapping = tcfg.get("language_mapping")
    if not lang_mapping:
        # No language mapping - uses default model for all languages
        return default_base

    # Normalize language
    lang_normalized = language.lower() if language else ""

    # Priority 1: Check for direct per-language mapping
    if lang_normalized in lang_mapping:
        lang_cfg = lang_mapping[lang_normalized]
        # Check if it's a per-language entry (no "languages" key)
        if "languages" not in lang_cfg:
            return lang_cfg.get("base_model_key", default_base)

    # Priority 2: Find which language group this language belongs to
    for group_name, group_cfg in lang_mapping.items():
        languages = group_cfg.get("languages", [])
        if lang_normalized in languages:
            # Found the language group - return its base_model_key
            return group_cfg.get("base_model_key", default_base)

    # Priority 3: Language not found in any group - use default
    return default_base

def _print_expert_selection_summary(per_lang_total: Counter, per_lang_expert: Dict[str, str]):
    """Print which expert/model was selected for each language"""
    print("\nEXPERT/MODEL SELECTION BY LANGUAGE")
    print("=" * 80)

    # Group by expert
    expert_langs = {}
    for lang, expert in per_lang_expert.items():
        if expert not in expert_langs:
            expert_langs[expert] = []
        expert_langs[expert].append(lang)

    # Define language groups
    groups = {
        "llama-2-7b-hf": ("European", ["english", "german", "spanish", "french"]),
        "aya-23": ("Asian", ["japanese", "chinese"])
    }

    total_samples = sum(per_lang_total.values())

    for expert, (group_name, expected_langs) in groups.items():
        if expert in expert_langs:
            print(f"\nLanguage Group: {group_name} ({expert})")
            print("-" * 80)

            group_total = 0
            for lang in sorted(expected_langs):
                if lang in per_lang_total:
                    count = per_lang_total[lang]
                    print(f"  {lang:<12} : {count:>4} samples")
                    group_total += count

            pct = (group_total / total_samples * 100) if total_samples > 0 else 0
            print(f"  {'Total':<12} : {group_total:>4} samples ({pct:.1f}%)")

    print("\n" + "=" * 80 + "\n")


def _print_expert_performance(per_expert_cm: Dict[str, Counter],
                               per_lang_total: Counter,
                               per_lang_expert: Dict[str, str],
                               per_lang_correct: Counter,
                               expert_labels: List[str]):
    """Print performance metrics broken down by expert/model"""
    print("\nPERFORMANCE BY EXPERT/MODEL")
    print("=" * 80)

    # Group languages by expert
    expert_to_langs = {}
    for lang, expert in per_lang_expert.items():
        if expert not in expert_to_langs:
            expert_to_langs[expert] = []
        expert_to_langs[expert].append(lang)

    for expert in sorted(per_expert_cm.keys()):
        langs = sorted(expert_to_langs.get(expert, []))

        if not langs:
            continue

        print(f"\n{expert} (Languages: {', '.join(langs)})")
        print("-" * 80)

        # Calculate metrics for this expert
        cm = per_expert_cm[expert]
        metrics = _compute_prf_bal_kappa(cm, expert_labels)

        # Total samples for this expert
        total = sum(per_lang_total[lang] for lang in langs)

        print(f"  Samples      : {total}")
        print(f"  Accuracy     : {_pct(metrics['accuracy'])}")
        print(f"  Macro F1     : {_pct(metrics['macro_f1'])}")
        print(f"  Weighted F1  : {_pct(metrics['weighted_f1'])}")

        # Per-language breakdown for this expert
        print(f"\n  Per-language performance:")
        for lang in langs:
            if lang in per_lang_total:
                n = per_lang_total[lang]
                correct = per_lang_correct.get(lang, 0)
                acc = correct / n if n > 0 else 0.0
                print(f"    {lang:<12} : Accuracy = {_pct(acc)}, Samples = {n}")

    print("\n" + "=" * 80 + "\n")


def _print_language_group_comparison(per_expert_cm: Dict[str, Counter],
                                      per_lang_expert: Dict[str, str],
                                      expert_labels: List[str]):
    """Compare European vs Asian language group performance"""
    print("\nLANGUAGE GROUP COMPARISON")
    print("=" * 80)

    # Calculate metrics for each group
    groups = {}
    for expert in ["llama-2-7b-hf", "aya-23"]:
        if expert in per_expert_cm:
            metrics = _compute_prf_bal_kappa(per_expert_cm[expert], expert_labels)
            groups[expert] = metrics

    if len(groups) == 2:
        print(f"\n{'Metric':<25} {'European (llama-2)':<20} {'Asian (aya-23)':<20} {'Difference':<15}")
        print("-" * 80)

        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('Macro F1', 'macro_f1'),
            ('Weighted F1', 'weighted_f1'),
            ('Balanced Accuracy', 'balanced_acc')
        ]

        for label, key in metrics_to_compare:
            euro_val = groups.get("llama-2-7b-hf", {}).get(key, 0.0)
            asian_val = groups.get("aya-23", {}).get(key, 0.0)
            diff = euro_val - asian_val

            print(f"{label:<25} {_pct(euro_val):<20} {_pct(asian_val):<20} {diff*100:+.1f}pp")

    print("\n" + "=" * 80 + "\n")


def _print_expert_confusion_matrices(per_expert_cm: Dict[str, Counter],
                                      expert_labels: List[str]):
    """Print separate confusion matrices for each expert"""
    print("\nCONFUSION MATRICES BY EXPERT/MODEL")
    print("=" * 80)

    for expert in sorted(per_expert_cm.keys()):
        cm = per_expert_cm[expert]
        _print_confusion(cm, expert_labels, title=f"\n{expert} Confusion Matrix (GT rows × Pred cols)")

    print("=" * 80 + "\n")

def load_prompts_from_csv(path="unified.csv"):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

if __name__ == "__main__":
    with open("test2_grouped_languages_flat.json", "r", encoding="utf-8") as f:
        test_prompts = json.load(f)

    TEST_N = 1020
    test_prompts = test_prompts[:TEST_N]
    # test_prompts = test_prompts[-TEST_N:]

    print("Initializing Prompt Routing System (Q-learning router + Transformer Domain CLS)...")
    system = PromptRoutingSystem()
    
    # Display system information
    stats = system.get_system_stats()
    print(f"System initialized with {stats['total_domains']} domains and {stats['total_tasks']} tasks")
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Available domains: {stats['domains']}")

    # Load training data
    with open("train1.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    # TRAIN_N = 1000
    # training_data = training_data[:TRAIN_N]

    # ---- Train Domain Classifier (Transformer) ----
    system.train_domain_classifier(
        training_data,
        epochs=1,
        batch_size=32,
        lr=2e-5,
        freeze_encoder=True,
        class_weighting=True
    )

    # ---- Train Q-learning task routers ----
    system.train_q_routers(training_data)
    
    system.save_all_models()
    
    # ----------- Evaluation on test_prompts -----------
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    print("=" * 70)
    
    domain_labels = sorted({item['domain'] for item in test_prompts if isinstance(item, dict) and 'domain' in item})
    task_labels   = sorted({item['task']   for item in test_prompts if isinstance(item, dict) and 'task'   in item})
    cm_domain = Counter()
    cm_task   = Counter()
    cm_expert = Counter()

    per_lang_total = Counter()
    per_lang_dom   = Counter()
    per_lang_task  = Counter()
    per_lang_exact = Counter()

    # NEW: Track expert/model usage
    per_expert_total = Counter()
    per_expert_correct = Counter()
    per_expert_cm = {}
    per_lang_expert = {}
    per_lang_correct = Counter()

    # CSV data collection
    csv_data = []

    for item in test_prompts:
        if not isinstance(item, dict):
            continue
        prompt      = item['prompt']
        review_title = item['review_title']
        text = item['classification_text']
        gt_domain = item['domain']
        gt_task   = item['task']
        gt_label  = item['label']

        print("Expected:", item['label'])

        result      = system.route_prompt(prompt, text, review_title)
        pred_domain = result['domain']
        pred_task   = result['task']
        lang_tag    = result.get('language', '?')
        pred_label  = result['result']
        raw_response = result.get('raw_response', '')

        # Collect CSV data
        csv_data.append({
            'review_title': review_title,
            'review_body': text[:100] + '...' if len(text) > 100 else text,
            'language': lang_tag,
            'expected_label': gt_label,
            'predicted_label': pred_label,
            'raw_response': raw_response
        })

        # NEW: Determine which expert was used
        expert_key = _get_expert_used(lang_tag, pred_domain, pred_task)

        # Track per-expert statistics
        per_expert_total[expert_key] += 1
        is_correct = (pred_label == gt_label)
        if is_correct:
            per_expert_correct[expert_key] += 1

        # Track per-expert confusion matrix
        if expert_key not in per_expert_cm:
            per_expert_cm[expert_key] = Counter()
        per_expert_cm[expert_key][(gt_label, pred_label)] += 1

        # Track which expert was used for each language
        per_lang_expert[lang_tag] = expert_key

        cm_domain[(gt_domain, pred_domain)] += 1
        cm_task[(gt_task, pred_task)]       += 1
        cm_expert[(gt_label, pred_label)] += 1

        per_lang_total[lang_tag] += 1
        per_lang_correct[lang_tag] += int(is_correct)
        dom_ok  = (pred_domain == gt_domain)
        task_ok = (pred_task   == gt_task)
        both_ok = dom_ok and task_ok
        per_lang_dom[lang_tag]   += int(dom_ok)
        per_lang_task[lg := lang_tag]  += int(task_ok)
        per_lang_exact[lang_tag] += int(both_ok)

    dom_metrics  = _compute_prf_bal_kappa(cm_domain, domain_labels)
    task_metrics = _compute_prf_bal_kappa(cm_task,   task_labels)
    expert_labels = sorted({item['label'] for item in test_prompts if isinstance(item, dict)})
    expert_metrics = _compute_prf_bal_kappa(cm_expert, expert_labels)

    print("\nADDITIONAL METRICS")
    print("=" * 80)
    print("Domain classification:")
    print(f"  Accuracy           : {_pct(dom_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {_pct(dom_metrics['macro_p'])} / {_pct(dom_metrics['macro_r'])} / {_pct(dom_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {_pct(dom_metrics['micro_p'])} / {_pct(dom_metrics['micro_r'])} / {_pct(dom_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {_pct(dom_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {_pct(dom_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {_pct(dom_metrics['kappa'])}")

    print("\nTask classification:")
    print(f"  Accuracy           : {_pct(task_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {_pct(task_metrics['macro_p'])} / {_pct(task_metrics['macro_r'])} / {_pct(task_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {_pct(task_metrics['micro_p'])} / {_pct(task_metrics['micro_r'])} / {_pct(task_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {_pct(task_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {_pct(task_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {_pct(task_metrics['kappa'])}")

    _print_confusion(cm_domain, domain_labels, title="\nDomain Confusion Matrix (GT rows × Pred cols)")
    _print_confusion(cm_task,   task_labels,   title="Task Confusion Matrix (GT rows × Pred cols)")
    print("Language list (detected):", sorted(per_lang_total))

    if per_lang_total:
        print("Per-language breakdown (using detected language):")
        print("-" * 80)
        print(f"{'lang':>6} | {'n':>4} | {'domain acc':>12} | {'task acc':>10} | {'exact acc':>10}")
        for lg in sorted(per_lang_total):
            n_l = per_lang_total[lg]
            dom_acc_l  = per_lang_dom[lg]   / n_l if n_l else 0.0
            task_acc_l = per_lang_task[lg]  / n_l if n_l else 0.0
            exact_l    = per_lang_exact[lg] / n_l if n_l else 0.0
            print(f"{lg:>6} | {n_l:>4} | {_pct(dom_acc_l):>12} | {_pct(task_acc_l):>10} | {_pct(exact_l):>10}")

    print("\nExpert (final output) classification:")
    print(f"  Accuracy           : {_pct(expert_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {_pct(expert_metrics['macro_p'])} / {_pct(expert_metrics['macro_r'])} / {_pct(expert_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {_pct(expert_metrics['micro_p'])} / {_pct(expert_metrics['micro_r'])} / {_pct(expert_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {_pct(expert_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {_pct(expert_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {_pct(expert_metrics['kappa'])}")

    _print_confusion(cm_expert, expert_labels, title="Expert Output Confusion Matrix (GT rows × Pred cols)")

    # NEW: Add expert-based visualizations
    _print_expert_selection_summary(per_lang_total, per_lang_expert)
    _print_expert_performance(per_expert_cm, per_lang_total, per_lang_expert, per_lang_correct, expert_labels)
    _print_language_group_comparison(per_expert_cm, per_lang_expert, expert_labels)
    _print_expert_confusion_matrices(per_expert_cm, expert_labels)

    # Save CSV file with raw responses
    csv_output_path = "predictions_with_raw_responses.csv"
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['review_title', 'review_body', 'language', 'expected_label', 'predicted_label', 'raw_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"\n✅ CSV file saved to: {csv_output_path}")
    print(f"   Total rows: {len(csv_data)}")
