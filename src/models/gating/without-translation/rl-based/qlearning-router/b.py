# router_xlmr_task_eval.py
# Router up to task classification using XLM-R fine-tuning (no expert pool):
# - FastText LanguageDetector (with robust fallback)
# - XLM-R DomainClassifier (linear head + prototype ensembling)
# - XLM-R Task Classifier (fine-tuned; one head per domain)
# - Evaluation: domain/task metrics, confusion matrices, per-language breakdown

import os, sys, json, random, requests
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
random.seed(42); np.random.seed(42)

# ---- Project root on path (same as your previous scripts)
project_root = Path(__file__).parents[6]
sys.path.insert(0, str(project_root))

from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 1) Language Detection (FastText + fallback)
# =============================================================================
import fasttext

class LanguageDetector:
    def __init__(self):
        self.model_path = Path(__file__).parent.parent / "models" / "lid.176.bin"
        self.model = None
        self._load_fasttext_model()
        self.language_mapping = {
            '__label__de':'german','__label__en':'english','__label__es':'spanish',
            '__label__fr':'french','__label__ja':'japanese','__label__zh':'chinese'
        }
    def _load_fasttext_model(self):
        try:
            if not self.model_path.exists():
                self._download_fasttext_model()
            self.model = fasttext.load_model(str(self.model_path))
            print("✅ FastText language model loaded")
        except Exception as e:
            print(f"⚠️ FastText failed, using fallback: {e}")
            self.model = None
    def _download_fasttext_model(self):
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading FastText model…")
        r = requests.get(url)
        with open(self.model_path, "wb") as f: f.write(r.content)
        print("✅ FastText model downloaded")
    def detect_language(self, text: str) -> str:
        if not text or len(text.strip()) < 3: return "english"
        if self.model is None: return self._fallback_detection(text)
        try:
            labels, _ = self.model.predict(text.replace("\n"," ").strip(), k=1)
            return self.language_mapping.get(labels[0], "english")
        except Exception:
            return self._fallback_detection(text)
    def _fallback_detection(self, text: str) -> str:
        patterns = {
            'english':['the','and','is','in','to','of','a','that','it','with','for','you','this'],
            'german' :['der','die','das','und','ist','ich','nicht','ein','eine','zu','den','von'],
            'spanish':['el','la','de','que','y','a','en','un','es','se','no','te','lo','por'],
            'french' :['le','de','et','à','un','il','être','en','avoir','que','pour','dans'],
            'japanese':['の','に','は','を','た','が','で','て','と','し','れ','さ','ある','いる'],
            'chinese' :['的','一','是','在','不','了','有','和','人','这','中','大','为','上'],
        }
        t = text.lower(); scores = {}
        if any('\u4e00' <= ch <= '\u9fff' for ch in text):
            scores['chinese'] = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        if any('\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff' for ch in text):
            scores['japanese'] = sum(1 for ch in text if '\u3040' <= ch <= '\u309f' or '\u30a0' <= ch <= '\u30ff')
        words = t.split()
        for lang, kws in patterns.items():
            if lang in ('japanese','chinese'): continue
            scores[lang] = sum(1 for w in words if w in kws)
        return max(scores, key=scores.get) if any(scores.values()) else "english"

# =============================================================================
# 2) Domain Classification (XLM-R + linear head + prototypes)
# =============================================================================
class _DomainDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int] | None, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], max_length=self.max_len, truncation=True,
                             padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None: item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

class DomainClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", model_dir: Path | None = None, max_len=128,
                 alpha_proto=0.30, proto_temp=10.0):
        super().__init__()
        self.model_dir = model_dir or (Path(__file__).parent.parent / "models" / "domain_xlmr")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.domains = ['finance','general']
        self.label2id = {d:i for i,d in enumerate(self.domains)}
        self.id2label = {i:d for d,i in self.label2id.items()}
        self.model_name, self.max_len = model_name, max_len
        self.alpha_proto, self.proto_temp = alpha_proto, proto_temp
        self.is_ready = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains)))
        self.prototypes = torch.zeros((len(self.domains), hidden), dtype=torch.float32)
        self.device_ = torch.device(DEVICE); self.to(self.device_)

    def _freeze_encoder(self, freeze=True):
        for p in self.encoder.parameters(): p.requires_grad = not freeze
    def _embed_cls(self, ids, att):
        out = self.encoder(input_ids=ids, attention_mask=att)
        return out.last_hidden_state[:,0,:]
    def _build_loaders(self, texts, labels, val_split=0.1, batch_size=32):
        idx = np.random.permutation(len(texts)); n_val = int(len(texts)*val_split)
        x_va=[texts[i] for i in idx[:n_val]]; y_va=[labels[i] for i in idx[:n_val]]
        x_tr=[texts[i] for i in idx[n_val:]]; y_tr=[labels[i] for i in idx[n_val:]]
        ds_tr=_DomainDataset(x_tr,y_tr,self.tokenizer,self.max_len)
        ds_va=_DomainDataset(x_va,y_va,self.tokenizer,self.max_len)
        pin=(DEVICE=="cuda")
        return (DataLoader(ds_tr,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=pin),
                DataLoader(ds_va,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=pin))
    @torch.no_grad()
    def _compute_prototypes(self, texts, labels, batch_size=64):
        self.eval(); self.encoder.eval()
        hidden = self.encoder.config.hidden_size
        sums = torch.zeros((len(self.domains), hidden), device=self.device_)
        counts = torch.zeros((len(self.domains),), device=self.device_)
        pin=(DEVICE=="cuda")
        ld = DataLoader(_DomainDataset(texts, labels, self.tokenizer, self.max_len),
                        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
        for b in ld:
            ids=b["input_ids"].to(self.device_); att=b["attention_mask"].to(self.device_)
            emb=self._embed_cls(ids,att); ys=b["labels"].to(self.device_)
            for c in range(len(self.domains)):
                m=(ys==c)
                if m.any(): sums[c]+=emb[m].sum(0); counts[c]+=m.sum()
        counts = counts.clamp(min=1.0); self.prototypes = (sums / counts.unsqueeze(1)).detach()
    def fit_from_labeled_prompts(self, data: List[Dict], epochs=1, batch_size=32, lr=2e-5,
                                 val_split=0.1, freeze_encoder=True, class_weighting=True):
        doms = sorted({d['domain'] for d in data if 'domain' in d})
        self.domains = doms; self.label2id = {d:i for i,d in enumerate(self.domains)}
        self.id2label = {i:d for d,i in self.label2id.items()}
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains))).to(self.device_)
        self.prototypes = torch.zeros((len(self.domains), hidden), device=self.device_)
        texts = [d['prompt'] for d in data]; labels = [self.label2id[d['domain']] for d in data]
        tr, va = self._build_loaders(texts, labels, val_split, batch_size)
        self._freeze_encoder(freeze_encoder)
        if class_weighting:
            cnt=Counter(labels); w=torch.tensor([1.0/max(1,cnt[i]) for i in range(len(self.domains))],
                                               dtype=torch.float, device=self.device_); w=w/w.mean()
        else:
            w=torch.ones(len(self.domains), device=self.device_)
        crit=nn.CrossEntropyLoss(weight=w)
        params = (list(self.classifier.parameters()) if freeze_encoder
                  else list(self.encoder.parameters())+list(self.classifier.parameters()))
        opt=optim.AdamW(params, lr=lr)
        best=0.0
        for ep in range(1,epochs+1):
            self.train(); tot=0.0; n=0
            for b in tr:
                ids=b["input_ids"].to(self.device_); att=b["attention_mask"].to(self.device_); y=b["labels"].to(self.device_)
                h=self._embed_cls(ids,att); logits=self.classifier(h); loss=crit(logits,y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.0); opt.step()
                tot+=loss.item()*y.size(0); n+=y.size(0)
            self.eval(); corr=0; total=0
            with torch.no_grad():
                for b in va:
                    ids=b["input_ids"].to(self.device_); att=b["attention_mask"].to(self.device_); y=b["labels"].to(self.device_)
                    h=self._embed_cls(ids,att); logits=self.classifier(h); pred=logits.argmax(-1)
                    corr+=(pred==y).sum().item(); total+=y.size(0)
            acc=corr/max(1,total)
            print(f"[DomainCLS] epoch {ep}/{epochs} | train_loss={(tot/max(1,n)):.4f} | val_acc={acc:.4f}")
            if acc>best: best=acc; self._compute_prototypes(texts, labels, batch_size=batch_size)
        self.is_ready=True; print(f"✅ Domain classifier training finished. Best val_acc={best:.4f}")
    @torch.no_grad()
    def _proto_distribution(self, h):
        if self.prototypes is None or self.prototypes.numel()==0:
            return torch.full((h.size(0), len(self.domains)), 1.0/len(self.domains), device=h.device)
        hN=F.normalize(h,dim=-1); pN=F.normalize(self.prototypes, dim=-1); sims=hN @ pN.T
        return F.softmax(self.proto_temp * sims, dim=-1)
    @torch.no_grad()
    def get_domain_probabilities(self, text: str) -> Dict[str,float]:
        if not self.is_ready:
            # rule fallback
            dkw={'finance':['market','stock','price','investment','trading','portfolio','risk','return','bank','money','revenue','profit','analysis','economic','financial'],
                 'general':['help','question','what','how','why','when','where','explain','summary']}
            t=text.lower(); sc={d:sum(1 for k in kws if k in t) for d,kws in dkw.items()}
            tot=sum(sc.values()) or 1; return {d:sc[d]/tot for d in self.domains}
        enc=self.tokenizer(text,max_length=self.max_len,truncation=True,padding="max_length",return_tensors="pt").to(self.device_)
        h=self._embed_cls(enc["input_ids"], enc["attention_mask"])
        p_head=F.softmax(self.classifier(h), dim=-1)
        p_proto=self._proto_distribution(h)
        p=((1-self.alpha_proto)*p_head + self.alpha_proto*p_proto).squeeze(0).cpu().tolist()
        return {self.id2label[i]: float(p[i]) for i in range(len(self.domains))}
    @torch.no_grad()
    def classify_domain(self, text: str) -> str:
        probs=self.get_domain_probabilities(text)
        return max(probs.items(), key=lambda kv: kv[1])[0]
    def save_model(self, filepath: str | Path | None = None):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        state={"format_version":2,"model_name":self.model_name,"domains":self.domains,
               "classifier":self.classifier.state_dict(),"prototypes_t":self.prototypes.detach().cpu(),
               "max_len":self.max_len,"alpha_proto":self.alpha_proto,"proto_temp":self.proto_temp,"is_ready":self.is_ready}
        path=self.model_dir/"domain_cls.pt"; torch.save(state, path); print(f"✅ Domain classifier saved to: {path}")
    def load_model(self, filepath: str | Path | None = None) -> bool:
        path=(filepath if filepath else self.model_dir/"domain_cls.pt")
        if not Path(path).exists(): print(f"⚠️ No saved domain classifier at {path}"); self.is_ready=False; return False
        try: state=torch.load(path, map_location=self.device_, weights_only=True)
        except Exception: state=torch.load(path, map_location=self.device_, weights_only=False)
        self.model_name=state["model_name"]; self.domains=state["domains"]
        self.label2id={d:i for i,d in enumerate(self.domains)}; self.id2label={i:d for d,i in self.label2id.items()}
        self.max_len=state.get("max_len", self.max_len); self.alpha_proto=state.get("alpha_proto", self.alpha_proto)
        self.proto_temp=state.get("proto_temp", self.proto_temp)
        hidden=self.encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Dropout(0.20), nn.Linear(hidden, len(self.domains))).to(self.device_)
        self.classifier.load_state_dict(state["classifier"])
        self.prototypes=(state["prototypes_t"].to(self.device_) if "prototypes_t" in state
                         else torch.as_tensor(state.get("prototypes", torch.zeros_like(self.prototypes.cpu())), dtype=torch.float32, device=self.device_))
        self.is_ready=bool(state.get("is_ready", True))
        print(f"✅ Domain classifier loaded from: {path}"); return True

# =============================================================================
# 3) Task Classification with XLM-R fine-tuning (per-domain heads)
# =============================================================================
class TaskDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, task2id: Dict[str,int], max_len=128):
        self.items=[it for it in items if it.get("task") in task2id]
        self.tokenizer=tokenizer; self.task2id=task2id; self.max_len=max_len
        self.texts=[it["prompt"] for it in self.items]
        self.labels=torch.tensor([task2id[it["task"]] for it in self.items], dtype=torch.long)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        enc=self.tokenizer(self.texts[i], padding="max_length", truncation=True,
                           max_length=self.max_len, return_tensors="pt")
        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), self.labels[i])

class XLMRTaskClassifier(nn.Module):
    """
    One shared XLM-R encoder + one linear classification head per domain.
    Fine-tunes the encoder (configurable) and the domain-specific head(s).
    """
    def __init__(self, domain_tasks, model_dir: Path = None, encoder_name="xlm-roberta-base",
                 max_len=128, batch_size=16, lr=2e-5, epochs=1, freeze_encoder=False,
                 class_weighting=True, val_split=0.1):
        super().__init__()
        if hasattr(domain_tasks, "domain_tasks"): domain_tasks=domain_tasks.domain_tasks
        self.domain_tasks: Dict[str, Dict[str, dict]] = domain_tasks

        self.device = torch.device(DEVICE)
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden = self.encoder.config.hidden_size

        # One head per domain
        self.heads = nn.ModuleDict()
        self.task2id: Dict[str, Dict[str,int]] = {}
        self.id2task: Dict[str, Dict[int,str]] = {}
        for dom, tasks in self.domain_tasks.items():
            names = list(tasks.keys())
            t2i = {t:i for i,t in enumerate(names)}
            i2t = {i:t for t,i in t2i.items()}
            self.task2id[dom] = t2i; self.id2task[dom] = i2t
            self.heads[dom] = nn.Linear(self.hidden, len(names))

        self.max_len=max_len; self.batch_size=batch_size; self.lr=lr; self.epochs=epochs
        self.val_split=val_split; self.freeze_encoder = freeze_encoder
        # persistence
        self.model_dir = model_dir or (Path(__file__).parent.parent / "models" / "task_classifiers_xlmr")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.to(self.device)

        if self.freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False

        self.class_weighting = class_weighting

    def _embed_cls(self, ids, att):
        out = self.encoder(input_ids=ids, attention_mask=att)
        return out.last_hidden_state[:,0,:]  # CLS token

    def _build_loaders(self, items: List[Dict], dom: str):
        ds_all = TaskDataset(items, self.tokenizer, self.task2id[dom], max_len=self.max_len)
        if len(ds_all)==0:
            return None, None
        idx = np.random.permutation(len(ds_all))
        n_val = int(self.val_split * len(ds_all))
        val_idx = idx[:n_val]; tr_idx = idx[n_val:]

        def subset(ds, ids):
            texts = [ds.texts[i] for i in ids]
            labels = [ds.labels[i].item() for i in ids]
            enc = self.tokenizer(texts, padding="max_length", truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            ids_t = enc["input_ids"]; att_t = enc["attention_mask"]
            return torch.utils.data.TensorDataset(ids_t, att_t, torch.tensor(labels, dtype=torch.long))

        ds_tr = subset(ds_all, tr_idx) if len(tr_idx)>0 else None
        ds_va = subset(ds_all, val_idx) if len(val_idx)>0 else None

        pin = (DEVICE=="cuda")
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True,  num_workers=2, pin_memory=pin) if ds_tr else None
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=pin) if ds_va else None
        return dl_tr, dl_va

    def train_all(self, training_data: List[Dict]):
        """
        Fine-tunes XLM-R + per-domain heads on labeled (prompt, domain, task).
        """
        params = list(self.heads.parameters())
        if not self.freeze_encoder:
            params += list(self.encoder.parameters())
        optimizer = optim.AdamW(params, lr=self.lr)

        for dom in self.domain_tasks.keys():
            dom_items=[d for d in training_data if d.get("domain")==dom and d.get("task") in self.task2id[dom]]
            if not dom_items:
                print(f"⚠️ No labeled items for domain '{dom}', skipping.")
                continue

            dl_tr, dl_va = self._build_loaders(dom_items, dom)
            if dl_tr is None:
                print(f"⚠️ Domain '{dom}': empty train split, skipping.")
                continue

            # optional class weights
            if self.class_weighting:
                cnt = Counter([d["task"] for d in dom_items])
                w = torch.tensor([1.0/max(1,cnt[self.id2task[dom][i]]) for i in range(len(self.task2id[dom]))],
                                 dtype=torch.float, device=self.device)
                w = w / w.mean()
            else:
                w = torch.ones(len(self.task2id[dom]), device=self.device)

            crit = nn.CrossEntropyLoss(weight=w)

            print(f"[TaskCLS:{dom}] training on {len(dom_items)} samples, "
                  f"tasks={len(self.task2id[dom])}, freeze_encoder={self.freeze_encoder}")

            for ep in range(1, self.epochs+1):
                # train
                self.train()
                tot=0.0; n=0
                for ids, att, y in dl_tr:
                    ids=ids.to(self.device); att=att.to(self.device); y=y.to(self.device)
                    h=self._embed_cls(ids, att)
                    logits=self.heads[dom](h)
                    loss=crit(logits,y)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    tot+=loss.item()*y.size(0); n+=y.size(0)

                # val
                acc=0.0
                if dl_va is not None:
                    self.eval(); corr=0; total=0
                    with torch.no_grad():
                        for ids, att, y in dl_va:
                            ids=ids.to(self.device); att=att.to(self.device); y=y.to(self.device)
                            h=self._embed_cls(ids, att)
                            logits=self.heads[dom](h)
                            pred=logits.argmax(-1)
                            corr+=(pred==y).sum().item(); total+=y.size(0)
                    acc = corr / max(1,total)

                print(f"[TaskCLS:{dom}] epoch {ep}/{self.epochs} | train_loss={(tot/max(1,n)):.4f} | val_acc={acc:.4f}")

        print("✅ Task classifiers fine-tuning finished.")

    @torch.no_grad()
    def classify_task(self, text: str, domain: str) -> str:
        if domain not in self.heads or len(self.task2id.get(domain,{}))==0:
            names=list(self.domain_tasks.get(domain,{}).keys())
            return names[0] if names else "unknown"
        self.eval()
        enc=self.tokenizer(text, padding="max_length", truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
        h=self._embed_cls(enc["input_ids"], enc["attention_mask"])
        logits=self.heads[domain](h)
        k=int(logits.argmax(-1).item())
        return self.id2task[domain].get(k,"unknown")

    # persistence
    def save_models(self):
        enc_path = self.model_dir/"encoder.pth"
        torch.save(self.encoder.state_dict(), enc_path)
        heads_path = self.model_dir/"heads.pth"
        torch.save(self.heads.state_dict(), heads_path)
        with open(self.model_dir/"task_config.json","w", encoding="utf-8") as f:
            json.dump({
                "encoder_name": self.encoder_name,
                "task2id": self.task2id,
                "id2task": self.id2task
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Task classifiers saved to: {self.model_dir}")

    def load_models(self)->bool:
        cfg = self.model_dir/"task_config.json"
        enc = self.model_dir/"encoder.pth"
        heads = self.model_dir/"heads.pth"
        ok=True
        if cfg.exists():
            with open(cfg, "r", encoding="utf-8") as f:
                cfgj=json.load(f)
            self.encoder_name = cfgj.get("encoder_name", self.encoder_name)
            # If stored encoder differs, reinstantiate to stored name
            if self.encoder_name != self.tokenizer.name_or_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
                self.encoder = AutoModel.from_pretrained(self.encoder_name).to(self.device)
                self.hidden = self.encoder.config.hidden_size
            # restore mappings
            self.task2id = {dom:{k:int(v) for k,v in d.items()} for dom,d in cfgj.get("task2id", {}).items()}
            self.id2task = {dom:{int(k):v for k,v in d.items()} for dom,d in cfgj.get("id2task", {}).items()}
            # rebuild heads to the right sizes
            new_heads = nn.ModuleDict()
            for dom, mapping in self.task2id.items():
                new_heads[dom] = nn.Linear(self.hidden, len(mapping))
            self.heads = new_heads.to(self.device)
        else:
            ok=False

        if enc.exists():
            self.encoder.load_state_dict(torch.load(enc, map_location=self.device, weights_only=False))
        else:
            ok=False

        if heads.exists():
            self.heads.load_state_dict(torch.load(heads, map_location=self.device, weights_only=False))
        else:
            ok=False

        print("✅ Task classifiers loaded." if ok else "⚠️ Task classifiers not fully found; will train new ones.")
        return ok

# =============================================================================
# 4) PromptRoutingSystem (no expert pool; stops at task)
# =============================================================================
class PromptRoutingSystem:
    def __init__(self):
        cfg = Path(__file__).parents[4] / "experts" / "config"
        self.language_detector = LanguageDetector()
        self.domain_classifier = DomainClassifier(
            model_name="xlm-roberta-base",
            model_dir=Path(__file__).parent.parent / "models" / "domain_xlmr",
            max_len=128, alpha_proto=0.30, proto_temp=10.0
        )
        self.model_loader = ModelLoader(cfg / "model_config.json")
        self.domain_tasks_obj = DomainTaskLoader(cfg / "domain_tasks.json")
        self.domain_tasks = (self.domain_tasks_obj.domain_tasks
                             if hasattr(self.domain_tasks_obj, "domain_tasks")
                             else self.domain_tasks_obj)
        self.task_classifier = XLMRTaskClassifier(
            self.domain_tasks,
            model_dir=Path(__file__).parent.parent / "models" / "task_classifiers_xlmr",
            encoder_name="xlm-roberta-base",
            max_len=128, batch_size=16, lr=2e-5, epochs=1,
            freeze_encoder=False, class_weighting=True, val_split=0.1
        )
        # load any existing models
        self.domain_classifier.load_model()
        self.task_classifier.load_models()
        # ensure external models are present (if any used by your loaders)
        print("Ensuring model availability for experts…")
        self.model_loader.download_all_models()

    def save_all_models(self):
        print("💾 Saving all models…")
        self.domain_classifier.save_model()
        self.task_classifier.save_models()
        print("✅ All models saved successfully!")

    def train_domain_classifier(self, training_data: List[Dict], **kwargs):
        print("Training Domain Classifier (Transformer)…")
        self.domain_classifier.fit_from_labeled_prompts(training_data, **kwargs)
        self.domain_classifier.save_model()

    def train_task_classifiers(self, training_data: List[Dict]):
        print("Training Task Classifiers (XLM-R fine-tuning)…")
        self.task_classifier.train_all(training_data)
        self.task_classifier.save_models()

    def route_prompt(self, prompt: str) -> Dict:
        lang = self.language_detector.detect_language(prompt)
        dom = self.domain_classifier.classify_domain(prompt)
        dom_probs = self.domain_classifier.get_domain_probabilities(prompt)
        task = self.task_classifier.classify_task(prompt, dom)
        return {
            'input': prompt, 'language': lang, 'domain': dom,
            'domain_probabilities': dom_probs, 'task': task,
            'routing_path': f"{lang} → {dom} → {task}"
        }

    def get_system_stats(self):
        total_tasks = sum(len(t) for t in self.domain_tasks.values())
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': len(self.language_detector.language_mapping),
            'domains': list(self.domain_tasks.keys())
        }

# =============================================================================
# 5) Metrics & Confusion (matching your first script style)
# =============================================================================
def _pct(x): return f"{x*100:6.2f}%"

def _compute_prf_bal_kappa(cm: Counter, labels: List[str]):
    support={c:0 for c in labels}; pred_tot={c:0 for c in labels}; tp={c:cm[(c,c)] for c in labels}
    for g in labels: support[g]=sum(cm[(g,p)] for p in labels)
    for p in labels: pred_tot[p]=sum(cm[(g,p)] for g in labels)
    total=sum(support.values()) if support else 0
    precisions=[]; recalls=[]; f1s=[]; weights=[]; recalls_only=[]
    micro_tp=sum(tp.values()); micro_fp=sum(pred_tot[c]-tp[c] for c in labels)
    micro_fn=sum(support[c]-tp[c] for c in labels)
    for c in labels:
        P = tp[c]/pred_tot[c] if pred_tot[c]>0 else 0.0
        R = tp[c]/support[c]  if support[c]>0  else 0.0
        F = (2*P*R/(P+R)) if (P+R)>0 else 0.0
        precisions.append(P); recalls.append(R); f1s.append(F); recalls_only.append(R)
        weights.append(support[c]/total if total else 0.0)
    macro_p=sum(precisions)/len(labels) if labels else 0.0
    macro_r=sum(recalls)/len(labels)    if labels else 0.0
    macro_f1=sum(f1s)/len(labels)       if labels else 0.0
    weighted_f1=sum(w*f for w,f in zip(weights,f1s)) if labels else 0.0
    bal_acc=sum(recalls_only)/len(labels) if labels else 0.0
    micro_p=micro_tp/(micro_tp+micro_fp) if (micro_tp+micro_fp)>0 else 0.0
    micro_r=micro_tp/(micro_tp+micro_fn) if (micro_tp+micro_fn)>0 else 0.0
    micro_f1=(2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p+micro_r)>0 else 0.0
    acc=micro_tp/total if total else 0.0
    pe=sum((support[c]/total)*(pred_tot[c]/total) for c in labels) if total else 0.0
    kappa=(acc-pe)/(1-pe) if (1-pe)>0 else 0.0
    return {'accuracy':acc,'macro_p':macro_p,'macro_r':macro_r,'macro_f1':macro_f1,
            'micro_p':micro_p,'micro_r':micro_r,'micro_f1':micro_f1,
            'weighted_f1':weighted_f1,'balanced_acc':bal_acc,'kappa':kappa,
            'support':support,'pred_tot':pred_tot,'total':total}

def _print_confusion(cm: Counter, labels: List[str], title: str):
    print(title)
    if not labels: print("  (no labels)\n"); return
    header = "      " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    for gt in labels:
        row=[f"{gt:>6}"]
        for pr in labels: row.append(f"{cm[(gt, pr)]:>22}")
        print(" ".join(row))
    print()

# =============================================================================
# 6) Main (train → eval)
# =============================================================================
if __name__ == "__main__":
    print("Initializing Prompt Routing System (XLM-R TaskCLS + XLM-R DomainCLS)…")
    system = PromptRoutingSystem()
    stats = system.get_system_stats()
    print(f"System initialized with {stats['total_domains']} domains and {stats['total_tasks']} tasks")
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Available domains: {stats['domains']}")

    # Optional training (domain + task)
    train_path = Path("train.json")
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            training_data = json.load(f)
        # Domain classifier (you can bump epochs if needed)
        system.train_domain_classifier(training_data, epochs=1, batch_size=32, lr=2e-5,
                                       freeze_encoder=True, class_weighting=True)
        # Task classifiers (fine-tune XLM-R)
        system.train_task_classifiers(training_data)
        system.save_all_models()

    # Evaluate like your first file
    test_path = Path("test.json")
    if not test_path.exists():
        print("⚠️ No test.json found; skipping evaluation.")
        sys.exit(0)

    with open(test_path, "r", encoding="utf-8") as f:
        test_prompts = json.load(f)

    print(f"\nTesting with {len(test_prompts)} sample prompts…")
    print("=" * 70)

    domain_labels = sorted({it['domain'] for it in test_prompts if isinstance(it, dict) and 'domain' in it})
    task_labels   = sorted({it['task']   for it in test_prompts if isinstance(it, dict) and 'task'   in it})
    cm_domain = Counter(); cm_task = Counter()

    per_lang_total = Counter(); per_lang_dom = Counter(); per_lang_task = Counter(); per_lang_exact = Counter()

    for it in test_prompts:
        if not isinstance(it, dict): continue
        text = it['prompt']; gt_domain = it['domain']; gt_task = it['task']
        out = system.route_prompt(text)
        pred_domain = out['domain']; pred_task = out['task']; lang = out.get('language','?')

        cm_domain[(gt_domain, pred_domain)] += 1
        cm_task[(gt_task, pred_task)]       += 1

        per_lang_total[lang] += 1
        dom_ok = (pred_domain == gt_domain)
        task_ok = (pred_task   == gt_task)
        per_lang_dom[lang]   += int(dom_ok)
        per_lang_task[lang]  += int(task_ok)
        per_lang_exact[lang] += int(dom_ok and task_ok)

    dom_metrics  = _compute_prf_bal_kappa(cm_domain, domain_labels)
    task_metrics = _compute_prf_bal_kappa(cm_task,   task_labels)

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
