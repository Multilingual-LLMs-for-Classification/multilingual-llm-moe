# src/models/experts/llms/expert_pool.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def _maybe_bnb_quant(load_in_4bit: bool) -> Dict:
    if not load_in_4bit:
        return {}
    try:
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        }
    except Exception:
        # BitsAndBytes not installed; fallback to full precision
        return {}


class LLMAdapterPool:
    """
    1) Caches base LLMs by key (shared across tasks).
    2) On demand, loads/attaches a LoRA adapter (by name) to that base model.
    3) Switches the active adapter before generation.
    """
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.base_models: Dict[str, Dict] = {}   # key -> {"model":..., "tok":..., "adapters_loaded": set(), "active": str|None}
        self.model_access_times: Dict[str, float] = {}  # Track last access time for LRU eviction
        self.max_loaded_models: int = 1  # Maximum models to keep in GPU memory
        self.default_gen = self.cfg.get("default_generation", {"max_new_tokens": 4})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Memory Management ---------- #
    def _update_access_time(self, base_key: str):
        """Update the last access time for a model (for LRU tracking)."""
        import time
        self.model_access_times[base_key] = time.time()

    def _get_lru_model(self) -> Optional[str]:
        """Get the least recently used model key."""
        if not self.model_access_times:
            return None
        return min(self.model_access_times.items(), key=lambda x: x[1])[0]

    def _unload_base_model(self, base_key: str):
        """Unload a base model from GPU memory to free up space."""
        if base_key not in self.base_models:
            return

        print(f"[Memory Management] Unloading model '{base_key}' from GPU...")

        # Move model to CPU to free GPU memory
        self.base_models[base_key]["model"].cpu()

        # Delete from cache
        del self.base_models[base_key]
        if base_key in self.model_access_times:
            del self.model_access_times[base_key]

        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print(f"✓ Model '{base_key}' unloaded successfully")

    def _ensure_memory_available(self, base_key_to_load: str):
        """Ensure sufficient memory is available before loading a new model."""
        # Count currently loaded models
        loaded_count = len(self.base_models)

        # If we're at capacity and need to load a new model
        if loaded_count >= self.max_loaded_models and base_key_to_load not in self.base_models:
            # Unload LRU model(s)
            while len(self.base_models) >= self.max_loaded_models:
                lru_key = self._get_lru_model()
                if lru_key:
                    self._unload_base_model(lru_key)
                else:
                    break

    # ---------- Base models ---------- #
    def _load_base_if_needed(self, base_key: str):
        if base_key in self.base_models:
            # Model already loaded, just update access time
            self._update_access_time(base_key)
            return

        # Ensure memory is available before loading new model
        self._ensure_memory_available(base_key)

        base = self.cfg["base_models"][base_key]
        hf_name = base["hf_name"]
        load_in_4bit = bool(base.get("load_in_4bit", False))
        device_map = base.get("device_map", "auto")

        print(f"[LLMAdapterPool] Loading base model: {base_key} ({hf_name})")
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        bnb_kw = _maybe_bnb_quant(load_in_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map,
            **bnb_kw
        )
        self.base_models[base_key] = {
            "model": model,
            "tok": tok,
            "adapters_loaded": set(),
            "active": None
        }
        # Mark as recently used
        self._update_access_time(base_key)

    # ---------- Adapters ---------- #
    def _ensure_adapter(self, base_key: str, adapter_name: str, adapter_path: str):
        """Load LoRA adapter onto base model if not yet loaded."""
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. `pip install peft`")

        slot = self.base_models[base_key]
        model = slot["model"]

        if adapter_name in slot["adapters_loaded"]:
            # Already loaded, no need to print (reduces terminal clutter)
            return

        root = Path(__file__).parents[4]
        full_adapter_path = (root / adapter_path).resolve()

        print(f"[Adapter] Loading adapter '{adapter_name}' from: {full_adapter_path}")

        if hasattr(model, "load_adapter"):
            model.load_adapter(full_adapter_path, adapter_name=adapter_name)
        else:
            peft_model = PeftModel.from_pretrained(
                model,
                full_adapter_path,
                adapter_name=adapter_name,
                is_trainable=False
            )
            self.base_models[base_key]["model"] = peft_model
            model = peft_model

        slot["adapters_loaded"].add(adapter_name)
        print(f"✓ Adapter '{adapter_name}' loaded successfully on base model '{base_key}'")

    def _activate_adapter(self, base_key: str, adapter_name: Optional[str]):
        """Activate/switch to the specified adapter on the base model."""
        slot = self.base_models[base_key]
        model = slot["model"]

        if adapter_name is None:
            # fall back to base (disable adapters if supported)
            if hasattr(model, "disable_adapter"):
                model.disable_adapter()  # peft>=0.11
            elif hasattr(model, "disable_adapters"):
                model.disable_adapters()
            slot["active"] = None
            print(f"[Adapter] Disabled adapters on '{base_key}', using base model")
            return

        # set active
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        elif hasattr(model, "set_active_adapters"):
            model.set_active_adapters(adapter_name)
        slot["active"] = adapter_name
        # Only print on first activation to reduce terminal clutter

    # ---------- Public API ---------- #
    def _resolve_base_model_for_language(self, task_key: str, language: Optional[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Resolve which base model, adapter, and template to use based on language mapping.
        Returns: (base_model_key, adapter_name, adapter_path, template_path)
        """
        tcfg = self.cfg["tasks"].get(task_key)
        if not tcfg:
            raise ValueError(f"Task {task_key} not found in registry")

        # Default: use task's base_model_key, adapter_name, adapter_path, and template_path
        default_base = tcfg.get("base_model_key")
        default_adapter_name = tcfg.get("adapter_name")
        default_adapter_path = tcfg.get("adapter_path")
        default_template = tcfg.get("template_path")

        # Check if task has language_mapping
        lang_mapping = tcfg.get("language_mapping")
        if not lang_mapping or not language:
            return default_base, default_adapter_name, default_adapter_path, default_template

        # Normalize language (e.g., 'english' detected by FastText)
        lang_normalized = language.lower()

        # Priority 1: Check for direct per-language mapping
        if lang_normalized in lang_mapping:
            lang_cfg = lang_mapping[lang_normalized]
            # Check if it's a per-language entry (no "languages" key)
            if "languages" not in lang_cfg:
                base_key = lang_cfg.get("base_model_key", default_base)
                adapter_name = lang_cfg.get("adapter_name", default_adapter_name)
                adapter_path = lang_cfg.get("adapter_path", default_adapter_path)
                template_path = lang_cfg.get("template_path", default_template)
                print(f"[LLMAdapterPool] Language '{language}' → per-language mapping → model '{base_key}', adapter '{adapter_name}'")
                return base_key, adapter_name, adapter_path, template_path

        # Priority 2: Find which group contains this language
        for group_name, group_cfg in lang_mapping.items():
            if lang_normalized in group_cfg.get("languages", []):
                base_key = group_cfg.get("base_model_key", default_base)
                adapter_name = group_cfg.get("adapter_name", default_adapter_name)
                adapter_path = group_cfg.get("adapter_path", default_adapter_path)
                template_path = group_cfg.get("template_path", default_template)
                print(f"[LLMAdapterPool] Language '{language}' → group '{group_name}' → model '{base_key}', adapter '{adapter_name}'")
                return base_key, adapter_name, adapter_path, template_path

        # Priority 3: Fallback to default if language not in any group
        print(f"[LLMAdapterPool] Language '{language}' not in mapping, using default '{default_base}'")
        return default_base, default_adapter_name, default_adapter_path, default_template

    def ensure_task_ready(self, task_key: str, language: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        task_key: 'domain/task' (e.g., 'finance/sentiment_analysis')
        language: detected language (e.g., 'english', 'japanese') for language-based model selection
        Returns model (with active adapter set) and tokenizer.
        """
        tcfg = self.cfg["tasks"].get(task_key)
        if tcfg is None:
            raise KeyError(f"Task '{task_key}' not found in experts_registry.json")

        # Resolve base model, adapter_name, adapter_path, and template based on language
        base_key, adapter_name, adapter_path, _ = self._resolve_base_model_for_language(task_key, language)
        self._load_base_if_needed(base_key)

        # Update access time since we're using this model
        self._update_access_time(base_key)

        # Use resolved adapter_name and adapter_path from language mapping
        if adapter_name and adapter_path:
            self._ensure_adapter(base_key, adapter_name, adapter_path)
            self._activate_adapter(base_key, adapter_name)
        else:
            # Use plain base model if no adapter specified
            self._activate_adapter(base_key, None)

        slot = self.base_models[base_key]
        return slot["model"], slot["tok"]

    def get_task_template(self, task_key: str) -> Optional[str]:
        tcfg = self.cfg["tasks"].get(task_key, {})
        tpath = tcfg.get("template_path")

        if not tpath:
            return None

        p = Path(__file__).parents[4] / tpath
        if not p.exists():
            return None

        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))

        return p.read_text(encoding="utf-8")

    def get_task_template_for_language(self, task_key: str, language: Optional[str] = None) -> Optional[str]:
        """
        Load template for specific language, respecting language_mapping overrides.
        Returns the template content (dict if JSON, string if text file).
        """
        _, _, _, tpath = self._resolve_base_model_for_language(task_key, language)

        if not tpath:
            return None

        p = Path(__file__).parents[4] / tpath
        if not p.exists():
            return None

        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))

        return p.read_text(encoding="utf-8")

    def default_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.default_gen)

    @torch.inference_mode()
    def generate(
        self,
        task_key: str,
        classification_text: str,
        review_title: str,
        prompt: str,
        language: str = "english",
        **gen_overrides
        ) -> Tuple[str, float]:

        # Pass language to ensure_task_ready for language-based model selection
        model, tok = self.ensure_task_ready(task_key, language=language)
        template = self.get_task_template_for_language(task_key, language)

        if isinstance(template, dict):
            # normalize language key
            lang_key = language.lower()
            if lang_key not in template:
                lang_key = "english"
            # Truncate inputs to match run_lora_star.py
            truncated_text = str(classification_text).replace('\n', ' ').strip()[:400]
            truncated_title = str(review_title).replace('\n', ' ').strip()[:80]
            text = template[lang_key].replace("{{input}}", truncated_text).replace("{{review_title}}", truncated_title)
        else:
            text = template.replace("{{input}}", classification_text).replace("{{review_title}}", review_title)

        inputs = tok(text, return_tensors="pt").to(model.device)
        gen_cfg = self.default_generation_config()
        for k, v in gen_overrides.items():
            setattr(gen_cfg, k, v)

        out = model.generate(
            **inputs,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=True
        )
        # decoded = tok.decode(out.sequences[0], skip_special_tokens=True)
        seq = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = seq[prompt_len:]

        decoded = tok.decode(new_tokens, skip_special_tokens=True)
        conf = 0.0
        if out.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s[0], dim=-1).max().item() for s in out.scores]
            if probs:
                conf = float(sum(probs) / len(probs))
        return decoded, conf
