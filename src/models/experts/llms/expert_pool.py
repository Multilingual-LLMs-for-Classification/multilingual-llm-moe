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
        self.default_gen = self.cfg.get("default_generation", {"max_new_tokens": 4})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Base models ---------- #
    def _load_base_if_needed(self, base_key: str):
        if base_key in self.base_models:
            return
        base = self.cfg["base_models"][base_key]
        hf_name = base["hf_name"]
        load_in_4bit = bool(base.get("load_in_4bit", False))
        device_map = base.get("device_map", "auto")

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

    # ---------- Adapters ---------- #
    def _ensure_adapter(self, base_key: str, adapter_name: str, adapter_path: str):
        """Load LoRA adapter onto base model if not yet loaded."""
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. `pip install peft`")

        slot = self.base_models[base_key]
        model = slot["model"]

        if adapter_name in slot["adapters_loaded"]:
            return

        # Load/attach adapter (idempotent)
        adapter_path = Path(__file__).parent / "adapters" / "finance" / "news_classification" / "llama-2-7b-hf"
        print("adapter_path:", adapter_path)
        if hasattr(model, "load_adapter"):
            model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            # Older PEFT pattern: wrap model on first adapter
            peft_model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name, is_trainable=False)
            # replace reference so pool keeps wrapped model
            self.base_models[base_key]["model"] = peft_model
            model = peft_model
        slot["adapters_loaded"].add(adapter_name)

    def _activate_adapter(self, base_key: str, adapter_name: Optional[str]):
        slot = self.base_models[base_key]
        model = slot["model"]
        if adapter_name is None:
            # fall back to base (disable adapters if supported)
            if hasattr(model, "disable_adapter"):
                model.disable_adapter()  # peft>=0.11
            elif hasattr(model, "disable_adapters"):
                model.disable_adapters()
            slot["active"] = None
            return

        # set active
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        elif hasattr(model, "set_active_adapters"):
            model.set_active_adapters(adapter_name)
        slot["active"] = adapter_name

    # ---------- Public API ---------- #
    def ensure_task_ready(self, task_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        task_key: 'domain/task' (e.g., 'finance/sentiment_analysis')
        Returns model (with active adapter set) and tokenizer.
        """
        tcfg = self.cfg["tasks"].get(task_key)
        if tcfg is None:
            raise KeyError(f"Task '{task_key}' not found in experts_registry.json")

        base_key = tcfg["base_model_key"]
        self._load_base_if_needed(base_key)

        adapter_name = tcfg.get("adapter_name")
        adapter_path = tcfg.get("adapter_path")
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
        p = Path(tpath)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def default_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.default_gen)

    @torch.inference_mode()
    def generate(self, task_key: str, prompt: str, **gen_overrides) -> Tuple[str, float]:
        model, tok = self.ensure_task_ready(task_key)

        # template = self.get_task_template(task_key)
        # text = template.replace("{{input}}", prompt) if template else prompt
        text = f"""<|system|>
            You are a rating pre-processor and executor. The task is KNOWN to be `rating`.

            Input format:
            A free-form string (RAW_INPUT) may include:
            • a user instruction (e.g., “Convert the sentiment … rate 1-5”)
            • the actual product review (possibly with a short title before a colon)
            • optional metadata like “(Category: …)”

            Your job:
            1) Remove the instruction/meta part and keep only the actual review.
            2) Extract title (if short phrase before first colon) and review text.
            3) Remove trailing parenthetical metadata like “(Category: …)”.
            4) Infer or preserve language.
            5) Apply the rating card below and output ONLY a single digit 1-5.

            [RATING_TASK_SYSTEM_CARD]
            You are an expert at analyzing product reviews and assigning star ratings from 1-5.
            1 = Very negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very positive.

            Instruction: Rate this review with a number from 1-5. Output ONLY the number (no explanation).
            [/RATING_TASK_SYSTEM_CARD]

            <|user|>
            # RAW_INPUT
            {prompt}

            # DO
            1) Strip instruction/meta from RAW_INPUT.
            2) Extract the review content (title + text).
            3) Apply the rating card and output ONLY 1-5.

            # CONSTRAINTS
            - Output ONLY a single integer 1-5.
            - No JSON, labels, or commentary.

            <|assistant|>"""

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
        gen_ids = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[prompt_len:]
        decoded = tok.decode(new_tokens, skip_special_tokens=True)
        m = re.search(r"\b([1-5])\b", decoded)
        clean = m.group(1) if m else "" 

        # A lightweight confidence proxy: average of top-token probs of generated tokens
        conf = 0.0
        if out.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s[0], dim=-1).max().item() for s in out.scores]
            if probs:
                conf = float(sum(probs) / len(probs))
        return clean, conf
