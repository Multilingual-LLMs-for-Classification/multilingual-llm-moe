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
        root = Path(__file__).parents[4]
        adapter_path = (root / adapter_path).resolve()
        # adapter_dir = Path(adapter_path).expanduser().resolve()
        # adapter_path = str(adapter_dir)
        print("Resolved adapter_path:", adapter_path)
        
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
        
        p = Path(__file__).parents[4] / tpath
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def default_generation_config(self) -> GenerationConfig:
        return GenerationConfig(**self.default_gen)

    @torch.inference_mode()
    def generate(self, task_key: str, classification_text: str, prompt: str, **gen_overrides) -> Tuple[str, float]:
        model, tok = self.ensure_task_ready(task_key)

        template = self.get_task_template(task_key)
        if template:
            text = template.replace("{{input}}", classification_text)
        else:
            print("################")
            text = prompt

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
        print("decoded: ", decoded)
        m = re.search(r"\b([1-5])\b", decoded)
        clean = m.group(1) if m else "" 
        print("Clean: ", clean)

        # A lightweight confidence proxy: average of top-token probs of generated tokens
        conf = 0.0
        if out.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s[0], dim=-1).max().item() for s in out.scores]
            if probs:
                conf = float(sum(probs) / len(probs))
        return clean, conf


    # @torch.inference_mode()
    # def generate(self, task_key: str, prompt: str, **gen_overrides) -> Tuple[str, float]:
    #     """
    #     Clean, robust generation pipeline for rating tasks:
    #     1. Use the LLM to extract the true review text from the raw prompt.
    #     2. Feed the clean review into a minimal 1–5 rating instruction.
    #     3. Generate with strict decoding that only accepts digits 1–5.
    #     """
    #     import re
    #     import torch.nn.functional as F

    #     # --------------------------------------------------
    #     # 1. Load model + adapter
    #     # --------------------------------------------------
    #     model, tok = self.ensure_task_ready(task_key)

    #     # --------------------------------------------------
    #     # 2. LLM-based extraction of the actual review
    #     # --------------------------------------------------
    #     def extract_review_llm(raw_prompt: str) -> str:
    #         """
    #         Use the same LLM to extract ONLY the clean review text.
    #         This handles multilingual text, titles, instructions, metadata.
    #         """
    #         extraction_prompt = f"""
    #             <s>[INST] <<SYS>>
    #             You extract ONLY the actual product review from mixed user prompts.

    #             Rules:
    #             - Remove instructions like “rate from 1–5”, “assign stars”, etc.
    #             - Remove Q/A wrappers or markers like “P:”, “Q:”, “R:”.
    #             - Remove category metadata in parentheses.
    #             - Preserve the actual review title and content.
    #             - Return ONLY the clean review, nothing else.
    #             <</SYS>>

    #             User Prompt:
    #             {raw_prompt}

    #             Extracted Review:
    #             [/INST]
    #         """

    #         inputs = tok(extraction_prompt, return_tensors="pt").to(model.device)

    #         out = model.generate(
    #             **inputs,
    #             max_new_tokens=128,
    #             temperature=0.1,
    #             top_p=0.9
    #         )

    #         # decode only generated portion
    #         gen_ids = out[0]
    #         decoded = tok.decode(gen_ids, skip_special_tokens=True).strip()

    #         # Typically the LLM answer is the last lines -> clean extraction
    #         return decoded.strip()

    #     review = extract_review_llm(prompt)
    #     print("LLM Extracted Review:", review)

    #     # --------------------------------------------------
    #     # 3. Build clean rating prompt (no numeric pollution)
    #     # --------------------------------------------------
    #     rating_prompt = f"""
    #         <s>[INST] <<SYS>>
    #         You are an expert product review rating model.

    #         Rate a review from 1 to 5:
    #         1 = very negative
    #         2 = negative
    #         3 = neutral
    #         4 = positive
    #         5 = very positive

    #         Respond ONLY with the rating (1–5).
    #         No words, no punctuation, no explanations.
    #         <</SYS>>

    #         Review:
    #         {review}
    #         [/INST]
    #     """

    #     # --------------------------------------------------
    #     # 4. Tokenize
    #     # --------------------------------------------------
    #     inputs = tok(rating_prompt, return_tensors="pt").to(model.device)

    #     # --------------------------------------------------
    #     # 5. Generation config
    #     # --------------------------------------------------
    #     gen_cfg = self.default_generation_config()
    #     gen_cfg.max_new_tokens = gen_overrides.get("max_new_tokens", 8)
    #     gen_cfg.temperature = gen_overrides.get("temperature", 0.1)
    #     gen_cfg.top_p = gen_overrides.get("top_p", 0.95)

    #     # --------------------------------------------------
    #     # 6. Generate output
    #     # --------------------------------------------------
    #     out = model.generate(
    #         **inputs,
    #         generation_config=gen_cfg,
    #         return_dict_in_generate=True,
    #         output_scores=True
    #     )

    #     # --------------------------------------------------
    #     # 7. Decode ONLY generated tokens
    #     # --------------------------------------------------
    #     gen_ids = out.sequences[0]
    #     prompt_len = inputs["input_ids"].shape[1]
    #     new_tokens = gen_ids[prompt_len:]
    #     decoded = tok.decode(new_tokens, skip_special_tokens=True).strip()
    #     print("DEBUG raw decoded:", repr(decoded))

    #     # --------------------------------------------------
    #     # 8. STRICT rating extraction: allow only digits 1–5
    #     # --------------------------------------------------
    #     m = re.search(r"\b([1-5])\b", decoded)
    #     clean = m.group(1) if m else ""

    #     # --------------------------------------------------
    #     # 9. Confidence estimate (optional)
    #     # --------------------------------------------------
    #     conf = 0.0
    #     if out.scores:
    #         probs = [F.softmax(s[0], dim=-1).max().item() for s in out.scores]
    #         if probs:
    #             conf = float(sum(probs) / len(probs))

    #     return clean, conf


