"""
Base model pool for evaluation without adapters.

This module provides a simplified model pool that loads base models
WITHOUT any LoRA adapters, for comparison with the adapter-based system.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def _maybe_bnb_quant(load_in_4bit: bool) -> Dict:
    """Configure 4-bit quantization if requested."""
    if not load_in_4bit:
        return {}
    try:
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        }
    except Exception:
        return {}


class BaseModelPool:
    """
    Simplified model pool that loads base models WITHOUT adapters.

    Used for baseline comparison to measure the benefit of fine-tuned adapters.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.base_models: Dict[str, Dict] = {}
        self.default_gen = self.cfg.get("default_generation", {"max_new_tokens": 256})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_base_model(self, base_key: str):
        """Load a base model without any adapters."""
        if base_key in self.base_models:
            return

        base = self.cfg["base_models"][base_key]
        hf_name = base["hf_name"]
        load_in_4bit = bool(base.get("load_in_4bit", False))
        device_map = base.get("device_map", "auto")

        print(f"[BaseModelPool] Loading base model: {base_key} ({hf_name})")
        print(f"[BaseModelPool] NO ADAPTERS will be loaded - using pure base model")

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
            "tok": tok
        }
        print(f"[BaseModelPool] Base model '{base_key}' loaded successfully")

    def get_model(self, base_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Get the base model and tokenizer (no adapter)."""
        self._load_base_model(base_key)
        slot = self.base_models[base_key]
        return slot["model"], slot["tok"]

    def get_task_config(self, task_key: str) -> Dict:
        """Get task configuration from registry."""
        return self.cfg["tasks"].get(task_key, {})

    def get_task_template(self, task_key: str) -> Optional[str]:
        """Load template for a task."""
        tcfg = self.cfg["tasks"].get(task_key, {})
        tpath = tcfg.get("template_path")

        if not tpath:
            return None

        # Project root is 7 levels up from this file
        # base_evaluation -> qlearning-router -> rl-based -> without-translation -> gating -> models -> src -> multilingual-llm-moe
        project_root = Path(__file__).parents[7]
        p = project_root / tpath

        if not p.exists():
            print(f"[BaseModelPool] Template not found: {p}")
            return None

        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))

        return p.read_text(encoding="utf-8")

    def default_generation_config(self) -> GenerationConfig:
        """Get default generation configuration."""
        return GenerationConfig(**self.default_gen)

    @torch.inference_mode()
    def generate(
        self,
        base_key: str,
        task_key: str,
        classification_text: str,
        review_title: str,
        language: str = "english",
        **gen_overrides
    ) -> Tuple[str, float]:
        """
        Generate output using the base model (without adapters).

        Args:
            base_key: Base model key (e.g., "llama-2-7b-hf")
            task_key: Task key (e.g., "finance/rating")
            classification_text: Main text to process
            review_title: Title text (for rating task)
            language: Language for template selection
            **gen_overrides: Generation parameter overrides

        Returns:
            Tuple of (decoded_output, confidence)
        """
        model, tok = self.get_model(base_key)
        template = self.get_task_template(task_key)

        if template is None:
            raise ValueError(f"No template found for task {task_key}")

        # Build prompt from template
        if isinstance(template, dict):
            # Use English template as default (matches training setup)
            lang_key = language.lower() if language.lower() in template else "english"
            if lang_key not in template:
                lang_key = next(iter(template.keys()))

            # Truncate inputs
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

        seq = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = seq[prompt_len:]

        decoded = tok.decode(new_tokens, skip_special_tokens=True)

        # Calculate confidence
        conf = 0.0
        if out.scores:
            import torch.nn.functional as F
            probs = [F.softmax(s[0], dim=-1).max().item() for s in out.scores]
            if probs:
                conf = float(sum(probs) / len(probs))

        return decoded, conf

    def unload_all(self):
        """Unload all models from memory."""
        import gc
        for key in list(self.base_models.keys()):
            self.base_models[key]["model"].cpu()
            del self.base_models[key]
        gc.collect()
        torch.cuda.empty_cache()
        print("[BaseModelPool] All models unloaded")
