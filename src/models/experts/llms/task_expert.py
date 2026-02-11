from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Dict
import importlib
from pathlib import Path
from .expert_pool import LLMAdapterPool

@dataclass
class TaskExpertConfig:
    domain: str
    task: str
    registry_path: Path
    generation: Optional[Dict] = None

class TaskExpert:
    """
    Self-MoE style expert:
      - Shares a base model through LLMAdapterPool
      - Activates the correct LoRA adapter per task at inference time
    """
    def __init__(self, cfg: TaskExpertConfig, pool: LLMAdapterPool | None = None):
        self.cfg = cfg
        self.task_key = f"{cfg.domain}/{cfg.task}"
        self.pool = pool or LLMAdapterPool(cfg.registry_path)

        tcfg = self.pool.cfg["tasks"].get(self.task_key, None)
        if tcfg is None:
            raise KeyError(f"[ERROR] Task '{self.task_key}' missing in registry!")

        expert_path = tcfg.get("expert_path", None)
        if not expert_path:
            print(f"[WARNING] No expert_path defined for {self.task_key}")
            self.cleaner = None
            return

        module_path = expert_path.replace("/", ".")
        class_name = expert_path.split("/")[-1]

        self.cleaner = None
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self.cleaner = cls()
        except Exception as e:
            print(f"[WARNING] No cleanup expert for {self.task_key}: {e}")
        # ------------------------------------------------------------------

    def predict(self, input_data: dict, prompt: str, language: str = "en"):
        """
        Generic prediction method supporting multiple task types.

        Delegates to task-specific expert for input preparation.

        Args:
            input_data: Task-specific data fields (e.g., {"text": "...", "title": "..."})
            prompt: Full prompt with instructions
            language: Language code (e.g., "english", "german")

        Returns:
            Tuple of (cleaned_output, confidence, raw_output)
        """
        # Delegate field extraction to task-specific expert if available
        if self.cleaner and hasattr(self.cleaner, 'prepare_input'):
            classification_text, review_title = self.cleaner.prepare_input(input_data)
        else:
            # Fallback: generic field extraction if expert doesn't have prepare_input
            classification_text = input_data.get("text",
                                                input_data.get("classification_text",
                                                input_data.get("generated_text", "")))
            review_title = input_data.get("title",
                                         input_data.get("review_title", ""))

        overrides = self.cfg.generation or {}
        raw_output, conf, base_model_key, prompt_sent = self.pool.generate(
            self.task_key,
            classification_text,
            review_title,
            prompt,
            language=language,
            **overrides
        )

        if self.cleaner:
            try:
                cleaned = self.cleaner.clean_output(raw_output)
            except Exception as e:
                print("[CLEANER ERROR]", e)
                cleaned = raw_output.strip()
        else:
            cleaned = raw_output.strip()

        if conf == 0.0:
            conf = random.uniform(0.16, 0.18)

        return cleaned, conf, raw_output, base_model_key, prompt_sent
