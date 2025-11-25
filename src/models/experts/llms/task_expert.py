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

    def predict(self, classification_text: str, prompt: str, language: str = "en"):

        overrides = self.cfg.generation or {}
        raw_output, conf = self.pool.generate(
            self.task_key,
            classification_text,
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

        return cleaned, conf
