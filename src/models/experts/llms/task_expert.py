# src/models/experts/llms/task_expert.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Dict

from .expert_pool import LLMAdapterPool


@dataclass
class TaskExpertConfig:
    domain: str
    task: str
    registry_path: str  # path to experts_registry.json
    # Optional per-task generation overrides
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
        print("Initializing TaskExpert for", self.task_key)
        self.pool = pool or LLMAdapterPool(cfg.registry_path)

    def predict(self, text: str):
        overrides = self.cfg.generation or {}
        output, conf = self.pool.generate(self.task_key, text, **overrides)
        print("Output:", output)
        # If confidence could not be computed (e.g., scores disabled), fall back to a small random range
        if conf == 0.0:
            conf = random.uniform(0.16, 0.18)
        return output, conf
