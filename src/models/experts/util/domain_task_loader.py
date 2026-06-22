import json
import os
from pathlib import Path

class DomainTaskLoader:
    def __init__(self, config_path=None):

        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "experts_registry.json"
        with open(config_path, 'r') as f:
            registry = json.load(f)

        # Derive domain/task structure from experts_registry.json "tasks" keys
        # e.g., "finance/rating" → {"finance": {"rating": {}}}
        self.domain_tasks = {}
        for task_key in registry.get("tasks", {}).keys():
            parts = task_key.split("/", 1)
            if len(parts) == 2:
                domain, task = parts
                if domain not in self.domain_tasks:
                    self.domain_tasks[domain] = {}
                self.domain_tasks[domain][task] = {}

    def items(self):
        return self.domain_tasks.items()

    def keys(self):
        return self.domain_tasks.keys()

    def values(self):
        return self.domain_tasks.values()

    def get(self, domain, default=None):
        return self.domain_tasks.get(domain, default)

    def __getitem__(self, key):
        return self.domain_tasks[key]

    def __len__(self):
        return len(self.domain_tasks)
