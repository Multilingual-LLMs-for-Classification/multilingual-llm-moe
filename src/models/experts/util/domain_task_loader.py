import json
import os
from pathlib import Path

class DomainTaskLoader:
    def __init__(self, config_path=None):
        
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "domain_tasks.json"
        with open(config_path, 'r') as f:
            self.domain_tasks = json.load(f)
    
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
