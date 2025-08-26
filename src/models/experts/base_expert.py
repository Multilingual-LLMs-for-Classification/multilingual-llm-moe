from abc import ABC, abstractmethod
import json
import pickle
import os

class BaseLLMExpert(ABC):
    def __init__(self, task_name, model_path=None, config_path=None):
        self.task_name = task_name
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = self._load_model()
    
    def _load_config(self, config_path):
        """Load model configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    @abstractmethod
    def _load_model(self):
        """Load the actual LLM model - implement in subclasses"""
        pass
    
    @abstractmethod
    def predict(self, text):
        """Make prediction - implement in subclasses"""
        pass
    
    def preprocess_input(self, text):
        """Common preprocessing for all experts"""
        # Add any common preprocessing here
        return text.strip()
    
    def postprocess_output(self, output):
        """Common postprocessing for all experts"""
        return output
