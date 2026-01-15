"""
Main routing system orchestrator.

Coordinates language detection, domain classification, task routing,
and expert execution. This is the top-level component that ties everything together.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

project_root = Path(__file__).parents[7]
sys.path.insert(0, str(project_root))

from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader
from src.models.experts.llms.task_expert import TaskExpert, TaskExpertConfig
from src.models.experts.llms.expert_pool import LLMAdapterPool

# Import sibling components
from .language_detector import LanguageDetector
from .domain_classifier import DomainClassifier
from .q_learning_router import QLearningTaskClassifier

# Get DEVICE constant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PromptRoutingSystem:
    def __init__(self):
        config_path = Path(__file__).parents[5] / "experts" / "config"
        self.expert_registry_path = config_path / "experts_registry.json"

        # Initialize language detector with registry path
        self.language_detector = LanguageDetector(registry_path=self.expert_registry_path)
        self.domain_classifier = DomainClassifier(
            model_name="xlm-roberta-base",
            model_dir=Path(__file__).parent.parent / "models" / "domain_xlmr",
            max_len=128,
            alpha_proto=0.30,
            proto_temp=10.0
        )

        # ModelLoader is optional (legacy component for external model downloads)
        # Not needed when using LLMAdapterPool for model management
        try:
            self.model_loader = ModelLoader(config_path / "model_config.json")
        except FileNotFoundError:
            print("ℹ️  model_config.json not found - skipping legacy ModelLoader (not needed for LLMAdapterPool)")
            self.model_loader = None

        self.domain_tasks_obj = DomainTaskLoader(config_path / "domain_tasks.json")
        self.domain_tasks = self.domain_tasks_obj.domain_tasks if hasattr(self.domain_tasks_obj, "domain_tasks") else self.domain_tasks_obj
        
        # Q-learning task classifier
        self.task_classifier = QLearningTaskClassifier(
            self.domain_tasks,
            model_dir=Path(__file__).parent.parent / "models" / "task_routers_qlearning",
            encoder_name="xlm-roberta-base",
            max_len=128,
            batch_size=16,
            lr=1e-5,
            epochs=1,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10000
        )
        # Try loading existing domain model & QRouters
        self.domain_classifier.load_model()
        self.task_classifier.load_models()

        # Download any external models if needed (optional legacy feature)
        if self.model_loader:
            print("Checking and downloading models if needed...")
            self.model_loader.download_all_models()

        self.expert_pool = LLMAdapterPool(self.expert_registry_path)
        
        # Instantiate experts per domain/task using the registry
        self.experts = {}
        for domain, tasks in self.domain_tasks.items():
            self.experts[domain] = {}
            for task in tasks.keys():
                self.experts[domain][task] = TaskExpert(
                    TaskExpertConfig(
                        domain=domain,
                        task=task,
                        registry_path=str(self.expert_registry_path),
                        generation=None  # or per-task overrides dict
                    ),
                    pool=self.expert_pool
                )
        print(f"Initialized experts: {self.experts}")
    
    def save_all_models(self):
        print("💾 Saving all models...")
        self.domain_classifier.save_model()
        self.task_classifier.save_models()
        print("✅ All models saved successfully!")

    def train_domain_classifier(self, training_data: List[Dict], **kwargs):
        """
        Train the transformer domain classifier on labeled prompts.
        kwargs are passed to fit_from_labeled_prompts (epochs, batch_size, lr, freeze_encoder, ...).
        """
        print("Training Domain Classifier (Transformer)...")
        self.domain_classifier.fit_from_labeled_prompts(training_data, **kwargs)
        self.domain_classifier.save_model()
    
    def train_q_routers(self, training_data: List[Dict]):
        """Train per-domain QRouters on labeled (domain, task, prompt) items."""
        self.task_classifier.train(training_data, val_split=0.1)
        self.task_classifier.save_models()
    
    def route_prompt(self, prompt: str, classification_text: str = None,
                    review_title: str = None, input_data: Dict[str, str] = None) -> Dict:
        """
        Generic routing method supporting multiple task types.

        Args:
            prompt: Full prompt with instructions
            classification_text: (Legacy) Text to classify (use input_data instead)
            review_title: (Legacy) Title text (use input_data instead)
            input_data: Task-specific data fields (e.g., {"text": "...", "title": "..."})

        Returns:
            Dict with routing results including language, domain, task, result

        Usage:
            # New way (preferred):
            system.route_prompt(prompt, input_data={"text": "...", "title": "..."})

            # Old way (backward compatible):
            system.route_prompt(prompt, classification_text, review_title)
        """
        # Handle backward compatibility
        if input_data is None:
            # Legacy mode: construct input_data from positional arguments
            input_data = {
                "classification_text": classification_text or "",
                "review_title": review_title or ""
            }

        language = self.language_detector.detect_language(prompt)
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        task = self.task_classifier.classify_task(prompt, domain)
        expert = self.experts[domain][task]

        # Pass input_data directly to expert - it handles field extraction
        result, expert_confidence, raw_response = expert.predict(
            input_data,
            prompt,
            language
        )
        output = {
            'input': prompt,
            'input_data': input_data,  # Include for downstream use
            'language': language,
            'domain': domain,
            'domain_probabilities': domain_probs,
            'task': task,
            'result': result,
            'expert_confidence': expert_confidence,
            'routing_path': f"{language} → {domain} → {task}",
            'raw_response': raw_response
        }
        return output
    def get_system_stats(self):
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.values())
        all_languages = self.language_detector.all_supported_languages
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': len(all_languages),
            'all_languages': sorted(all_languages),
            'languages_by_task': self.language_detector.supported_languages_by_task,
            'domains': list(self.domain_tasks.keys())
        }
