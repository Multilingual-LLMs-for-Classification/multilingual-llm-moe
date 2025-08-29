import json
import os
from importlib import import_module
import requests
from pathlib import Path
import shutil

class ModelLoader:
    def __init__(self, config_path=None):
        
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "model_config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create base llms directory if it doesn't exist
        self.models_base_dir = Path(__file__).parent.parent / "llms"
        self.models_base_dir.mkdir(exist_ok=True)
        
        # Create domain directories
        self._create_domain_directories()
    
    def _create_domain_directories(self):
        """Create directories for each domain under llms/"""
        for domain in self.config.keys():
            domain_dir = self.models_base_dir / domain
            domain_dir.mkdir(exist_ok=True)
            print(f"Created/verified domain directory: {domain_dir}")
    
    def _get_domain_cache_path(self, domain, task):
        """Get the cache path for a specific domain and task"""
        return self.models_base_dir / domain / task
    
    def _download_custom_model(self, model_config, domain, task):
        
        print("sss")
        """Download custom model files from URL"""
        # Update model path to use domain-based structure
        original_path = model_config.get('model_path', '')
        domain_cache_path = self._get_domain_cache_path(domain, task)
        domain_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create new path under domain directory
        model_filename = Path(original_path).name if original_path else f"{task}_model.pkl"
        model_path = domain_cache_path / model_filename
        
        download_url = model_config.get('download_url')
        
        if not model_path.exists() and download_url:
            print(f"Downloading custom model from {download_url}...")
            print(f"Saving to: {model_path}")
            
            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                
                print(f"Model downloaded to {model_path}")
                return True
            except Exception as e:
                print(f"Error downloading model: {e}")
                return False
        
        return model_path.exists()
    
    def _check_huggingface_model(self, model_config, domain, task):
        print("_check_huggingface_model initiated...")
        """Check and download HuggingFace model if needed"""
        model_name = model_config['model_name']
        download_if_missing = model_config.get('download_if_missing', True)
        
        # Use domain-based cache directory
        cache_dir = self._get_domain_cache_path(domain, task)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"Cache directory for {domain}/{task}: {cache_dir}")
            
            # Check if model exists locally
            model_exists = self._model_exists_locally(model_name, cache_dir)
            
            if not model_exists and download_if_missing:
                print(f"Downloading HuggingFace model: {model_name}")
                print(f"Domain: {domain}, Task: {task}")
                print(f"Cache directory: {cache_dir}")
                
                # Download model and tokenizer to domain-specific cache directory
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                
                print(f"Model {model_name} downloaded successfully to {cache_dir}")
                return True
            
            return model_exists
            
        except Exception as e:
            print(f"Error with HuggingFace model {model_name}: {e}")
            return False
    
    def _model_exists_locally(self, model_name, cache_dir=None):
        """Check if HuggingFace model exists locally"""
        if cache_dir:
            cache_path = Path(cache_dir)
            
            # Check various possible directory structures
            possible_dirs = [
                cache_path / "models--" / model_name.replace("/", "--"),
                cache_path / model_name.replace("/", "--"),
                cache_path / model_name.split("/")[-1],  # Just model name without org
                cache_path
            ]
            
            for model_dir in possible_dirs:
                if model_dir.exists():
                    # Check for essential model files
                    essential_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                    files_in_dir = [f.name for f in model_dir.rglob('*') if f.is_file()]
                    
                    # If we find any essential files, consider model exists
                    if any(essential in ' '.join(files_in_dir) for essential in essential_files):
                        print(f"Found existing model at: {model_dir}")
                        return True
        
        # Check in default HuggingFace cache as fallback
        try:
            from transformers import AutoModel
            AutoModel.from_pretrained(model_name, local_files_only=True)
            return True
        except:
            return False
    
    def ensure_model_available(self, domain, task):
        """Ensure model is available locally, download if needed"""
        model_config = self.config[domain][task]
        model_type = model_config.get('model_type')
        
        print(f"Ensuring model availability for {domain}/{task}")
        
        if model_type == 'huggingface':
            return self._check_huggingface_model(model_config, domain, task)
        elif model_type == 'custom':
            return self._download_custom_model(model_config, domain, task)
        
        return True
    
    def load_expert(self, domain, task):
        """Dynamically load the appropriate expert model"""
        try:
            # First ensure model is available
            if not self.ensure_model_available(domain, task):
                print(f"Failed to ensure model availability for {domain}/{task}")
                return None
            
            # Get configuration for this domain/task with updated cache path
            model_config = self.config[domain][task].copy()
            
            # Update cache_dir to use domain-based structure
            cache_dir = self._get_domain_cache_path(domain, task)
            model_config['cache_dir'] = str(cache_dir)
            
            # Dynamically import the expert class
            module_name = f"experts.{domain}.{task}_model"
            class_name = f"{domain.capitalize()}{task.replace('_', '').capitalize()}Expert"
            
            module = import_module(module_name)
            expert_class = getattr(module, class_name)
            
            # Create and return expert instance
            return expert_class(
                model_config=model_config,
                config_path="config/model_config.json"
            )
            
        except Exception as e:
            print(f"Error loading expert for {domain}/{task}: {e}")
            # Fallback to base expert
            from experts.base_expert import BaseLLMExpert
            return BaseLLMExpert(f"{domain}_{task}")
    
    def download_all_models(self):
        """Download all configured models"""
        print("Downloading all configured models...")
        
        for domain, tasks in self.config.items():
            print(f"\nProcessing {domain.upper()} domain:")
            domain_dir = self.models_base_dir / domain
            print(f"Domain directory: {domain_dir}")
            
            for task, model_config in tasks.items():
                print(f"  Checking {task}...")
                task_dir = domain_dir / task
                print(f"    Task directory: {task_dir}")
                
                if self.ensure_model_available(domain, task):
                    print(f"    ✓ {task} model ready")
                else:
                    print(f"    ✗ {task} model failed to download")
    
    def get_model_info(self):
        """Get information about all configured models"""
        info = {}
        
        for domain, tasks in self.config.items():
            info[domain] = {}
            
            for task, model_config in tasks.items():
                model_type = model_config.get('model_type')
                model_name = model_config.get('model_name', 'N/A')
                cache_dir = self._get_domain_cache_path(domain, task)
                
                if model_type == 'huggingface':
                    exists = self._model_exists_locally(model_name, cache_dir)
                elif model_type == 'custom':
                    # Check in domain-specific directory
                    model_filename = Path(model_config.get('model_path', f'{task}_model.pkl')).name
                    model_path = cache_dir / model_filename
                    exists = model_path.exists()
                else:
                    exists = True
                
                info[domain][task] = {
                    'type': model_type,
                    'name': model_name,
                    'available': exists,
                    'path': str(cache_dir)
                }
        
        return info
    
    def get_directory_structure(self):
        """Get the current directory structure"""
        structure = {}
        
        if self.models_base_dir.exists():
            for domain_dir in self.models_base_dir.iterdir():
                if domain_dir.is_dir():
                    structure[domain_dir.name] = []
                    for task_dir in domain_dir.iterdir():
                        if task_dir.is_dir():
                            structure[domain_dir.name].append(task_dir.name)
        
        return structure
    
    def get_available_models(self):
        """Get list of all available models"""
        models = {}
        for domain, tasks in self.config.items():
            models[domain] = list(tasks.keys())
        return models
