# Multi-Stage Prompt Routing System with Reinforcement Learning
# Complete Implementation

import numpy as np
import random
from collections import deque
import os
import sys
from pathlib import Path
import re

# Add project root to Python path
project_root = Path(__file__).parents[6]  # Go up 6 levels to reach project root
sys.path.insert(0, str(project_root))

from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader

# 1. Language Detection Module
class LanguageDetector:
    def __init__(self):
        self.language_patterns = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'you', 'this'],
            'indonesian': ['dan', 'yang', 'di', 'untuk', 'dengan', 'dari', 'pada', 'ke', 'ini', 'itu', 'adalah'],
            'spanish': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'por', 'con'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']
        }
    
    def detect_language(self, text):
        text_lower = text.lower().split()
        scores = {}
        
        for lang, keywords in self.language_patterns.items():
            score = sum(1 for word in text_lower if word in keywords)
            scores[lang] = score
        
        if any(scores.values()):
            return max(scores, key=scores.get)
        return 'english'

# 2. Domain Classification Module
class DomainClassifier:
    def __init__(self):
        self.domain_keywords = {
            'finance': ['market', 'stock', 'price', 'investment', 'trading', 'portfolio', 'risk', 'return', 
                       'bank', 'money', 'revenue', 'profit', 'analysis', 'economic', 'financial'],
            'general': ['help', 'question', 'what', 'how', 'why', 'when', 'where', 'explain', 'summary']
        }
        
    def classify_domain(self, text):
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[domain] = score
        
        return max(scores, key=scores.get)

# 3. Task Expert Models
class TaskExpert:
    def __init__(self, task_name):
        self.task_name = task_name
        
    def predict(self, text):
        # In production, this would be your actual trained model
        confidence = random.uniform(0.6, 0.95)
        prediction = f"{self.task_name}_result"
        return prediction, confidence

# 4. Simple Neural Network for RL Agent
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

# 5. PPO Agent for Task Routing
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = SimpleNN(state_dim, 64, action_dim, lr)
        self.value_net = SimpleNN(state_dim, 64, 1, lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        
    def get_action(self, state):
        state = np.array(state).reshape(1, -1)
        action_probs = self.policy_net.forward(state)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action, action_probs[action]
    
    def store_transition(self, state, action, reward, action_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_probs.append(action_prob)
    
    def update(self):
        if len(self.states) == 0:
            return 0
            
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        old_action_probs = np.array(self.action_probs)
        
        # Calculate advantages (simplified)
        values = self.value_net.forward(states).flatten()
        advantages = rewards - values
        
        # Update value network
        targets = rewards.reshape(-1, 1)
        self.value_net.backward(states, targets, self.value_net.forward(states))
        
        # Update policy network (simplified PPO)
        current_action_probs = self.policy_net.forward(states)
        
        # Create one-hot encoded actions for loss calculation
        y_target = np.zeros_like(current_action_probs)
        for i, action in enumerate(actions):
            y_target[i, action] = advantages[i]
        
        self.policy_net.backward(states, y_target, current_action_probs)
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        
        return np.mean(np.abs(advantages))

# 6. Complete Prompt Routing System
class PromptRoutingSystem:
    def __init__(self):
        
        config_path = Path(__file__).parents[4] / "experts" / "config"
        
        self.language_detector = LanguageDetector()
        self.domain_classifier = DomainClassifier()
        self.model_loader = ModelLoader(config_path / "model_config.json")
        self.domain_tasks = DomainTaskLoader(config_path / "domain_tasks.json")
        
        print("Checking and downloading models if needed...")
        self.model_loader.download_all_models()
        
        # Initialize task experts
        self.experts = {}
        for domain, tasks in self.domain_tasks.items():
            self.experts[domain] = {}
            for task in tasks.keys():
                self.experts[domain][task] = TaskExpert(task)
        
        # Initialize RL agents for each domain
        self.routing_agents = {}
        for domain in self.domain_tasks.keys():
            num_tasks = len(self.domain_tasks[domain])
            self.routing_agents[domain] = PPOAgent(state_dim=10, action_dim=num_tasks)
    
    def extract_features(self, prompt):
        """Extract features from prompt for RL state"""
        features = np.zeros(10)
        prompt_lower = prompt.lower()
        
        # Basic features
        features[0] = min(len(prompt) / 1000, 1.0)  # Length
        features[1] = 1.0 if '?' in prompt else 0.0  # Question
        features[2] = min(len(prompt.split()) / 50, 1.0)  # Word count
        
        # Domain-specific keywords
        finance_kw = ['market', 'stock', 'price', 'trading', 'investment']
        tech_kw = ['code', 'AI', 'algorithm', 'software', 'programming']
        health_kw = ['patient', 'medical', 'treatment', 'symptom']
        
        features[3] = sum(1 for kw in finance_kw if kw in prompt_lower) / len(finance_kw)
        features[4] = sum(1 for kw in tech_kw if kw in prompt_lower) / len(tech_kw)
        features[5] = sum(1 for kw in health_kw if kw in prompt_lower) / len(health_kw)
        
        # Task indicators
        features[6] = 1.0 if any(w in prompt_lower for w in ['analyze', 'sentiment']) else 0.0
        features[7] = 1.0 if any(w in prompt_lower for w in ['predict', 'forecast']) else 0.0
        features[8] = 1.0 if any(w in prompt_lower for w in ['generate', 'create', 'write']) else 0.0
        features[9] = 1.0 if any(w in prompt_lower for w in ['detect', 'find', 'debug']) else 0.0
        
        return features
    
    def classify_task(self, prompt, domain):
        """Classify task within domain using keyword matching"""
        prompt_lower = prompt.lower()
        task_scores = {}
        
        for task, keywords in self.domain_tasks[domain].items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            task_scores[task] = score
        
        if max(task_scores.values()) > 0:
            return max(task_scores, key=task_scores.get)
        else:
            return list(self.domain_tasks[domain].keys())[0]
    
    def route_prompt(self, prompt, use_rl=False):
        """Route prompt through the pipeline"""
        # Step 1: Language Detection
        language = self.language_detector.detect_language(prompt)
        
        # Step 2: Domain Classification
        domain = self.domain_classifier.classify_domain(prompt)
        
        # Step 3: Task Classification
        if use_rl:
            features = self.extract_features(prompt)
            agent = self.routing_agents[domain]
            task_idx, routing_confidence = agent.get_action(features)
            task = list(self.domain_tasks[domain].keys())[task_idx]
        else:
            task = self.classify_task(prompt, domain)
            routing_confidence = 0.8  # Default confidence
        
        # Step 4: Expert Processing
        expert = self.experts[domain][task]
        result, expert_confidence = expert.predict(prompt)
        
        return {
            'input': prompt,
            'language': language,
            'domain': domain,
            'task': task,
            'result': result,
            'routing_confidence': routing_confidence,
            'expert_confidence': expert_confidence,
            'routing_path': f"{language} → {domain} → {task}"
        }
    
    def train_agents(self, training_data, epochs=50):
        """Train the RL routing agents"""
        print(f"Training routing agents for {epochs} epochs...")
        
        for epoch in range(epochs):
            for sample in training_data:
                prompt = sample['prompt']
                correct_domain = sample['domain']
                correct_task = sample['task']
                
                # Only train if domain classification is correct
                predicted_domain = self.domain_classifier.classify_domain(prompt)
                if predicted_domain == correct_domain:
                    features = self.extract_features(prompt)
                    agent = self.routing_agents[correct_domain]
                    
                    correct_task_idx = list(self.domain_tasks[correct_domain].keys()).index(correct_task)
                    predicted_task_idx, action_prob = agent.get_action(features)
                    
                    reward = 1.0 if predicted_task_idx == correct_task_idx else -0.5
                    agent.store_transition(features, predicted_task_idx, reward, action_prob)
            
            # Update agents periodically
            if epoch % 10 == 0:
                for domain, agent in self.routing_agents.items():
                    loss = agent.update()
                print(f"Epoch {epoch} completed")
        
        print("Training completed!")
    
    def batch_process(self, prompts, use_rl=False):
        """Process multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.route_prompt(prompt, use_rl=use_rl)
            results.append(result)
        return results
    
    def get_system_stats(self):
        """Get system statistics"""
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.values())
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': len(self.language_detector.language_patterns),
            'domains': list(self.domain_tasks.keys())
        }

# 7. Utility Functions
def create_sample_training_data():
    """Create sample training data for demonstration"""
    return [
        {'prompt': 'What is the stock price of Apple?', 'domain': 'finance', 'task': 'price_prediction'},
        {'prompt': 'Analyze sentiment of this earnings report', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Assess investment risk of cryptocurrency', 'domain': 'finance', 'task': 'risk_assessment'},
        {'prompt': 'Detect fraud in this transaction', 'domain': 'finance', 'task': 'fraud_detection'},
        {'prompt': 'Write Python code for sorting', 'domain': 'technology', 'task': 'code_generation'},
        {'prompt': 'Find bugs in this JavaScript code', 'domain': 'technology', 'task': 'bug_detection'},
        {'prompt': 'Optimize system performance', 'domain': 'technology', 'task': 'system_optimization'},
        {'prompt': 'What is the capital of France?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Summarize this research paper', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Classify these documents by type', 'domain': 'general', 'task': 'classification'}
    ]

# 8. Usage Example and Testing
if __name__ == "__main__":
    # Initialize the system
    print("Initializing Prompt Routing System...")
    system = PromptRoutingSystem()
    
    # Display system information
    stats = system.get_system_stats()
    print(f"System initialized with {stats['total_domains']} domains and {stats['total_tasks']} tasks")
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Available domains: {stats['domains']}")
    
    # Sample training data
    training_data = create_sample_training_data()
    
    # Optional: Train the RL agents
    train_rl = input("\nWould you like to train RL agents? (y/n): ").lower() == 'y'
    if train_rl:
        system.train_agents(training_data, epochs=20)
    
    # Test prompts
    test_prompts = [
        "What is the current sentiment for Tesla stock?",
        "How do I fix this Python error?",
        "What are the symptoms of COVID-19?",
        # "Summarize the key findings from this report",
        "puedes compartir el texto del informe o los hallazgos principales?",
        "Predict Bitcoin price for next week",
        "Generate code for binary search algorithm",
        "Assess risk in my investment portfolio"
    ]
    
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    print("=" * 70)
    
    # Process prompts
    for i, prompt in enumerate(test_prompts, 1):
        result = system.route_prompt(prompt, use_rl=train_rl)
        print(f"Test {i}: {prompt}")
        print(f"   Route: {result['routing_path']}")
        print(f"   Result: {result['result']}")
        print(f"   Confidence: {result['expert_confidence']:.3f}")
        print()
    
    # Batch processing example
    print("Batch processing all test prompts...")
    batch_results = system.batch_process(test_prompts, use_rl=train_rl)
    
    print(f"Successfully processed {len(batch_results)} prompts")
    print("\nSystem ready for production use!")