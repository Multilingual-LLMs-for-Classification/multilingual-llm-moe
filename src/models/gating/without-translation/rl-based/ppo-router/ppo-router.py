# Multi-Stage Prompt Routing System with Reinforcement Learning
# Complete Implementation

import numpy as np
import random
from collections import deque
import os
import sys
from pathlib import Path
import re
import fasttext
import requests
import json
from prompts_multilingual import test_prompts

# Add project root to Python path
project_root = Path(__file__).parents[6]  # Go up 6 levels to reach project root
sys.path.insert(0, str(project_root))

from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader

# 1. Language Detection Module
class LanguageDetector:
    def __init__(self):
        self.model_path = Path(__file__).parent.parent / "models" / "lid.176.bin"
        self.model = None
        self._load_fasttext_model()
        
        self.language_mapping = {
            '__label__de': 'german',
            '__label__en': 'english',
            '__label__es': 'spanish',
            '__label__fr': 'french',
            '__label__ja': 'japanese',
            '__label__zh': 'chinese'
        }
    
    def _load_fasttext_model(self):
        try:
            if not self.model_path.exists():
                self._download_fasttext_model()
            self.model = fasttext.load_model(str(self.model_path))
            print("✅ FastText language model loaded")
        except Exception as e:
            print(f"⚠️ FastText model failed, using fallback: {e}")
            self.model = None
    
    def _download_fasttext_model(self):
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("Downloading FastText model...")
        response = requests.get(url)
        with open(self.model_path, 'wb') as f:
            f.write(response.content)
        print("✅ FastText model downloaded")
    
    def detect_language(self, text):
        if self.model is None:
            print("Using fallback language detection")
            return self._fallback_detection(text)
        
        try:
            # print("Using FastText for language detection")
            # Clean the text first
            cleaned_text = text.replace('\n', ' ').strip()
            if len(cleaned_text) < 3:
                return 'english'
            
            # FastText expects single line text
            predictions = self.model.predict(cleaned_text, k=1)
            print("Predictions:", predictions)
            
            # Extract the result properly
            labels, scores = predictions
            detected_lang = labels[0]  # Get the first (and only) label
            confidence = scores[0]     # Get the confidence score
            
            # print(f"Detected language code: {detected_lang}, confidence: {confidence}")
            
            # Map to your language names
            mapped_lang = self.language_mapping.get(detected_lang, 'english')
            # print(f"Mapped to: {mapped_lang}")
            
            return mapped_lang
            
        except Exception as e:
            print(f"FastText detection failed with error: {e}")
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text):
        """Enhanced fallback detection for target languages"""
        
        patterns = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'you', 'this'],
            'german': ['der', 'die', 'das', 'und', 'ist', 'ich', 'nicht', 'ein', 'eine', 'zu', 'den', 'von'],
            'spanish': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'por'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans'],
            'japanese': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる'],
            'chinese': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上']
        }
    
        text_lower = text.lower()
        scores = {}
        
        # For CJK languages, check character-by-character
        if any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese characters
            scores['chinese'] = len([char for char in text if '\u4e00' <= char <= '\u9fff'])
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):  # Japanese characters
            scores['japanese'] = len([char for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'])
        
        # For Latin script languages, check words
        text_words = text_lower.split()
        for lang, keywords in patterns.items():
            if lang not in ['japanese', 'chinese']:  # Skip CJK for word-based detection
                score = sum(1 for word in text_words if word in keywords)
                scores[lang] = score
        
        return max(scores, key=scores.get) if any(scores.values()) else 'english'

# 2. Domain Classification Module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import numpy as np

class DomainClassifier:
    def __init__(self):
        self.pipeline = None
        self.domains = ['finance', 'general']
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        # Create training data programmatically
        training_data = self._create_training_data()
        
        # Create ML pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',     # <-- character n-grams, robust across languages
                ngram_range=(3, 5),     # tri- to 5-grams
                max_features=50000,     # raise capacity a bit
                lowercase=True          # ok even for non-Latin scripts
                # NOTE: remove stop_words='english'
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train the classifier
        texts, labels = zip(*training_data)
        self.pipeline.fit(texts, labels)
        print("✅ Domain classifier trained")
    
    def _create_training_data(self):
        data = []

        # EN
        finance_en = [
            "stock market analysis", "investment portfolio management", "financial risk assessment",
            "trading strategies", "market volatility", "revenue growth", "profit margins"
        ]
        general_en = [
            "how to solve this problem", "explain this concept", "summary of information",
            "why does this happen"
        ]
        data += [(x, 'finance') for x in finance_en]
        data += [(x, 'general') for x in general_en]

        # DE
        finance_de = [
            "Börsenvolatilität", "Anlagestrategien", "Rendite und Risiko", "Finanzanalyse von Umsatz und Gewinn"
        ]
        general_de = [
            "Erkläre dieses Konzept", "Zusammenfassung des Artikels", "Unterschied zwischen RAM und Speicher", "Warum tritt dieser Fehler auf"
        ]
        data += [(x, 'finance') for x in finance_de]
        data += [(x, 'general') for x in general_de]

        # ES
        finance_es = [
            "volatilidad del mercado", "cartera de inversión", "análisis financiero de ingresos y beneficios",
            "estrategia de trading"
        ]
        general_es = [
            "explica este concepto", "resumen del artículo", "diferencia entre RAM y almacenamiento", "por qué ocurre este error"
        ]
        data += [(x, 'finance') for x in finance_es]
        data += [(x, 'general') for x in general_es]

        # FR
        finance_fr = [
            "volatilité du marché", "gestion de portefeuille", "analyse financière des revenus et profits",
            "stratégie de trading"
        ]
        general_fr = [
            "expliquer ce concept", "résumé de l'article", "différence entre RAM et stockage", "pourquoi cette erreur se produit"
        ]
        data += [(x, 'finance') for x in finance_fr]
        data += [(x, 'general') for x in general_fr]

        # JA
        finance_ja = [
            "株式市場のボラティリティ", "ポートフォリオのリスク", "金融分析", "取引戦略"
        ]
        general_ja = [
            "この概念を説明", "記事の要約", "RAMとストレージの違い", "なぜこのエラーが発生"
        ]
        data += [(x, 'finance') for x in finance_ja]
        data += [(x, 'general') for x in general_ja]

        # ZH (Simplified)
        finance_zh = [
            "股票市场波动", "投资组合风险", "金融分析", "交易策略"
        ]
        general_zh = [
            "解释这个概念", "文章摘要", "RAM和存储的区别", "为什么会出现这个错误"
        ]
        data += [(x, 'finance') for x in finance_zh]
        data += [(x, 'general') for x in general_zh]

        return data

    
    def classify_domain(self, text):
        if self.pipeline is None:
            print("Using fallback domain classification")
            return self._fallback_classification(text)
        
        try:
            prediction = self.pipeline.predict([text])[0]
            print(f"Predicted domain: {prediction}")
            
            probs = self.get_domain_probabilities(text)
            print(f"Domain probabilities: {probs}")
            return prediction
        except:
            print("Failed to classify domain, using fallback")
            return self._fallback_classification(text)
    
    def get_domain_probabilities(self, text):
        """Get probabilities for all domains"""
        if self.pipeline is None:
            return {'finance': 0.5, 'general': 0.5}
        
        try:
            probabilities = self.pipeline.predict_proba([text])[0]
            return dict(zip(self.domains, probabilities))
        except:
            print("Failed to get domain probabilities, using fallback")
            return {'finance': 0.5, 'general': 0.5}
    
    def _fallback_classification(self, text):
        # Your existing keyword-based classification
        domain_keywords = {
            'finance': ['market', 'stock', 'price', 'investment', 'trading', 'portfolio', 'risk', 'return', 
                       'bank', 'money', 'revenue', 'profit', 'analysis', 'economic', 'financial'],
            'general': ['help', 'question', 'what', 'how', 'why', 'when', 'where', 'explain', 'summary']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[domain] = score
        
        return max(scores, key=scores.get)

# 3. Task Classification Module with PPO
class TaskClassifier:
    def __init__(self, domain_tasks):
        self.domain_tasks = domain_tasks
        self.task_pipelines = {}
        self.ppo_agents = {}
        self._initialize_task_classifiers()
    
    def _initialize_task_classifiers(self):
        for domain, tasks in self.domain_tasks.items():
            # Create ML classifier for each domain
            self.task_pipelines[domain] = self._create_task_pipeline(domain, tasks)
            
            # Create PPO agent for task routing optimization
            num_tasks = len(tasks)
            self.ppo_agents[domain] = PPOAgent(state_dim=15, action_dim=num_tasks)
    
    def _create_task_pipeline(self, domain, tasks):
        # Create task-specific training data
        training_data = self._create_task_training_data(domain, tasks)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        if training_data:
            texts, labels = zip(*training_data)
            pipeline.fit(texts, labels)
        
        return pipeline
    
    def _create_task_training_data(self, domain, tasks):
        # Generate task-specific training data
        task_samples = {
            'sentiment_analysis': [
                "analyze sentiment of this text", "what's the mood of this review",
                "positive or negative opinion", "emotional tone analysis",
                "how does this make you feel", "determine the sentiment",
                "is this positive or negative", "sentiment classification"
            ],
            'news_classification': [
                "classify this news article", "what category is this news",
                "categorize this headline", "what type of news is this",
                "classify this media report", "what topic does this article cover",
                "determine news category", "news article classification"
            ],
            'question_answering': [
                "what is the answer", "can you explain", "help me understand",
                "provide information about"
            ],
            'text_summarization': [
                "summarize this document", "brief overview of", "key points summary",
                "condensed version of text"
            ],
            'classification': [
                "classify this item", "categorize the document", "sort into groups",
                "organize by type"
            ]
        }
        
        training_data = []
        for task in tasks:
            if task in task_samples:
                for sample in task_samples[task]:
                    training_data.append((sample, task))
        
        return training_data
    
    def classify_task(self, text, domain, use_ppo=False):
        print(f"Classifying task for domain: {domain} with PPO: {use_ppo}")
        if use_ppo:
            return self._ppo_task_classification(text, domain)
        else:
            return self._ml_task_classification(text, domain)
    
    def _ml_task_classification(self, text, domain):
        """Traditional ML task classification"""
        if domain not in self.task_pipelines:
            return list(self.domain_tasks[domain].keys())[0]
        
        try:
            pipeline = self.task_pipelines[domain]
            prediction = pipeline.predict([text])[0]
            return prediction
        except:
            return list(self.domain_tasks[domain].keys())[0]
    
    def _ppo_task_classification(self, text, domain):
        """PPO-enhanced task classification"""
        features = self._extract_task_features(text, domain)
        print(f"Extracted features: {features}")
        agent = self.ppo_agents[domain]
        
        task_idx, confidence = agent.get_action(features)
        print(f"PPO selected task index: {task_idx} with confidence {confidence:.4f}")
        task_names = list(self.domain_tasks[domain].keys())
        print(f"Available tasks: {task_names}")
        
        if task_idx < len(task_names):
            return task_names[task_idx]
        else:
            return task_names[0]
    
    def _extract_task_features(self, text, domain):
        """Extract features for PPO task classification"""
        features = np.zeros(15)
        text_lower = text.lower()
        
        # Basic text features
        features[0] = len(text) / 1000  # Normalized length
        features[1] = len(text.split()) / 50  # Normalized word count
        features[2] = 1.0 if '?' in text else 0.0  # Question indicator
        
        # Task-specific keyword features
        # task_keywords = {
        #     'sentiment': ['sentiment', 'feeling', 'opinion', 'positive', 'negative'],
        #     'risk': ['risk', 'danger', 'safety', 'assess', 'evaluate'],
        #     'prediction': ['predict', 'forecast', 'estimate', 'future'],
        #     'detection': ['detect', 'find', 'identify', 'discover'],
        #     'question': ['what', 'how', 'why', 'when', 'where'],
        #     'summary': ['summary', 'brief', 'overview', 'key points'],
        #     'classification': ['classify', 'categorize', 'sort', 'group']
        # }
        task_keywords = {
            'sentiment': ['sentiment', 'feeling', 'opinion', 'positive', 'negative', 'mood', 'emotion'],
            'news': ['news', 'headline', 'article', 'press', 'media', 'report', 'story'],
            'classification': ['classify', 'categorize', 'sort', 'group', 'category', 'type'],
            'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
            'summary': ['summary', 'brief', 'overview', 'key points', 'summarize'],
            'analysis': ['analyze', 'analysis', 'examine', 'evaluate', 'assess'],
            'financial': ['stock', 'market', 'price', 'investment', 'trading', 'finance']
        }
        
        feature_idx = 3
        for category, keywords in task_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower) / len(keywords)
            features[feature_idx] = score
            feature_idx += 1
        
        # Domain-specific features
        if domain == 'finance':
            finance_kw = ['market', 'stock', 'investment', 'trading', 'financial']
            features[10] = sum(1 for kw in finance_kw if kw in text_lower) / len(finance_kw)
        
        # Additional contextual features
        features[11] = text.count('!') / max(len(text), 1)  # Exclamation density
        features[12] = text.count('.') / max(len(text), 1)  # Period density
        features[13] = len([w for w in text.split() if w.isupper()]) / max(len(text.split()), 1)  # Caps ratio
        features[14] = len(set(text.lower().split())) / max(len(text.split()), 1)  # Unique word ratio
        
        return features
    
    def train_ppo_agent(self, domain, training_data):
        """Train PPO agent for specific domain"""
        agent = self.ppo_agents[domain]
        task_names = list(self.domain_tasks[domain].keys())
        
        for epoch in range(50):
            for sample in training_data:
                if sample['domain'] == domain:
                    text = sample['prompt']
                    correct_task = sample['task']
                    
                    features = self._extract_task_features(text, domain)
                    predicted_idx, action_prob = agent.get_action(features)
                    
                    correct_idx = task_names.index(correct_task) if correct_task in task_names else 0
                    reward = 1.0 if predicted_idx == correct_idx else -0.5
                    
                    agent.store_transition(features, predicted_idx, reward, action_prob)
            
            if epoch % 10 == 0:
                loss = agent.update()
                print(f"Domain {domain} - Epoch {epoch}, Loss: {loss:.4f}")

# 4. Task Expert Models
class TaskExpert:
    def __init__(self, task_name):
        self.task_name = task_name
        
    def predict(self, text):
        # In production, this would be your actual trained model
        confidence = random.uniform(0.1, 0.2)
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
        
        # Initialize components
        self.language_detector = LanguageDetector()  # FastText-based
        self.domain_classifier = DomainClassifier()  # ML-based
        self.model_loader = ModelLoader(config_path / "model_config.json")
        self.domain_tasks = DomainTaskLoader(config_path / "domain_tasks.json")
        
        # Initialize task classifier with PPO
        self.task_classifier = TaskClassifier(self.domain_tasks.domain_tasks)
        
        # Download models and initialize experts
        print("Checking and downloading models if needed...")
        self.model_loader.download_all_models()
        
        self.experts = {}
        for domain, tasks in self.domain_tasks.items():
            self.experts[domain] = {}
            for task in tasks.keys():
                self.experts[domain][task] = TaskExpert(task)
        
        # Use PPO agents from task classifier
        self.routing_agents = self.task_classifier.ppo_agents
    
    def route_prompt(self, prompt, use_ppo=False):
        # Step 1: FastText Language Detection
        language = self.language_detector.detect_language(prompt)
        
        # Step 2: ML-based Domain Classification
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        
        # Step 3: PPO-enhanced Task Classification
        task = self.task_classifier.classify_task(prompt, domain, use_ppo=use_ppo)
        print(f"task: {task}")
        
        # Step 4: Expert Processing
        expert = self.experts[domain][task]
        result, expert_confidence = expert.predict(prompt)
        
        return {
            'input': prompt,
            'language': language,
            'domain': domain,
            'domain_probabilities': domain_probs,
            'task': task,
            'result': result,
            'expert_confidence': expert_confidence,
            'routing_path': f"{language} → {domain} → {task}"
        }
    
    def train_ppo_agents(self, training_data):
        """Train all PPO agents"""
        for domain in self.domain_tasks.keys():
            print(f"Training PPO agent for {domain} domain...")
            self.task_classifier.train_ppo_agent(domain, training_data)
    
    def batch_process(self, prompts, use_ppo=False):
        """Process multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.route_prompt(prompt, use_ppo=use_ppo)
            results.append(result)
        return results
    
    def get_system_stats(self):
        """Get system statistics"""
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.values())
        supported_languages = len(self.language_detector.language_mapping)
        
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': supported_languages,
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
    
    # Optional: Train the PPO agents
    train_rl = input("\nWould you like to train PPO agents? (y/n): ").lower() == 'y'
    if train_rl:
        system.train_ppo_agents(training_data)  # Use new training method
    
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    print("=" * 70)
    
    # Process prompts with PPO
    for i, prompt in enumerate(test_prompts, 1):
        result = system.route_prompt(prompt, use_ppo=train_rl)
        print(f"Test {i}: {prompt}")
        print(f"Route: {result['routing_path']}")
        print(f"Result: {result['result']}")
        print(f"Confidence: {result['expert_confidence']:.3f}")
        print()
    
    # Batch processing example
    # print("Batch processing all test prompts...")
    # batch_results = system.batch_process(test_prompts, use_ppo=train_rl)
    
    # print(f"Successfully processed {len(batch_results)} prompts")
    # print("\nSystem ready for production use!")
