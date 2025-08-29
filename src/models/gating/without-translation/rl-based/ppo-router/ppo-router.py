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
import joblib
import pickle
import numpy as np

class DomainClassifier:
    def __init__(self):
        self.pipeline = None
        self.domains = ['finance', 'general']
        self.model_path = Path(__file__).parent.parent / "models" / "domain_classifier.joblib"
        self._initialize_classifier()
    
    def save_model(self, filepath=None):
        """Save the trained domain classifier model"""
        if filepath is None:
            filepath = self.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the trained pipeline
        joblib.dump(self.pipeline, filepath)
        print(f"✅ Domain classifier saved to: {filepath}")
    
    def load_model(self, filepath=None):
        """Load a pre-trained domain classifier model"""
        if filepath is None:
            filepath = self.model_path
        
        if os.path.exists(filepath):
            self.pipeline = joblib.load(filepath)
            print(f"✅ Domain classifier loaded from: {filepath}")
            return True
        else:
            print(f"⚠️ No saved model found at: {filepath}")
            return False
    
    # def _initialize_classifier(self):
    #     training_data = self._create_training_data()
        
    #     self.pipeline = Pipeline([
    #         ('tfidf', TfidfVectorizer(
    #             analyzer='char_wb',     # <-- character n-grams, robust across languages
    #             ngram_range=(3, 5),     # tri- to 5-grams
    #             max_features=50000,     # raise capacity a bit
    #             lowercase=True          # ok even for non-Latin scripts
    #         )),
    #         ('classifier', MultinomialNB(alpha=0.1))
    #     ])
        
    #     # Train the classifier
    #     texts, labels = zip(*training_data)
    #     self.pipeline.fit(texts, labels)
    #     print("✅ Domain classifier trained")
    
    def _initialize_classifier(self):
        # Try to load existing model first
        if not self.load_model():
            # If no saved model, train new one
            print("Training new domain classifier...")
            training_data = self._create_training_data()
            
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    max_features=50000,
                    lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            texts, labels = zip(*training_data)
            self.pipeline.fit(texts, labels)
            print("✅ Domain classifier trained")
            
            # Automatically save the newly trained model
            self.save_model()
    
    def _create_training_data(self):
        data = []

        # ENGLISH - Expanded Finance Training
        finance_en = [
            # Core financial terms
            "stock market analysis", "investment portfolio management", "financial risk assessment",
            "trading strategies", "market volatility", "revenue growth", "profit margins",
            
            # Sentiment analysis finance terms
            "sentiment analysis of earnings report", "positive financial news about Apple", 
            "negative opinion about Tesla stock", "emotional tone of financial article",
            "is this financial news positive or negative", "analyze market sentiment",
            "financial sentiment analysis", "stock sentiment",
            
            # News classification finance terms  
            "classify business news", "financial article categorization", "economic report category",
            "business sector classification", "financial news classification", "market update classification",
            "categorize financial article", "business industry",
            
            # General finance terms
            "Apple stock analysis", "Tesla earnings report", "economic outlook", "business performance",
            "financial sector news", "investment analysis", "market trends", "economic indicators",
            "quarterly earnings", "stock price movement", "market sentiment", "financial markets",
            "business analysis", "economic data", "financial performance", "market research"
        ]
        
        general_en = [
            # Core general terms
            "how to solve this problem", "explain this concept", "summary of information",
            "why does this happen", "help me understand", "what is the meaning",
            "define this term", "how does it work", "tell me about", "what is the capital",
            
            # Classification and organization
            "classify documents", "categorize items", "organize files",
            "sort data", "group items", "arrange by classification",
            
            # Summarization
            "summarize article", "key points", "overview", "brief explanation", 
            "condensed information", "main points", "document summary", "text overview",
            
            # Q&A and support
            "answer question", "provide explanation", "give information", "help understanding",
            "technical support", "troubleshooting", "user guidance", "information request"
        ]

        # GERMAN - Enhanced Finance Training
        finance_de = [
            "Börsenvolatilität", "Anlagestrategien", "Finanzrisikobewertung", "Handelsstrategien", "Marktvolatilität",
            "Umsatzwachstum", "Gewinnmargen", "Stimmungsanalyse des Gewinnberichts", "positive Finanznachrichten",
            "negative Meinung zu Tesla-Aktien", "emotionale Tonalität", "positive und negative",
            "Analyse von Marktstimmungen", "Finanzstimmung", "Aktienstimmung",
            "wirtschafliche Nachrichten klassifizieren", "finanzielle Kategorisierung", "ökonomisches Berichtskategorie",
            "branchenbezogene Klassifikation", "finanzielle Nachrichten klassifizieren", "Marktupdates kategorisieren",
            "finanzielle Artikel nach Branche", "zu Branchen zuordnen", "wirtschafliche Schlagzeilen klassifizieren",
            "Analyse von Aktien", "Bericht über Tesla", "wirtschaftliche Aussichten", "Unternehmensleistung",
            "Nachrichten im Finanzsektor", "Investitionsanalyse", "Markttrends", "wirtschaftliche Indikatoren",
            "Quartalsgewinne", "Aktienkursbewegungen", "Marktstimmung", "Finanzmärkte", "Geschäftsanalyse",
            "Wirtschaftsdaten", "finanzielle Leistung", "Marktforschung"
        ]
        
        general_de = [
            "wie löst man dieses Problem", "erkläre dieses Konzept", "zusammenfassung",
            "warum passiert das", "hilf mir zu verstehen", "was bedeutet das",
            "definiere den Begriff", "wie funktioniert das", "erzähl mir davon", "was ist die Hauptstadt",
            "Dokumente klassifizieren", "Elemente kategorisieren", "Dateien organisieren",
            "Daten sortieren", "Elemente gruppieren", "ordnen",
            "Artikel zusammenfassen", "wichtige Punkte", "übersicht", "kurze Erklärung",
            "komprimierte Informationen", "Hauptpunkte", "Dokumentzusammenfassung", "Textübersicht",
            "Fragen beantworten", "Erklärungen geben", "Informationen bereitstellen", "Hilfen anbieten",
            "technischer Support", "Fehlerbehebung", "Benutzerhilfe", "Informationsanfrage"
        ]

        # SPANISH - Enhanced Finance Training
        finance_es = [
            "volatilidad del mercado", "gestión de cartera", "evaluación de riesgos financieros", "estrategias de trading",
            "crecimiento de ingresos", "márgenes de beneficio", "análisis de sentimiento", "noticias financieras positivas",
            "opiniones negativas sobre Tesla", "tono emocional", "positivo o negativo",
            "análisis de sentimientos", "sentimiento de acciones",
            "clasificar noticias", "categorización de artículos financieros", "categoría de informes",
            "clasificación por sector", "noticias clasificadas", "actualización de mercado",
            "artículos por industria", "pertenencia a sector", "clasificación de titulares",
            "análisis de acciones", "informe de Tesla", "perspectivas económicas", "desempeño empresarial",
            "noticias del sector financiero", "análisis de inversiones", "tendencias del mercado", "indicadores económicos",
            "ganancias trimestrales", "movimiento de precios de acciones", "sentimiento en el mercado", "mercados financieros",
            "análisis empresarial", "datos económicos", "rendimiento financiero", "investigación de mercado"
        ]
        
        general_es = [
            "cómo resolver problemas", "explicar conceptos", "resumen de información", "por qué sucede",
            "ayuda para entender", "qué significa", "definir términos", "cómo funciona",
            "contar sobre", "capital de país", "clasificar documentos", "categorizar objetos",
            "organizar archivos", "ordenar datos", "agrupar objetos", "arreglar por clasificación",
            "resumir artículos", "puntos clave", "visión general", "explicación breve",
            "información condensada", "extracción de puntos", "resumen del documento", "visión general de texto",
            "responder preguntas", "proporcionar explicaciones", "dar información", "ayuda para entender",
            "soporte técnico", "ayuda para problemas", "guía para usuarios", "solicitud de información"
        ]

        # FRENCH - Enhanced Finance Training  
        finance_fr = [
            "volatilité du marché", "gestion de portefeuille", "évaluation des risques financiers", "stratégies de trading",
            "croissance des revenus", "marges bénéficiaires", "analyse de sentiment", "nouvelles financières positives",
            "opinions négatives sur Tesla", "ton émotionnel", "positif ou négatif",
            "analyse des sentiments", "sentiment des actions",
            "classer les nouvelles", "catégorisation d'articles financiers", "catégorie de rapport",
            "classification sectorielle", "nouvelles classées", "mise à jour de marché",
            "articles par industrie", "appartenance à un secteur", "classification de titres",
            "analyse d'actions", "rapport Tesla", "perspectives économiques", "performance commerciale",
            "nouvelles secteur financier", "analyse des investissements", "tendances du marché", "indicateurs économiques",
            "bénéfices trimestriels", "mouvement des prix des actions", "sentiment du marché", "marchés financiers",
            "analyse commerciale", "données économiques", "performance financière", "recherches de marché"
        ]
        
        general_fr = [
            "comment résoudre les problèmes", "expliquer les concepts", "résumé d'informations", "pourquoi cela se produit",
            "aide à comprendre", "signification", "définir des termes", "comment cela fonctionne",
            "parler de", "capitale", "classer des documents", "catégoriser des éléments",
            "organiser des fichiers", "trier des données", "grouper des éléments", "arranger par classification",
            "résumer des articles", "points clés", "aperçu", "explication brève",
            "informations condensées", "extraction de points", "résumé de documents", "aperçu de texte",
            "répondre aux questions", "fournir des explications", "donner des informations", "aide à comprendre",
            "support technique", "dépannage", "guide utilisateur", "demande d'informations"
        ]

        # JAPANESE - Enhanced Finance Training
        finance_ja = [
            "株式市場の変動性", "ポートフォリオ管理", "リスク評価", "トレーディング戦略", "市場のボラティリティ",
            "収益の増加", "利益率", "収益報告の感情分析", "ポジティブな金融ニュース",
            "テスラの株に関するネガティブな意見", "感情のトーン", "これらの金融ニュースはプラスかマイナスか",
            "感情分析", "株式の感情", "経済ニュースの分類", "金融記事の分類",
            "経済レポートのカテゴリー", "産業の分類", "金融ニュースのカテゴライズ",
            "市場のアップデート分類", "業界別のカテゴリー",
            "これらのビジネスニュースはどのセクターに属するか",
            "経済レポートを分類してください", "経済ニュースのセクター別分類",
            "アップルの株の分析", "テスラの収益レポート", "経済の見通し", "事業のパフォーマンス",
            "金融セクターのニュース", "投資分析", "市場動向", "経済指標",
            "四半期の利益", "株価の動き", "市場の感情", "金融市場",
            "ビジネスの分析", "経済データ", "金融のパフォーマンス", "市場調査"
        ]
        
        general_ja = [
            "問題の解決方法", "概念を説明してください", "情報の要約", "なぜこうなるのか",
            "理解を助けて", "意味は何ですか", "用語を定義する", "どのように機能するか",
            "について教えて", "都道府県の首都", "ドキュメントを分類", "アイテムをカテゴライズ",
            "ファイルを整理", "データを篩い分け", "特徴ごとに分ける", "分類する",
            "記事を要約", "重要なポイント", "概要", "簡潔な説明",
            "内容を凝縮", "主なポイント", "文書の要約", "テキストの概要",
            "質問に答える", "説明を提供", "情報を伝える", "理解を助ける",
            "テクニカルサポート", "トラブルシューティング", "ユーザーガイド", "情報要請"
        ]

        # CHINESE - Enhanced Finance Training
        finance_zh = [
            "股票市场波动", "投资组合管理", "风险评估", "交易策略", "市场波动",
            "收入增长", "利润率", "收益报告的情感分析", "积极的金融新闻",
            "对特斯拉股票的负面意见", "情绪基调", "这条金融新闻是积极还是消极",
            "情绪分析", "股票情绪", "商业新闻分类", "金融文章分类",
            "经济报告类别", "行业分类", "金融新闻分类",
            "市场更新分类", "按行业分类的文章",
            "这些商业新闻属于哪个部门", "经济报告分类", "商业新闻部门分类",
            "苹果股票分析", "特斯拉收益报告", "经济展望", "业务表现",
            "金融部门新闻", "投资分析", "市场趋势", "经济指标",
            "季度收益", "股票价格变动", "市场情绪", "金融市场",
            "商业分析", "经济数据", "金融表现", "市场调查"
        ]
        
        general_zh = [
            "如何解决问题", "解释概念", "摘要信息", "为什么会这样",
            "帮助理解", "这是什么意思", "定义术语", "如何工作",
            "讲述", "首都", "文件分类", "项目分类",
            "文件整理", "数据筛选", "按特征分组", "分类",
            "文章摘要", "要点", "概述", "简要说明",
            "回答问题", "提供解释", "给出信息", "帮助理解",
            "技术支持", "故障排除", "用户指南", "信息请求"
        ]

        # Aggregate all data
        data += [(x, 'finance') for x in finance_en]
        data += [(x, 'general') for x in general_en]
        data += [(x, 'finance') for x in finance_de]
        data += [(x, 'general') for x in general_de]
        data += [(x, 'finance') for x in finance_es]
        data += [(x, 'general') for x in general_es]
        data += [(x, 'finance') for x in finance_fr]
        data += [(x, 'general') for x in general_fr]
        data += [(x, 'finance') for x in finance_ja]
        data += [(x, 'general') for x in general_ja]
        data += [(x, 'finance') for x in finance_zh]
        data += [(x, 'general') for x in general_zh]

        return data

    
    def classify_domain(self, text):
        if self.pipeline is None:
            print("Using fallback domain classification")
            return self._fallback_classification(text)
        
        try:
            prediction = self.pipeline.predict([text])[0]
            
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
        self.models_dir = Path(__file__).parent.parent / "models" / "task_classifiers"
        os.makedirs(self.models_dir, exist_ok=True)
        self._initialize_task_classifiers()
        
    def save_models(self):
        """Save all task classification models"""
        # Save ML pipelines
        for domain, pipeline in self.task_pipelines.items():
            pipeline_path = self.models_dir / f"{domain}_task_pipeline.joblib"
            joblib.dump(pipeline, pipeline_path)
            print(f"✅ Task pipeline for {domain} saved to: {pipeline_path}")
        
        # Save PPO agents
        for domain, agent in self.ppo_agents.items():
            agent.save_model(domain)
    
    def load_models(self):
        """Load all task classification models"""
        # Load ML pipelines
        for domain in self.domain_tasks.keys():
            pipeline_path = self.models_dir / f"{domain}_task_pipeline.joblib"
            if os.path.exists(pipeline_path):
                self.task_pipelines[domain] = joblib.load(pipeline_path)
                print(f"✅ Task pipeline for {domain} loaded from: {pipeline_path}")
            else:
                print(f"⚠️ No saved task pipeline found for {domain}")
                return False
        
        # Load PPO agents
        for domain in self.domain_tasks.keys():
            if not self.ppo_agents[domain].load_model(domain):
                return False
        
        return True
    
    def _initialize_task_classifiers(self):
        # Try to load existing models first
        if not self.load_models():
            print("Training new task classifiers...")
            for domain, tasks in self.domain_tasks.items():
                # Create ML classifier for each domain
                self.task_pipelines[domain] = self._create_task_pipeline(domain, tasks)
                
                # Create PPO agent for task routing optimization
                num_tasks = len(tasks)
                self.ppo_agents[domain] = PPOAgent(state_dim=15, action_dim=num_tasks)
            
            print("✅ Task classifiers initialized")
    
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
        agent = self.ppo_agents[domain]
        
        task_idx, confidence = agent.get_action(features)
        print(f"PPO selected task index: {task_idx} with confidence {confidence:.4f}")
        task_names = list(self.domain_tasks[domain].keys())
        
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
        
        for epoch in range(1000):
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
# class SimpleNN:
#     def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
#         self.W1 = np.random.randn(input_size, hidden_size) * 0.1
#         self.b1 = np.zeros((1, hidden_size))
#         self.W2 = np.random.randn(hidden_size, output_size) * 0.1
#         self.b2 = np.zeros((1, output_size))
#         self.learning_rate = learning_rate
        
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
#     def softmax(self, x):
#         exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
#     def forward(self, X):
#         self.z1 = np.dot(X, self.W1) + self.b1
#         self.a1 = self.sigmoid(self.z1)
#         self.z2 = np.dot(self.a1, self.W2) + self.b2
#         self.a2 = self.softmax(self.z2)
#         return self.a2
    
#     def backward(self, X, y, output):
#         m = X.shape[0]
        
#         # Calculate gradients
#         dz2 = output - y
#         dW2 = np.dot(self.a1.T, dz2) / m
#         db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
#         da1 = np.dot(dz2, self.W2.T)
#         dz1 = da1 * self.a1 * (1 - self.a1)
#         dW1 = np.dot(X.T, dz1) / m
#         db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
#         # Update weights
#         self.W2 -= self.learning_rate * dW2
#         self.b2 -= self.learning_rate * db2
#         self.W1 -= self.learning_rate * dW1
#         self.b1 -= self.learning_rate * db1
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Batch normalization
                nn.ReLU(),                   # ReLU activation
                nn.Dropout(0.2)              # Dropout for regularization
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)

class EnhancedValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

# Enhanced PPO Agent with PyTorch
class EnhancedPPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = EnhancedPolicyNet(state_dim, action_dim).to(self.device)
        self.value_net = EnhancedValueNet(state_dim).to(self.device)
        
        # Use Adam optimizer (better than vanilla SGD)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.memory = []
    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set networks to evaluation mode for inference
        self.policy_net.eval()
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        
        # Set back to training mode
        self.policy_net.train()
        
        return action.item(), action_log_prob.item()


# 5. PPO Agent for Task Routing
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create networks without BatchNorm to avoid single-sample issues
        self.policy_net = self._create_policy_net(state_dim, action_dim).to(self.device)
        self.value_net = self._create_value_net(state_dim).to(self.device)
        
        # Use Adam optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        
        self.model_dir = Path(__file__).parent.parent / "models" / "ppo_agents"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_model(self, domain_name):
        """Save PPO agent models"""
        policy_path = self.model_dir / f"{domain_name}_policy.pth"
        value_path = self.model_dir / f"{domain_name}_value.pth"
        optimizer_path = self.model_dir / f"{domain_name}_optimizers.pth"
        
        # Save model state dictionaries
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.value_net.state_dict(), value_path)
        
        # Save optimizer states
        torch.save({
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, optimizer_path)
        
        print(f"✅ PPO agent for {domain_name} saved:")
        print(f"   Policy: {policy_path}")
        print(f"   Value: {value_path}")
        print(f"   Optimizers: {optimizer_path}")
    
    def load_model(self, domain_name):
        """Load PPO agent models"""
        policy_path = self.model_dir / f"{domain_name}_policy.pth"
        value_path = self.model_dir / f"{domain_name}_value.pth"
        optimizer_path = self.model_dir / f"{domain_name}_optimizers.pth"
        
        if all(os.path.exists(path) for path in [policy_path, value_path, optimizer_path]):
            # Load model weights
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))
            
            # Load optimizer states
            optimizer_data = torch.load(optimizer_path, map_location=self.device)
            self.policy_optimizer.load_state_dict(optimizer_data['policy_optimizer'])
            self.value_optimizer.load_state_dict(optimizer_data['value_optimizer'])
            
            print(f"✅ PPO agent for {domain_name} loaded successfully")
            return True
        else:
            print(f"⚠️ No saved PPO model found for {domain_name}")
            return False
    
    def _create_policy_net(self, input_dim, output_dim):
        """Create policy network without BatchNorm"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def _create_value_net(self, input_dim):
        """Create value network without BatchNorm"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set to eval mode for inference
        self.policy_net.eval()
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
        
        # Set back to training mode
        self.policy_net.train()
        
        return action.item(), torch.exp(action_log_prob).item()
    
    def store_transition(self, state, action, reward, action_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_probs.append(action_prob)
    
    def update(self):
        if len(self.states) == 0:
            return 0
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        
        # Ensure networks are in training mode
        self.policy_net.train()
        self.value_net.train()
        
        # Calculate advantages
        values = self.value_net(states).squeeze()
        advantages = rewards - values.detach()
        
        # Update value network
        value_loss = F.mse_loss(values, rewards)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        logits = self.policy_net(states)
        action_probs = F.softmax(logits, dim=-1)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_probs = []
        
        return advantages.abs().mean().item()


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
        
    def save_all_models(self):
        """Save all trained models"""
        print("💾 Saving all models...")
        
        # Save domain classifier
        self.domain_classifier.save_model()
        
        # Save task classifiers and PPO agents
        self.task_classifier.save_models()
        
        print("✅ All models saved successfully!")
    
    def load_all_models(self):
        """Load all pre-trained models"""
        print("📂 Loading all models...")
        
        # Models are loaded automatically in __init__ methods
        # This method can be used for explicit reloading if needed
        
        print("✅ All models loaded successfully!")
    
    def route_prompt(self, prompt, use_ppo=False):
        # Step 1: FastText Language Detection
        language = self.language_detector.detect_language(prompt)
        
        # Step 2: ML-based Domain Classification
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        
        # Step 3: PPO-enhanced Task Classification
        task = self.task_classifier.classify_task(prompt, domain, use_ppo=use_ppo)
        
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
        """Train all PPO agents and save them"""
        for domain in self.domain_tasks.keys():
            print(f"Training PPO agent for {domain} domain...")
            self.task_classifier.train_ppo_agent(domain, training_data)
        
        # Save models after training
        print("💾 Saving trained PPO models...")
        self.task_classifier.save_models()
        print("✅ All PPO agents trained and saved!")
    
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
    """Create sample training data aligned with new tasks"""
    return [
        # Finance - Sentiment Analysis
        {'prompt': 'Analyze sentiment from the latest quarterly earnings report', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'What is the sentiment regarding Tesla new product launch', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Determine emotional tone of this market analysis report', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Is this financial news article positive or negative', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Analyze sentiment of this earnings report', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'What is the mood of this financial review', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Positive or negative opinion about stock performance', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Emotional tone analysis of market news', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification  
        {'prompt': 'Classify this business news headline into its correct sector', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Categorize this financial article by industry type', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'What category does this economic report belong to', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Classify this market update by financial sector', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'What type of news is this financial article', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Categorize this headline by business sector', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Classify this media report about markets', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Determine news category for this article', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': 'What is the capital of France', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Can you explain this concept', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Help me understand this topic', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'What is the difference between RAM and storage', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Why does this error occur', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'How does this process work', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': 'Summarize this research paper', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Give me key points summary', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Brief overview of this document', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Condensed version of this text', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Main points of this article', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Summary of important information', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification  
        {'prompt': 'Classify these documents by type', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Categorize this item into groups', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Sort these items by category', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Organize these documents by type', 'domain': 'general', 'task': 'classification'},
    ]


# 8. Usage Example and Testing
if __name__ == "__main__":
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
        
    system.save_all_models()
    
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    print("=" * 70)
    
    # Process prompts with PPO
    for i, prompt in enumerate(test_prompts, 1):
        result = system.route_prompt(prompt, use_ppo=train_rl)
        print(f"Test {i}: {prompt}")
        print(f"Route: {result['routing_path']}")
        # print(f"Result: {result['result']}")
        # print(f"Confidence: {result['expert_confidence']:.3f}")
        print()
    
    # Batch processing example
    # print("Batch processing all test prompts...")
    # batch_results = system.batch_process(test_prompts, use_ppo=train_rl)
    
    # print(f"Successfully processed {len(batch_results)} prompts")
    # print("\nSystem ready for production use!")
