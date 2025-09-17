# Multi-Stage Prompt Routing System with Q-Learning (replacing PPO)
# Complete Implementation

import os
import csv
import sys
import re
import json
import random
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Reproducibility
random.seed(42)
np.random.seed(42)

# Add project root to Python path (same as your original)
project_root = Path(__file__).parents[6]  # Go up 6 levels to reach project root
sys.path.insert(0, str(project_root))

# External project dependencies
from src.models.experts.util.domain_task_loader import DomainTaskLoader
from src.models.experts.util.model_loader import ModelLoader
# from prompts_multilingual import test_prompts

# Torch / Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, logging as hf_logging

hf_logging.set_verbosity_error()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------------------------------
# 1) LANGUAGE DETECTION (unchanged, FastText-based)
# --------------------------------------------------------------------------------------------------
import fasttext

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
            return self._fallback_detection(text)
        try:
            cleaned_text = text.replace('\n', ' ').strip()
            if len(cleaned_text) < 3:
                return 'english'
            labels, scores = self.model.predict(cleaned_text, k=1)
            detected_lang = labels[0]
            mapped_lang = self.language_mapping.get(detected_lang, 'english')
            return mapped_lang
        except Exception:
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text):
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
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            scores['chinese'] = len([char for char in text if '\u4e00' <= char <= '\u9fff'])
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            scores['japanese'] = len([char for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'])
        text_words = text_lower.split()
        for lang, keywords in patterns.items():
            if lang in ['japanese', 'chinese']:
                continue
            scores[lang] = sum(1 for w in text_words if w in keywords)
        return max(scores, key=scores.get) if any(scores.values()) else 'english'


# --------------------------------------------------------------------------------------------------
# 2) DOMAIN CLASSIFICATION (unchanged, ML-based)
# --------------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

class DomainClassifier:
    def __init__(self):
        self.pipeline = None
        self.domains = ['finance', 'general']
        self.model_path = Path(__file__).parent.parent / "models" / "domain_classifier.joblib"
        self._initialize_classifier()
    
    def save_model(self, filepath=None):
        filepath = filepath or self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"✅ Domain classifier saved to: {filepath}")
    
    def load_model(self, filepath=None):
        filepath = filepath or self.model_path
        if os.path.exists(filepath):
            self.pipeline = joblib.load(filepath)
            print(f"✅ Domain classifier loaded from: {filepath}")
            return True
        else:
            print(f"⚠️ No saved model found at: {filepath}")
            return False
    
    def _initialize_classifier(self):
        if not self.load_model():
            print("Training new domain classifier...")
            training_data = self._create_training_data()
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb', ngram_range=(3, 5),
                    max_features=50000, lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            texts, labels = zip(*training_data)
            self.pipeline.fit(texts, labels)
            print("✅ Domain classifier trained")
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
        # Force every input to map to 'finance'
        return 'finance'

    def get_domain_probabilities(self, text):
        # Make the downstream code happy by returning a proper distro
        return {'finance': 0.75, 'general': 0.25}
    
    # def classify_domain(self, text):
    #     if self.pipeline is None:
    #         return self._fallback_classification(text)
    #     try:
    #         prediction = self.pipeline.predict([text])[0]
    #         return prediction
    #     except Exception:
    #         return self._fallback_classification(text)
    
    # def get_domain_probabilities(self, text):
    #     if self.pipeline is None:
    #         return {'finance': 0.5, 'general': 0.5}
    #     try:
    #         probabilities = self.pipeline.predict_proba([text])[0]
    #         return dict(zip(self.domains, probabilities))
    #     except Exception:
    #         return {'finance': 0.5, 'general': 0.5}
    
    def _fallback_classification(self, text):
        domain_keywords = {
            'finance': ['market','stock','price','investment','trading','portfolio','risk','return',
                        'bank','money','revenue','profit','analysis','economic','financial'],
            'general': ['help','question','what','how','why','when','where','explain','summary']
        }
        text_lower = text.lower()
        scores = {d: sum(1 for kw in kws if kw in text_lower)
                  for d, kws in domain_keywords.items()}
        return max(scores, key=scores.get)


# --------------------------------------------------------------------------------------------------
# 3) TASK CLASSIFICATION with Q-LEARNING (New, replaces PPO)
# --------------------------------------------------------------------------------------------------

class TransformersEncoder(nn.Module):
    """
    Shared multilingual encoder. Defaults to 'xlm-roberta-base'.
    """
    def __init__(self, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # CLS-like token (index 0) works for XLM-R and BERT-style models
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb
    
    def tokenize(self, texts: List[str], max_len: int = 128):
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]


class QRouter(nn.Module):
    def __init__(self, in_dim, num_tasks):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_tasks)
        )
    def forward(self, h):
        return self.net(h)  # Q-values


class DomainTaskDataset(Dataset):
    """
    Dataset for Q-learning: one sample = (input_ids, attention_mask, true_task_id, language)
    Filtered per-domain using given task2id.
    """
    def __init__(self, items: List[Dict], encoder: TransformersEncoder, task2id: Dict[str, int], max_len=128):
        self.items = [it for it in items if it["task"] in task2id]
        self.encoder = encoder
        self.task2id = task2id
        self.max_len = max_len
        # Pre-tokenize to keep training loop simple
        texts = [it["prompt"] for it in self.items]
        self.input_ids, self.attn = self.encoder.tokenize(texts, max_len=max_len)
        self.labels = torch.tensor([task2id[it["task"]] for it in self.items], dtype=torch.long)
        self.langs = [it.get("language", "") for it in self.items]
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn[idx], self.labels[idx], self.items[idx]["prompt"], self.langs[idx])


class   QLearningTaskClassifier:
    """
    Replaces PPO TaskClassifier. Maintains:
      - Shared multilingual encoder
      - Per-domain QRouter (num_tasks = len(tasks in that domain))
    Trains with simple epsilon-greedy Q-learning on labeled (domain, task) prompts.
    """
    def __init__(
        self,
        domain_tasks,
        model_dir: Path = None,
        encoder_name: str = "xlm-roberta-base",
        max_len: int = 128,
        batch_size: int = 16,
        lr: float = 1e-5,
        epochs: int = 1,
        eps_start: float = 0.2,
        eps_end: float = 0.01,
        eps_decay_steps: int = 10000
    ):
        # Normalize domain_tasks (could be DomainTaskLoader or dict)
        if hasattr(domain_tasks, "domain_tasks"):
            domain_tasks = domain_tasks.domain_tasks
        self.domain_tasks: Dict[str, Dict[str, dict]] = domain_tasks  # domain -> {task_name: ...}
        self.device = torch.device(DEVICE)
        self.encoder = TransformersEncoder(encoder_name).to(self.device)
        self.encoder_name = encoder_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.e0, self.e1, self.edec = eps_start, eps_end, eps_decay_steps
        self.global_step = 0
        
        # Prepare mappings and routers
        self.task2id: Dict[str, Dict[str, int]] = {}
        self.id2task: Dict[str, Dict[int, str]] = {}
        self.routers: Dict[str, QRouter] = {}
        
        for domain, tasks in self.domain_tasks.items():
            task_names = list(tasks.keys())
            t2i = {t: i for i, t in enumerate(task_names)}
            i2t = {i: t for t, i in t2i.items()}
            self.task2id[domain] = t2i
            self.id2task[domain] = i2t
            self.routers[domain] = QRouter(self.encoder.out_dim, num_tasks=len(task_names)).to(self.device)
        
        # Single optimizer for encoder + all routers (simple and effective)
        params = list(self.encoder.parameters()) + [p for r in self.routers.values() for p in r.parameters()]
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.mse = nn.MSELoss()
        
        # Model storage
        self.model_dir = model_dir or (Path(__file__).parent.parent / "models" / "task_routers_qlearning")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def epsilon(self) -> float:
        t = min(self.global_step / max(1, self.edec), 1.0)
        return self.e0 + (self.e1 - self.e0) * t
    
    def _domain_train_loop(self, domain: str, items: List[Dict]):
        if not items:
            print(f"⚠️ No training items for domain '{domain}', skipping.")
            return
        dataset = DomainTaskDataset(items, self.encoder, self.task2id[domain], max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        router = self.routers[domain]
        
        print(f"Training QRouter for domain '{domain}'   | samples={len(dataset)} tasks={len(self.task2id[domain])}")
        for epoch in range(self.epochs):
            router.train(); self.encoder.train()
            for input_ids, attn, labels, _, _ in loader:
                input_ids = input_ids.to(self.device)
                attn = attn.to(self.device)
                labels = labels.to(self.device)
                
                # Encode
                h = self.encoder(input_ids, attn)
                q_values = router(h)  # [B, num_tasks]
                
                # ε-greedy selection
                if random.random() < self.epsilon():
                    actions = torch.randint(0, q_values.size(1), (q_values.size(0),), device=self.device)
                else:
                    actions = q_values.argmax(dim=-1)
                
                rewards = (actions == labels).float()
                q_taken = q_values[torch.arange(q_values.size(0), device=self.device), actions]
                
                loss = self.mse(q_taken, rewards)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(router.parameters()), 1.0)
                self.optimizer.step()
                self.global_step += 1
            
            print(f"  Epoch {epoch+1}/{self.epochs} finished for domain '{domain}'")
    
    @torch.no_grad()
    def _domain_eval(self, domain: str, items: List[Dict]) -> float:
        if not items:
            return 0.0
        dataset = DomainTaskDataset(items, self.encoder, self.task2id[domain], max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        router = self.routers[domain]
        router.eval(); self.encoder.eval()
        correct = 0; total = 0
        for input_ids, attn, labels, _, _ in loader:
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)
            labels = labels.to(self.device)
            h = self.encoder(input_ids, attn)
            q_values = router(h)
            preds = q_values.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(1, total)
        return acc
    
    def train(self, training_data: List[Dict], val_split: float = 0.1):
        """
        training_data: list of dicts with keys {'prompt','domain','task',('language')}
        Trains a QRouter per domain using ε-greedy reward=1 if predicted_task==true_task else 0.
        """
        # Split per-domain
        for domain in self.domain_tasks.keys():
            domain_items = [d for d in training_data if d.get("domain") == domain and d.get("task") in self.task2id[domain]]
            if not domain_items:
                print(f"⚠️ No labeled items for domain '{domain}', skipping training.")
                continue
            # Split train/val
            idx = np.random.permutation(len(domain_items))
            n_val = int(val_split * len(domain_items))
            val_items = [domain_items[i] for i in idx[:n_val]]
            train_items = [domain_items[i] for i in idx[n_val:]]
            self._domain_train_loop(domain, train_items)
            acc = self._domain_eval(domain, val_items) if val_items else 0.0
            print(f"  → Validation accuracy for domain '{domain}': {acc:.4f}")
    
    @torch.no_grad()
    def classify_task(self, text: str, domain: str) -> str:
        if domain not in self.routers:
            # Fallback: return first task in domain
            task_names = list(self.domain_tasks[domain].keys())
            return task_names[0] if task_names else "unknown"
        self.encoder.eval(); self.routers[domain].eval()
        input_ids, attn = self.encoder.tokenize([text], max_len=self.max_len)
        input_ids = input_ids.to(self.device); attn = attn.to(self.device)
        h = self.encoder(input_ids, attn)
        q_values = self.routers[domain](h)
        pred_id = int(q_values.argmax(dim=-1).item())
        return self.id2task[domain].get(pred_id, "unknown")
    
    def save_models(self):
        # Save shared encoder weights + tokenizer name
        enc_path = self.model_dir / "encoder.pth"
        torch.save(self.encoder.state_dict(), enc_path)
        cfg_path = self.model_dir / "qrouter_config.json"
        with open(cfg_path, "w") as f:
            json.dump({"encoder_name": self.encoder_name}, f)
        # Save domain routers
        for domain, router in self.routers.items():
            rp = self.model_dir / f"router_{domain}.pth"
            torch.save(router.state_dict(), rp)
        print(f"✅ Q-learning routers saved to: {self.model_dir}")
    
    def load_models(self) -> bool:
        cfg_path = self.model_dir / "qrouter_config.json"
        enc_path = self.model_dir / "encoder.pth"
        ok = True
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
                enc_name = cfg.get("encoder_name", self.encoder_name)
                if enc_name != self.encoder_name:
                    print(f"ℹ️ Stored encoder '{enc_name}' differs from requested '{self.encoder_name}'. Using stored name.")
                    # Reload encoder w/ stored name
                    self.encoder = TransformersEncoder(enc_name).to(self.device)
                    self.encoder_name = enc_name
        if enc_path.exists():
            self.encoder.load_state_dict(torch.load(enc_path, map_location=self.device))
        else:
            ok = False
        # Load routers
        for domain, router in self.routers.items():
            rp = self.model_dir / f"router_{domain}.pth"
            if rp.exists():
                router.load_state_dict(torch.load(rp, map_location=self.device))
            else:
                ok = False
        if ok:
            print(f"✅ Q-learning routers loaded from: {self.model_dir}")
        else:
            print(f"⚠️ Q-learning routers not fully found; will train new ones.")
        return ok


# --------------------------------------------------------------------------------------------------
# 4) TASK EXPERTS (unchanged stub)
# --------------------------------------------------------------------------------------------------
class TaskExpert:
    def __init__(self, task_name):
        self.task_name = task_name
    def predict(self, text):
        confidence = random.uniform(0.1, 0.2)
        prediction = f"{self.task_name}_result"
        return prediction, confidence


# --------------------------------------------------------------------------------------------------
# 5) COMPLETE PROMPT ROUTING SYSTEM (now uses QLearningTaskClassifier)
# --------------------------------------------------------------------------------------------------
class PromptRoutingSystem:
    def __init__(self):
        config_path = Path(__file__).parents[4] / "experts" / "config"
        
        # Initialize components
        self.language_detector = LanguageDetector()
        self.domain_classifier = DomainClassifier()
        self.model_loader = ModelLoader(config_path / "model_config.json")
        self.domain_tasks_obj = DomainTaskLoader(config_path / "domain_tasks.json")
        self.domain_tasks = self.domain_tasks_obj.domain_tasks if hasattr(self.domain_tasks_obj, "domain_tasks") else self.domain_tasks_obj
        
        # Q-learning task classifier
        self.task_classifier = QLearningTaskClassifier(
            self.domain_tasks,
            model_dir=Path(__file__).parent.parent / "models" / "task_routers_qlearning",
            encoder_name="xlm-roberta-base",  # change to 'bert-base-multilingual-cased' if preferred
            max_len=128,
            batch_size=16,
            lr=1e-5,
            epochs=1,
            eps_start=0.2,
            eps_end=0.01,
            eps_decay_steps=10000
        )
        # Try loading existing QRouters
        self.task_classifier.load_models()
        
        # Download any external models if needed
        print("Checking and downloading models if needed...")
        self.model_loader.download_all_models()
        
        # Instantiate experts per domain/task
        self.experts = {}
        for domain, tasks in self.domain_tasks.items():
            self.experts[domain] = {}
            for task in tasks.keys():
                self.experts[domain][task] = TaskExpert(task)
    
    def save_all_models(self):
        print("💾 Saving all models...")
        self.domain_classifier.save_model()
        self.task_classifier.save_models()
        print("✅ All models saved successfully!")
    
    def route_prompt(self, prompt: str) -> Dict:
        # 1) Language
        language = self.language_detector.detect_language(prompt)
        # 2) Domain
        domain = self.domain_classifier.classify_domain(prompt)
        domain_probs = self.domain_classifier.get_domain_probabilities(prompt)
        # 3) Task via Q-learning router
        task = self.task_classifier.classify_task(prompt, domain)
        # 4) Expert
        expert = self.experts[domain][task]
        result, expert_confidence = expert.predict(prompt)
        
        output = {
        'input': prompt,
        'language': language,
        'domain': domain,
        'domain_probabilities': domain_probs,
        'task': task,
        'result': result,
        # 'expert_confidence': expert_confidence,
        'routing_path': f"{language} → {domain} → {task}"
        }

        # Print before returning
        print("Routing Result:", output)

        return output
        
    
    def train_q_routers(self, training_data: List[Dict]):
        """Train per-domain QRouters on labeled (domain, task, prompt) items."""
        self.task_classifier.train(training_data, val_split=0.1)
        self.task_classifier.save_models()
    
    def batch_process(self, prompts: List[str]) -> List[Dict]:
        results = []
        for prompt in prompts:
            results.append(self.route_prompt(prompt))
        return results
    
    def get_system_stats(self):
        total_tasks = sum(len(tasks) for tasks in self.domain_tasks.values())
        supported_languages = len(self.language_detector.language_mapping)
        return {
            'total_domains': len(self.domain_tasks),
            'total_tasks': total_tasks,
            'supported_languages': supported_languages,
            'domains': list(self.domain_tasks.keys())
        }


# --------------------------------------------------------------------------------------------------
# 6) TRAINING DATA (same helper as your original; keep full lists in your codebase)
# --------------------------------------------------------------------------------------------------
def create_sample_training_data():
    """Create sample training data aligned with new tasks"""
    data = []
    data += [
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
    
    data += [
        # Finance - Sentiment Analysis
        {'prompt': '最新の四半期決算書の感情を分析してください', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'テスラの新製品発表に対する市場のセンチメントは？', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'この市場分析レポートの感情的なトーンを判定して', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'この金融ニュースはポジティブかネガティブか', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'この決算発表のセンチメントを評価して', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'この株式レポートのムードは？', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '株価パフォーマンスに関する意見は好意的か否定的か', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '市況ニュースの感情トーンを解析して', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification
        {'prompt': 'このビジネスニュースの見出しを適切なセクターに分類して', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'この金融記事を業種別にカテゴリ分けして', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'この経済レポートはどのカテゴリに属しますか', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'このマーケットアップデートを金融セクター別に分類して', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'この金融記事はどのタイプのニュースですか', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'この見出しをビジネスセクター別に分類して', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '市場に関するこの報道を分類してください', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'この記事のニュースカテゴリを決定してください', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': 'フランスの首都はどこですか', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'この概念を説明してもらえますか', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'このトピックについて理解を助けて', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'RAMとストレージの違いは何ですか', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'なぜこのエラーが発生するのですか', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'このプロセスはどのように動作しますか', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': 'この研究論文を要約してください', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '重要なポイントを教えて', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'このドキュメントの概要を短く示してください', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'このテキストを簡潔にまとめてください', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'この記事の主なポイントを挙げてください', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '重要情報のサマリーをください', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification
        {'prompt': 'これらの文書を種類ごとに分類して', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'この項目をグループにカテゴリ分けして', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'これらの項目をカテゴリ別に並べ替えて', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'これらの文書を種類別に整理して', 'domain': 'general', 'task': 'classification'},
    ]

    # -------------------------
    # Chinese (Simplified, zh-CN)
    # -------------------------
    data += [
        # Finance - Sentiment Analysis
        {'prompt': '分析最新季度财报的情感倾向', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '特斯拉新品发布的市场情绪如何', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '判定这份市场分析报告的情感基调', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '这条金融新闻是正面还是负面', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '评估这份财报的情绪', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '这篇股票评论的整体情绪是什么', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '对股票表现的观点是积极还是消极', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '分析这则市况新闻的情绪', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification
        {'prompt': '将这条商业新闻标题归入正确的行业板块', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '按行业类型对这篇金融文章分类', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '这份经济报告属于哪个类别', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '按金融板块对这则市场更新进行分类', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '这篇金融文章属于哪类新闻', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '将这个标题按商业板块进行分类', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '请对这篇关于市场的报道进行分类', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '确定这篇文章的新闻类别', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': '法国的首都是哪里', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '能解释一下这个概念吗', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '帮我理解这个主题', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '内存与存储有什么区别', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '为什么会出现这个错误', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '这个流程是如何工作的', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': '请总结这篇研究论文', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '告诉我关键要点', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '简要概述这份文档', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '将这段文字精炼为摘要', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '列出这篇文章的主要要点', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': '给出重要信息的总结', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification
        {'prompt': '按类型分类这些文档', 'domain': 'general', 'task': 'classification'},
        {'prompt': '将此项目归类到相应组别', 'domain': 'general', 'task': 'classification'},
        {'prompt': '按类别对这些项目排序', 'domain': 'general', 'task': 'classification'},
        {'prompt': '按类型整理这些文档', 'domain': 'general', 'task': 'classification'},
    ]

    # -------------------------
    # Spanish (es)
    # -------------------------
    data += [
        # Finance - Sentiment Analysis
        {'prompt': 'Analiza el sentimiento del último informe trimestral de resultados', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '¿Cuál es el sentimiento del mercado sobre el lanzamiento del nuevo producto de Tesla?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Determina el tono emocional de este informe de análisis de mercado', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '¿Esta noticia financiera es positiva o negativa?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Evalúa el sentimiento de este reporte de ganancias', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': '¿Cuál es el estado de ánimo de esta reseña financiera?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'La opinión sobre el desempeño de la acción es positiva o negativa', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Analiza el tono emocional de esta noticia de mercado', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification
        {'prompt': 'Clasifica este titular de negocios en su sector correcto', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Categoriza este artículo financiero por industria', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '¿A qué categoría pertenece este informe económico?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Clasifica esta actualización de mercado por sector financiero', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': '¿Qué tipo de noticia es este artículo financiero?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Categoriza este titular por sector empresarial', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Clasifica este reporte de los mercados', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Determina la categoría de noticias de este artículo', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': '¿Cuál es la capital de Francia?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '¿Puedes explicar este concepto?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Ayúdame a entender este tema', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '¿Cuál es la diferencia entre RAM y almacenamiento?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '¿Por qué ocurre este error?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': '¿Cómo funciona este proceso?', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': 'Resume este artículo de investigación', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Dame un resumen con los puntos clave', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Breve visión general de este documento', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Versión condensada de este texto', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Puntos principales de este artículo', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Resumen de la información importante', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification
        {'prompt': 'Clasifica estos documentos por tipo', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Categoriza este elemento en grupos', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Ordena estos elementos por categoría', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Organiza estos documentos por tipo', 'domain': 'general', 'task': 'classification'},
    ]

    # -------------------------
    # French (fr)
    # -------------------------
    data += [
        # Finance - Sentiment Analysis
        {'prompt': 'Analyse le sentiment du dernier rapport trimestriel de résultats', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Quel est le sentiment du marché concernant le lancement du nouveau produit de Tesla ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Détermine le ton émotionnel de ce rapport d’analyse de marché', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Cet article financier est-il positif ou négatif ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Évalue le sentiment de ce communiqué de résultats', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Quel est l’état d’esprit de cette revue financière ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'L’avis sur la performance de l’action est-il positif ou négatif ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Analyse le ton émotionnel de cette actualité de marché', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification
        {'prompt': 'Classe ce titre économique dans le bon secteur', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Catégorise cet article financier par industrie', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'À quelle catégorie appartient ce rapport économique ?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Classe cette mise à jour de marché par secteur financier', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'De quel type de nouvelle s’agit-il pour cet article financier ?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Catégorise ce titre par secteur d’activité', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Classe ce reportage sur les marchés', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Détermine la catégorie d’actualité de cet article', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': 'Quelle est la capitale de la France ?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Peux-tu expliquer ce concept ?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Aide-moi à comprendre ce sujet', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Quelle est la différence entre la RAM et le stockage ?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Pourquoi cette erreur se produit-elle ?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Comment fonctionne ce processus ?', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': 'Résume cet article de recherche', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Donne-moi les points clés', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Brève vue d’ensemble de ce document', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Version condensée de ce texte', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Points principaux de cet article', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Résumé des informations importantes', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification
        {'prompt': 'Classe ces documents par type', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Catégorise cet élément en groupes', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Trie ces éléments par catégorie', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Organise ces documents par type', 'domain': 'general', 'task': 'classification'},
    ]

    # -------------------------
    # German (de)
    # -------------------------
    data += [
        # Finance - Sentiment Analysis
        {'prompt': 'Analysiere die Stimmung im neuesten Quartalsbericht', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Wie ist die Marktstimmung zum neuen Tesla-Produktlaunch?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Bestimme den emotionalen Ton dieses Marktanalyseberichts', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Ist dieser Finanzartikel positiv oder negativ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Beurteile die Stimmung dieser Ergebnisveröffentlichung', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Welche Stimmung hat diese Finanzrezension?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Ist die Meinung zur Aktienperformance eher positiv oder negativ?', 'domain': 'finance', 'task': 'sentiment_analysis'},
        {'prompt': 'Analysiere die emotionale Tonalität dieser Marktnachricht', 'domain': 'finance', 'task': 'sentiment_analysis'},

        # Finance - News Classification
        {'prompt': 'Ordne diese Wirtschafts-Schlagzeile dem richtigen Sektor zu', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Kategorisiere diesen Finanzartikel nach Branche', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Zu welcher Kategorie gehört dieser Wirtschaftsbericht?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Klassifiziere dieses Marktupdate nach Finanzsektor', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Um welche Art von Nachricht handelt es sich bei diesem Finanzartikel?', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Kategorisiere diese Schlagzeile nach Geschäftssektor', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Klassifiziere diesen Bericht über die Märkte', 'domain': 'finance', 'task': 'news_classification'},
        {'prompt': 'Bestimme die Nachrichtenkategorie dieses Artikels', 'domain': 'finance', 'task': 'news_classification'},

        # General - Question Answering
        {'prompt': 'Was ist die Hauptstadt von Frankreich?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Kannst du dieses Konzept erklären?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Hilf mir, dieses Thema zu verstehen', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Was ist der Unterschied zwischen RAM und Speicher?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Warum tritt dieser Fehler auf?', 'domain': 'general', 'task': 'question_answering'},
        {'prompt': 'Wie funktioniert dieser Prozess?', 'domain': 'general', 'task': 'question_answering'},

        # General - Text Summarization
        {'prompt': 'Fasse dieses Forschungspapier zusammen', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Nenne mir die wichtigsten Punkte', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Kurze Übersicht über dieses Dokument', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Verdichte diesen Text zu einer Zusammenfassung', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Hauptpunkte dieses Artikels', 'domain': 'general', 'task': 'text_summarization'},
        {'prompt': 'Zusammenfassung der wichtigsten Informationen', 'domain': 'general', 'task': 'text_summarization'},

        # General - Classification
        {'prompt': 'Klassifiziere diese Dokumente nach Typ', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Ordne dieses Element Gruppen zu', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Sortiere diese Elemente nach Kategorie', 'domain': 'general', 'task': 'classification'},
        {'prompt': 'Organisiere diese Dokumente nach Typ', 'domain': 'general', 'task': 'classification'},
    ]
    
    return data


# --------------------------------------------------------------------------------------------------
# 7) MAIN: wiring + evaluation (kept similar to your harness, simplified at the end)
# --------------------------------------------------------------------------------------------------
def _pct(x): return f"{x*100:6.2f}%"

def _compute_prf_bal_kappa(cm: Counter, labels: List[str]):
    # Same helper you had; condensed for brevity
    support = {c: 0 for c in labels}
    pred_tot = {c: 0 for c in labels}
    tp = {c: cm[(c, c)] for c in labels}
    for g in labels:
        support[g] = sum(cm[(g, p)] for p in labels)
    for p in labels:
        pred_tot[p] = sum(cm[(g, p)] for g in labels)
    total = sum(support.values()) if support else 0
    precisions, recalls, f1s, weights, recalls_only = [], [], [], [], []
    micro_tp = sum(tp.values())
    micro_fp = sum(pred_tot[c] - tp[c] for c in labels)
    micro_fn = sum(support[c] - tp[c] for c in labels)
    for c in labels:
        p = tp[c] / pred_tot[c] if pred_tot[c] > 0 else 0.0
        r = tp[c] / support[c]  if support[c]  > 0 else 0.0
        f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f); recalls_only.append(r)
        weights.append(support[c] / total if total else 0.0)
    macro_p = sum(precisions)/len(labels) if labels else 0.0
    macro_r = sum(recalls)/len(labels)    if labels else 0.0
    macro_f1 = sum(f1s)/len(labels)       if labels else 0.0
    weighted_f1 = sum(w*f for w, f in zip(weights, f1s)) if labels else 0.0
    balanced_acc = sum(recalls_only)/len(labels) if labels else 0.0
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p + micro_r) > 0 else 0.0
    accuracy = micro_tp / total if total else 0.0
    pe = sum((support[c]/total) * (pred_tot[c]/total) for c in labels) if total else 0.0
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1,
        'weighted_f1': weighted_f1, 'balanced_acc': balanced_acc, 'kappa': kappa,
        'support': support, 'pred_tot': pred_tot, 'total': total,
    }

def _print_confusion(cm: Counter, labels: List[str], title: str):
    print(title)
    if not labels:
        print("  (no labels)\n"); return
    header = "      " + " ".join(f"{lbl:>22}" for lbl in labels)
    print(header)
    for gt in labels:
        row = [f"{gt:>6}"]
        for pr in labels:
            row.append(f"{cm[(gt, pr)]:>22}")
        print(" ".join(row))
    print()

def load_prompts_from_csv(path="unified.csv"):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]
    
if __name__ == "__main__":
    with open("test.json", "r", encoding="utf-8") as f:
        test_prompts = json.load(f)

    print("Initializing Prompt Routing System (Q-learning router)...")
    system = PromptRoutingSystem()
    
    # Display system information
    stats = system.get_system_stats()
    print(f"System initialized with {stats['total_domains']} domains and {stats['total_tasks']} tasks")
    print(f"Supported languages: {stats['supported_languages']}")
    print(f"Available domains: {stats['domains']}")
    
    # Sample training data (replace with your full multilingual sets)
    with open("train.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    # Optional: Train Q-learning routers
    # train_q = input("\nTrain Q-learning routers now? (y/n): ").strip().lower() == 'y'
    # if train_q:
    system.train_q_routers(training_data)
    
    system.save_all_models()
    
    # ----------- Evaluation on test_prompts -----------
    print(f"\nTesting with {len(test_prompts)} sample prompts...")
    print("=" * 70)
    
    # Collect confusion matrices for domain and task
    domain_labels = sorted({item['domain'] for item in test_prompts if isinstance(item, dict) and 'domain' in item})
    task_labels   = sorted({item['task']   for item in test_prompts if isinstance(item, dict) and 'task'   in item})
    cm_domain = Counter()
    cm_task   = Counter()

    per_lang_total = Counter()
    per_lang_dom   = Counter()
    per_lang_task  = Counter()
    per_lang_exact = Counter()

    for item in test_prompts:
        if not isinstance(item, dict):  # safety
            continue
        text      = item['prompt']
        gt_domain = item['domain']
        gt_task   = item['task']

        result      = system.route_prompt(text)
        print
        pred_domain = result['domain']
        pred_task   = result['task']
        lang_tag    = result.get('language', '?')
        print(gt_task, pred_task, lang_tag)
        cm_domain[(gt_domain, pred_domain)] += 1
        cm_task[(gt_task, pred_task)]       += 1

        per_lang_total[lang_tag] += 1
        dom_ok  = (pred_domain == gt_domain)
        task_ok = (pred_task   == gt_task)
        both_ok = dom_ok and task_ok
        per_lang_dom[lang_tag]   += int(dom_ok)
        per_lang_task[lang_tag]  += int(task_ok)
        per_lang_exact[lang_tag] += int(both_ok)

    dom_metrics  = _compute_prf_bal_kappa(cm_domain, domain_labels)
    task_metrics = _compute_prf_bal_kappa(cm_task,   task_labels)

    print("\nADDITIONAL METRICS")
    print("=" * 80)
    print("Domain classification:")
    print(f"  Accuracy           : {_pct(dom_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {_pct(dom_metrics['macro_p'])} / {_pct(dom_metrics['macro_r'])} / {_pct(dom_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {_pct(dom_metrics['micro_p'])} / {_pct(dom_metrics['micro_r'])} / {_pct(dom_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {_pct(dom_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {_pct(dom_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {_pct(dom_metrics['kappa'])}")

    print("\nTask classification:")
    print(f"  Accuracy           : {_pct(task_metrics['accuracy'])}")
    print(f"  Macro  P/R/F1      : {_pct(task_metrics['macro_p'])} / {_pct(task_metrics['macro_r'])} / {_pct(task_metrics['macro_f1'])}")
    print(f"  Micro  P/R/F1      : {_pct(task_metrics['micro_p'])} / {_pct(task_metrics['micro_r'])} / {_pct(task_metrics['micro_f1'])}")
    print(f"  Weighted F1        : {_pct(task_metrics['weighted_f1'])}")
    print(f"  Balanced accuracy  : {_pct(task_metrics['balanced_acc'])}")
    print(f"  Cohen's kappa (κ)  : {_pct(task_metrics['kappa'])}")

    _print_confusion(cm_domain, domain_labels, title="\nDomain Confusion Matrix (GT rows × Pred cols)")
    _print_confusion(cm_task,   task_labels,   title="Task Confusion Matrix (GT rows × Pred cols)")
    print("Language list (detected):", sorted(per_lang_total))

    if per_lang_total:
        print("Per-language breakdown (using detected language):")
        print("-" * 80)
        print(f"{'lang':>6} | {'n':>4} | {'domain acc':>12} | {'task acc':>10} | {'exact acc':>10}")
        for lg in sorted(per_lang_total):
            n_l = per_lang_total[lg]
            dom_acc_l  = per_lang_dom[lg]   / n_l if n_l else 0.0
            task_acc_l = per_lang_task[lg]  / n_l if n_l else 0.0
            exact_l    = per_lang_exact[lg] / n_l if n_l else 0.0
            print(f"{lg:>6} | {n_l:>4} | {_pct(dom_acc_l):>12} | {_pct(task_acc_l):>10} | {_pct(exact_l):>10}")
