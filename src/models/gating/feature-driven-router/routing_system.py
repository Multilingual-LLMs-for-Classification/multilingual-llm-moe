# routing_system.py

import fasttext
import pickle
import re
from collections import Counter
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class BaseGating:
    """Abstract gating interface for domain decision."""
    def decide(self, text: str, lang: str, models: dict, keywords: dict) -> (str, float):
        raise NotImplementedError("Gating strategy must implement decide()")

class HeuristicGating(BaseGating):
    """Current TF-IDF + keyword heuristic gating."""
    def __init__(self, alpha: float, tfidf_models: dict, keyword_dicts: dict):
        self.alpha = alpha
        self.tfidf_models = tfidf_models
        self.keyword_dicts = keyword_dicts

    def decide(self, text: str, lang: str, models: dict, keywords: dict) -> (str, float):
        # TF-IDF probability
        if lang in self.tfidf_models:
            vectorizer, classifier = self.tfidf_models[lang]
            X = vectorizer.transform([text])
            proba = classifier.predict_proba(X)[0]
            tfidf_scores = dict(zip(classifier.classes_, proba))
        else:
            tfidf_scores = {'edu': 0.5, 'fin': 0.5}

        # Keyword scores
        kw = self.keyword_dicts.get(lang, {})
        tokens = text.split()
        counts = Counter(tokens)
        edu_count = sum(counts[w] for w in kw.get('edu', []))
        fin_count = sum(counts[w] for w in kw.get('fin', []))
        max_count = max(edu_count, fin_count, 1)
        kw_scores = {'edu': edu_count / max_count, 'fin': fin_count / max_count}

        # Combined score
        final_scores = {
            d: self.alpha * tfidf_scores.get(d, 0.0) + (1 - self.alpha) * kw_scores.get(d, 0.0)
            for d in ('edu', 'fin')
        }
        domain = max(final_scores, key=final_scores.get)
        return domain, final_scores[domain]

class ReinforcementGating(BaseGating):
    """Placeholder for future RL-based gating module."""
    def __init__(self, policy_model_path: str):
        # Load or initialize RL policy network
        self.policy = self._load_policy(policy_model_path)

    def _load_policy(self, path: str):
        # TODO: implement policy loading
        return None

    def decide(self, text: str, lang: str, models: dict, keywords: dict) -> (str, float):
        """
        RL-based decision: use state representation from text/lang
        to query policy network and output (domain, confidence).
        """
        # TODO: extract features and query RL policy
        raise NotImplementedError("RL gating not yet implemented")


class MultilingualRouter:
    def __init__(self,
                 lang_model_path: str = 'lang_detect.bin',
                 tfidf_model_paths: dict = None,
                 keyword_dict_paths: dict = None,
                 gating_strategy: BaseGating = None,
                 confidence_threshold: float = 0.6):
        # Load FastText language detection
        self.fasttext_model = fasttext.load_model(lang_model_path)
        
        # Load TF-IDF vectorizers + classifiers
        self.tfidf_models = {}
        for lang, paths in (tfidf_model_paths or {}).items():
            vec_path, clf_path = paths
            with open(vec_path, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(clf_path, 'rb') as f:
                classifier = pickle.load(f)
            self.tfidf_models[lang] = (vectorizer, classifier)
        
        # Load keyword dictionaries
        self.keyword_dicts = {}
        for lang, dicts in (keyword_dict_paths or {}).items():
            edu = self._load_keywords(dicts.get('edu', []))
            fin = self._load_keywords(dicts.get('fin', []))
            self.keyword_dicts[lang] = {'edu': edu, 'fin': fin}

        # Gating strategy (heuristic by default)
        if gating_strategy is None:
            self.gating = HeuristicGating(alpha=0.7,
                                          tfidf_models=self.tfidf_models,
                                          keyword_dicts=self.keyword_dicts)
        else:
            self.gating = gating_strategy

        self.confidence_threshold = confidence_threshold

    def _load_keywords(self, filepath: str) -> set:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            return set()

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()

    def detect_language(self, text: str) -> str:
        preds = self.fasttext_model.predict(text, k=1)
        return preds[0][0].replace('__label__', '')

    def fallback_classification(self, text: str) -> str:
        if re.search(r'\b(portfolio|investment|risk|stock|bond|loan)\b', text):
            return 'fin'
        if re.search(r'\b(curriculum|lecture|assessment|syllabus|pedagogy)\b', text):
            return 'edu'
        nums = re.findall(r'\d+', text)
        if len(''.join(nums)) / max(len(text),1) > 0.1:
            return 'fin'
        return 'edu'

    @lru_cache(maxsize=1024)
    def route(self, text: str) -> dict:
        processed = self.preprocess(text)
        lang = self.detect_language(processed)
        
        # Use gating strategy to decide domain
        domain, score = self.gating.decide(processed, lang,
                                           models=self.tfidf_models,
                                           keywords=self.keyword_dicts)

        # Fallback if below threshold
        if score < self.confidence_threshold:
            domain = self.fallback_classification(processed)
        
        expert = 'education_expert' if domain == 'edu' else 'finance_expert'
        return {'language': lang,
                'domain': domain,
                'score': score,
                'expert': expert}


# Example usage with default heuristic gating:
if __name__ == '__main__':
    tfidf_paths = {
        'en': ('models/en_tfidf_vec.pkl', 'models/en_nb_clf.pkl'),
        'es': ('models/es_tfidf_vec.pkl', 'models/es_nb_clf.pkl'),
    }
    kw_paths = {
        'en': {'edu': 'keywords/en_edu.txt', 'fin': 'keywords/en_fin.txt'},
        'es': {'edu': 'keywords/es_edu.txt', 'fin': 'keywords/es_fin.txt'},
    }

    router = MultilingualRouter(
        lang_model_path='lang_detect.bin',
        tfidf_model_paths=tfidf_paths,
        keyword_dict_paths=kw_paths
    )

    sample = "¿Cuál es la mejor estrategia de inversión para un portafolio diversificado?"
    print(router.route(sample))

    # To use future RL gating:
    # from some_rl_module import ReinforcementGating
    # rl_gating = ReinforcementGating(policy_model_path='policy.pt')
    # router_rl = MultilingualRouter(gating_strategy=rl_gating, ...)
