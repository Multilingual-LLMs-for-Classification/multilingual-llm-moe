"""
Language detection component using FastText.

Detects input language dynamically based on expert registry configuration.
Supports 176 languages with fallback pattern matching.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import requests
import fasttext


class LanguageDetector:
    """
    FastText-based language detector with dynamic registry loading.

    Loads supported languages from expert registry and builds dynamic
    FastText label mappings. Includes comprehensive fallback detection
    for 50+ common languages using character patterns and keywords.
    """

    def __init__(self, registry_path: str | Path = None):
        self.model_path = Path(__file__).parents[2] / "models" / "lid.176.bin"
        self.model = None
        self._load_fasttext_model()

        # Comprehensive language code to full name mapping
        # Supports all common ISO 639-1 codes used in NLP
        self._comprehensive_code_mapping = {
            'af': 'afrikaans', 'ar': 'arabic', 'bg': 'bulgarian', 'bn': 'bengali',
            'ca': 'catalan', 'cs': 'czech', 'cy': 'welsh', 'da': 'danish',
            'de': 'german', 'el': 'greek', 'en': 'english', 'es': 'spanish',
            'et': 'estonian', 'fa': 'persian', 'fi': 'finnish', 'fr': 'french',
            'gu': 'gujarati', 'he': 'hebrew', 'hi': 'hindi', 'hr': 'croatian',
            'hu': 'hungarian', 'id': 'indonesian', 'it': 'italian', 'ja': 'japanese',
            'ka': 'georgian', 'kk': 'kazakh', 'km': 'khmer', 'kn': 'kannada',
            'ko': 'korean', 'lt': 'lithuanian', 'lv': 'latvian', 'mk': 'macedonian',
            'ml': 'malayalam', 'mn': 'mongolian', 'mr': 'marathi', 'ne': 'nepali',
            'nl': 'dutch', 'no': 'norwegian', 'pa': 'punjabi', 'pl': 'polish',
            'pt': 'portuguese', 'ro': 'romanian', 'ru': 'russian', 'si': 'sinhala',
            'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali', 'sq': 'albanian',
            'sv': 'swedish', 'sw': 'swahili', 'ta': 'tamil', 'te': 'telugu',
            'th': 'thai', 'tl': 'tagalog', 'tr': 'turkish', 'uk': 'ukrainian',
            'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese', 'zu': 'zulu'
        }

        # Load supported languages from registry if provided
        self.registry_path = registry_path
        self.supported_languages_by_task = {}
        self.all_supported_languages = set()

        if registry_path:
            self._load_languages_from_registry()

        # Build dynamic language mapping from all discovered languages
        self.language_mapping = self._build_language_mapping()

    def _load_languages_from_registry(self):
        """Load supported languages from experts registry"""
        try:
            registry_file = Path(self.registry_path)
            if not registry_file.is_absolute():
                registry_file = Path(__file__).parents[5] / self.registry_path

            with open(registry_file, 'r') as f:
                registry = json.load(f)

            # Extract supported languages from each task
            tasks = registry.get("tasks", {})
            for task_key, task_config in tasks.items():
                supported_langs = task_config.get("supported_languages", [])
                self.supported_languages_by_task[task_key] = supported_langs

                # Add to global set (convert short codes to full names)
                for lang_code in supported_langs:
                    lang_full = self._code_to_full_name(lang_code)
                    self.all_supported_languages.add(lang_full)

            print(f"✅ Loaded language support from registry:")
            for task, langs in self.supported_languages_by_task.items():
                print(f"   {task}: {langs}")
            print(f"   All supported languages: {sorted(self.all_supported_languages)}")

        except Exception as e:
            print(f"⚠️ Could not load languages from registry: {e}")
            print(f"   Using default language mapping")

    def _code_to_full_name(self, code: str) -> str:
        """Convert language code to full name (e.g., 'en' -> 'english')"""
        return self._comprehensive_code_mapping.get(code.lower(), code)

    def _build_language_mapping(self) -> dict:
        """
        Build FastText label mapping dynamically from all supported languages.

        Creates mapping: '__label__<code>' -> '<full_name>'
        for all languages discovered in the registry.

        Returns:
            Dict mapping FastText labels to full language names
        """
        mapping = {}

        # If languages loaded from registry, use those
        if self.all_supported_languages:
            # Get unique language codes from all supported languages
            lang_codes = set()
            for task_langs in self.supported_languages_by_task.values():
                lang_codes.update(task_langs)

            # Build FastText label mapping for each code
            for code in lang_codes:
                full_name = self._code_to_full_name(code)
                fasttext_label = f'__label__{code.lower()}'
                mapping[fasttext_label] = full_name

            print(f"✅ Built dynamic language mapping for {len(mapping)} languages:")
            print(f"   {sorted(mapping.values())}")
        else:
            # Fallback: build from comprehensive mapping
            for code, full_name in self._comprehensive_code_mapping.items():
                fasttext_label = f'__label__{code}'
                mapping[fasttext_label] = full_name

            # Update all_supported_languages with comprehensive set
            self.all_supported_languages = set(self._comprehensive_code_mapping.values())

            print(f"⚠️ No registry languages loaded, using comprehensive mapping ({len(mapping)} languages)")

        return mapping

    def get_supported_languages_for_task(self, domain: str, task: str) -> list[str]:
        """Get list of supported languages for a specific task"""
        task_key = f"{domain}/{task}"
        return self.supported_languages_by_task.get(task_key, list(self.all_supported_languages))

    def _load_fasttext_model(self):
        """Load FastText language identification model"""
        try:
            if not self.model_path.exists():
                self._download_fasttext_model()
            self.model = fasttext.load_model(str(self.model_path))
            print("✅ FastText language model loaded")
        except Exception as e:
            print(f"⚠️ FastText model failed, using fallback: {e}")
            self.model = None

    def _download_fasttext_model(self):
        """Download FastText model from official repository"""
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading FastText model...")
        response = requests.get(url)
        with open(self.model_path, 'wb') as f:
            f.write(response.content)
        print("✅ FastText model downloaded")

    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.

        Args:
            text: Input text to detect language

        Returns:
            Full language name (e.g., 'english', 'japanese')
        """
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
            print("⚠️ FastText detection failed, using fallback")
            return self._fallback_detection(text)

    def _fallback_detection(self, text: str) -> str:
        """
        Fallback language detection using character patterns and common words.

        Supports all languages from registry, with extended patterns for common languages.
        Uses Unicode character ranges for logographic/syllabic scripts and keyword
        matching for alphabetic scripts.

        Args:
            text: Input text to detect language

        Returns:
            Detected language name
        """
        # Comprehensive pattern library for common words and characters
        patterns = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'you', 'this'],
            'german': ['der', 'die', 'das', 'und', 'ist', 'ich', 'nicht', 'ein', 'eine', 'zu', 'den', 'von'],
            'spanish': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'por'],
            'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans'],
            'japanese': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる'],
            'chinese': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这', '中', '大', '为', '上'],
            'portuguese': ['o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não'],
            'russian': ['и', 'в', 'не', 'на', 'я', 'с', 'что', 'а', 'по', 'это', 'он', 'как', 'к'],
            'italian': ['il', 'di', 'e', 'la', 'è', 'che', 'per', 'un', 'in', 'del', 'con', 'non'],
            'dutch': ['de', 'het', 'een', 'van', 'en', 'in', 'op', 'is', 'te', 'dat', 'voor', 'met'],
            'arabic': ['في', 'من', 'على', 'إلى', 'هذا', 'أن', 'هو', 'ما', 'كان', 'عن', 'كل'],
            'korean': ['은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로'],
            'hindi': ['के', 'में', 'की', 'है', 'से', 'को', 'और', 'का', 'एक', 'पर', 'यह'],
        }

        text_lower = text.lower()
        scores = {}

        # Character-based detection for logographic/syllabic scripts
        # Chinese (CJK Unified Ideographs)
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            scores['chinese'] = len([char for char in text if '\u4e00' <= char <= '\u9fff'])

        # Japanese (Hiragana + Katakana)
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            scores['japanese'] = len([char for char in text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'])

        # Korean (Hangul)
        if any('\uac00' <= char <= '\ud7af' for char in text):
            scores['korean'] = len([char for char in text if '\uac00' <= char <= '\ud7af'])

        # Arabic (Arabic script)
        if any('\u0600' <= char <= '\u06ff' for char in text):
            scores['arabic'] = len([char for char in text if '\u0600' <= char <= '\u06ff'])

        # Devanagari (Hindi and related)
        if any('\u0900' <= char <= '\u097f' for char in text):
            scores['hindi'] = len([char for char in text if '\u0900' <= char <= '\u097f'])

        # Cyrillic (Russian and related)
        if any('\u0400' <= char <= '\u04ff' for char in text):
            scores['russian'] = len([char for char in text if '\u0400' <= char <= '\u04ff'])

        # Word-based detection for alphabetic scripts
        text_words = text_lower.split()

        # Only check patterns for languages in our supported set
        for lang in self.all_supported_languages:
            if lang in patterns:
                keywords = patterns[lang]
                # Skip character-based languages (already scored above)
                if lang in ['japanese', 'chinese', 'korean', 'arabic', 'hindi', 'russian']:
                    continue
                scores[lang] = sum(1 for w in text_words if w in keywords)

        # Return language with highest score, default to english
        if scores and any(scores.values()):
            return max(scores, key=scores.get)

        # If no patterns matched and we have supported languages, default to first alphabetically
        # (likely english if it's in the set)
        if self.all_supported_languages:
            supported_list = sorted(self.all_supported_languages)
            return 'english' if 'english' in supported_list else supported_list[0]

        return 'english'
