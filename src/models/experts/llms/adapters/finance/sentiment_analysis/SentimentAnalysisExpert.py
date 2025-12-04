from __future__ import annotations

import re
from typing import Callable, Dict


def _default_cleaner(raw: str) -> str:
    match = re.search(r"\b([1-5])\b", raw)
    return match.group(1) if match else ""


def _llama_cleaner(raw: str) -> str:
    cleaned = raw.replace("⭐", "").replace("stars", "").strip()
    return _default_cleaner(cleaned)


ADAPTER_CLEANERS: Dict[str, Callable[[str], str]] = {
    "llama-2-7b-hf": _llama_cleaner,
}


class SentimentAnalysisExpert:
    """Adapter-aware cleaner for sentiment analysis outputs."""

    def __init__(self, adapter_name: str | None = None, **_: object) -> None:
        self.adapter_name = adapter_name

    def clean_output(self, raw: str) -> str:
        cleaner = ADAPTER_CLEANERS.get(self.adapter_name, _default_cleaner)
        return cleaner(raw or "")
