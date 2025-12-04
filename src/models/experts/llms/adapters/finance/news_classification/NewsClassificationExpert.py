from __future__ import annotations

from typing import Callable, Dict

CATEGORIES = [
    "Finance",
    "Tax & Accounting",
    "Government & Controls",
    "Technology",
    "Industry",
    "Business & Management",
]


def _default_cleaner(raw: str) -> str:
    """Fallback cleaner used when no adapter-specific override is present."""
    raw = raw.lower().strip()

    for category in CATEGORIES:
        if category.lower() == raw:
            return category

    for category in CATEGORIES:
        lowered = category.lower()
        if raw in lowered or lowered in raw:
            return category

    return "unknown"


def _gemma_cleaner(raw: str) -> str:
    """Gemma-specific cleanup handling enumerations like 'Category: Finance'."""
    raw = raw.lower().strip()
    if raw.startswith("category:"):
        raw = raw.split(":", 1)[1].strip()
    return _default_cleaner(raw)


def _llama_cleaner(raw: str) -> str:
    """Llama-specific cleanup handling bullet/numbered responses."""
    cleaned = raw.strip().lower()
    for token in ("-", "--", "*", "1.", "2."):
        if cleaned.startswith(token):
            cleaned = cleaned[len(token):].strip()
    return _default_cleaner(cleaned)


ADAPTER_CLEANERS: Dict[str, Callable[[str], str]] = {
    "gemma-7b": _gemma_cleaner,
    "llama-2-7b-hf": _llama_cleaner,
}


class NewsClassificationExpert:
    """Adapter-aware cleaner for news classification outputs."""

    def __init__(self, adapter_name: str | None = None, **_: object) -> None:
        self.adapter_name = adapter_name

    def clean_output(self, raw: str) -> str:
        cleaner = ADAPTER_CLEANERS.get(self.adapter_name, _default_cleaner)
        return cleaner(raw or "")
