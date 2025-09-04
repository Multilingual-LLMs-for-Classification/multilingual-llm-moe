#!/usr/bin/env python3
r"""
prompt_generation.py

Generate multilingual classification prompts from CSV datasets using templates.

Supported input schemas:
1) Amazon-style:
   Unnamed: 0, review_id, product_id, reviewer_id, stars, review_body,
   review_title, language, product_category, sentiment

2) Generic / News-style (your case):
   text, label, lang, id
   (optionally: headline, headline_title, headline_text, title)

Behavior:
- Detect language from 'language'/'lang'/locale-like columns; fallback to reviewer_id prefix.
- Build one-sentence "review_text" from (title, body[, category]) with graceful fallbacks.
  IMPORTANT: We DO NOT append the gold 'label' field to avoid leakage.
- Templates may use "{review_text}" or "{headline_text}".
- Write prompts to a JSON array file (default: generated_prompts_news.json).
"""

from __future__ import annotations
import argparse
import csv
import glob
import io
import json
import os
import random
import sys
from typing import Dict, List, Any, Optional, Iterable, Tuple

# ---------- Config (safe defaults; override via CLI) ----------
TEMPLATES_FILE_DEFAULT = os.path.join("templates", "star_rating.json")
OUTPUT_FILE_DEFAULT = "generated_prompts_news.json"

SUPPORTED_LANG_KEYS = {
    # canonical -> "english" | "spanish" | "french" | "german" | "japanese" | "chinese" | "turkish"
    "en": "english", "eng": "english", "english": "english",
    "es": "spanish", "spa": "spanish", "spanish": "spanish",
    "fr": "french",  "fra": "french", "fre": "french", "french": "french",
    "de": "german",  "ger": "german", "deu": "german", "german": "german",
    "ja": "japanese","jpn": "japanese","jp": "japanese","japanese": "japanese",
    "zh": "chinese", "chi": "chinese","zho": "chinese","cn": "chinese","chinese": "chinese", "turkish": "turkish", "danish": "danish", "polish": "polish"
}

# reviewer_id prefixes to language (Amazon-style rows)
REVIEWER_PREFIX_TO_LANG = {
    "en_": "english",
    "es_": "spanish",
    "fr_": "french",
    "de_": "german",
    "ja_": "japanese",
    "zh_": "chinese",
}

# For metadata fallback across schemas (NO gold 'label' here to prevent leakage)
FALLBACK_CATEGORY_KEYS = ["product_category", "topic"]

# ---------- Utilities ----------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def first_nonempty(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""

def normalize_lang(value: Optional[str]) -> Optional[str]:
    """
    Normalize language codes like 'English', 'en-US', 'zh-Hans', 'de_DE' to our buckets.
    """
    if not value:
        return None
    key = str(value).strip().lower().replace("_", "-")
    primary = key.split("-", 1)[0]

    variant_map = {
        # English
        "en-us": "english", "en-gb": "english", "en": "english",
        # Spanish
        "es-es": "spanish", "es-mx": "spanish", "es": "spanish",
        # French
        "fr-fr": "french",  "fr-ca": "french",  "fr": "french",
        # German
        "de-de": "german",  "de-at": "german",  "de": "german",
        # Japanese
        "ja-jp": "japanese","ja": "japanese",
        # Chinese (group both Hans/Hant & regions)
        "zh-cn": "chinese", "zh-sg": "chinese", "zh-hans": "chinese",
        "zh-tw": "chinese", "zh-hk": "chinese", "zh-hant": "chinese",
        "zh": "chinese",
    }

    return (
        SUPPORTED_LANG_KEYS.get(key)
        or variant_map.get(key)
        or SUPPORTED_LANG_KEYS.get(primary)
    )

def infer_lang_from_reviewer(reviewer_id: Optional[str]) -> Optional[str]:
    if not reviewer_id:
        return None
    rid = reviewer_id.strip()
    for prefix, lang in REVIEWER_PREFIX_TO_LANG.items():
        if rid.startswith(prefix):
            return lang
    if "_" in rid:
        token = rid.split("_", 1)[0].lower()
        guess = SUPPORTED_LANG_KEYS.get(token)
        if guess:
            return guess
    return None

def detect_language(row: Dict[str, Any], default_lang: Optional[str] = None) -> Optional[str]:
    # Prefer explicit language columns (works for both schemas & news files)
    lang_val = first_nonempty(row, ["language", "lang", "language_code", "lang_code", "locale", "langid"])
    lang_norm = normalize_lang(lang_val) if lang_val else None
    if lang_norm:
        return lang_norm

    # Fallback for Amazon rows
    reviewer_id = (row.get("reviewer_id") or "").strip()
    inferred = infer_lang_from_reviewer(reviewer_id)
    if inferred:
        return inferred

    # Optional default
    if default_lang:
        dl = normalize_lang(default_lang)
        if dl:
            return dl
    return None

def combine_to_one_sentence(title: str, body: str, category: str) -> str:
    """
    "{title}: {body} (Category: {category})" with graceful fallbacks.
    If 'category' is empty (e.g., news rows where 'label' is gold), it will be omitted.
    """
    t = title.strip() if title else ""
    b = body.strip() if body else ""
    c = category.strip() if category else ""

    if t and b:
        core = f"{t}: {b}"
    elif t:
        core = t
    else:
        core = b

    if c:
        return f"{core} (Category: {c})" if core else f"(Category: {c})"
    return core

def load_templates_grouped_by_lang(path: str, encoding: str = "utf-8") -> Dict[str, List[Dict[str, Any]]]:
    """
    Load templates JSON file.
    Accepts either:
      - a flat list: [{"id": "...","lang":"english","template":"..."}, ...]
      - or an object containing such a list as its first list value.
    Returns: {lang: [template_obj, ...]}
    Templates are accepted if they contain "{review_text}" or "{headline_text}".
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    with io.open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if not lists:
            raise ValueError("Template JSON must contain a list of templates.")
        data = lists[0]

    if not isinstance(data, list):
        raise ValueError("Template JSON must be a list of template objects.")

    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            eprint(f"[warn] Template entry #{i} is not an object; skipping.")
            continue
        tmpl = item.get("template")
        lang = item.get("lang")
        if not tmpl or not isinstance(tmpl, str) or ("{review_text}" not in tmpl and "{headline_text}" not in tmpl):
            eprint(f"[warn] Template entry #{i} missing/invalid 'template' with '{{review_text}}' or '{{headline_text}}'; skipping.")
            continue
        if not lang or not isinstance(lang, str):
            eprint(f"[warn] Template entry #{i} missing 'lang'; skipping.")
            continue
        lang_key = normalize_lang(lang)
        if not lang_key:
            eprint(f"[warn] Template entry #{i} has unsupported lang '{lang}'; skipping.")
            continue
        if "id" not in item:
            item["id"] = f"{lang_key}-{i}"
        by_lang.setdefault(lang_key, []).append(item)

    if not by_lang:
        raise ValueError("No usable templates loaded (check 'lang' and placeholders).")

    return by_lang

def iter_csv_rows(csv_path: str, encoding_order: List[str] = ["utf-8", "utf-8-sig", "latin-1"]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    last_err = None
    for enc in encoding_order:
        try:
            with io.open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    yield i, row
            return
        except UnicodeDecodeError as ex:
            last_err = ex
            continue
        except Exception:
            raise
    if last_err:
        raise last_err

def resolve_csv_paths(data_arg: str) -> List[str]:
    if os.path.isdir(data_arg):
        pat = os.path.join(data_arg, "*.csv")
        return sorted(glob.glob(pat))
    if any(ch in data_arg for ch in ["*", "?", "["]):
        return sorted(glob.glob(data_arg))
    return [data_arg]

# ---------- Prompt generation ----------
def generate_prompt_records_for_file(
    csv_path: str,
    templates_by_lang: Dict[str, List[Dict[str, Any]]],
    default_lang: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    for row_idx, row in iter_csv_rows(csv_path):
        # 1) Detect language
        lang = detect_language(row, default_lang=default_lang)
        if not lang:
            eprint(f"[warn] {csv_path} row {row_idx}: no language detected; skipping.")
            continue

        tmpl_list = templates_by_lang.get(lang)
        if not tmpl_list:
            eprint(f"[warn] {csv_path} row {row_idx}: no templates for language '{lang}'; skipping.")
            continue

        # 2) Build review_text (Amazon + Generic + News)
        title = first_nonempty(row, ["review_title", "headline", "headline_title", "title"])
        body = first_nonempty(row, ["review_body", "text", "headline", "headline_text"])
        # DO NOT include the gold 'label' to avoid leakage:
        category = first_nonempty(row, FALLBACK_CATEGORY_KEYS)  # product_category or topic (not 'label')
        review_text = combine_to_one_sentence(title=title, body=body, category=category).strip()
        if not review_text:
            eprint(f"[warn] {csv_path} row {row_idx}: empty review_text; skipping.")
            continue

        # 3) Choose a random template within the language bucket
        tmpl = random.choice(tmpl_list)
        prompt_text = tmpl["template"]
        # Support both placeholders
        if "{review_text}" in prompt_text:
            prompt_text = prompt_text.replace("{review_text}", review_text)
        if "{headline_text}" in prompt_text:
            prompt_text = prompt_text.replace("{headline_text}", review_text)

        # 4) Emit record
        record = {
            "template_id": tmpl.get("id"),
            "template_lang": lang,
            "template_text": tmpl.get("template"),
            "prompt": prompt_text,
            "review_text": review_text,
            # Amazon-style metadata
            "review_id": row.get("review_id"),
            "product_id": row.get("product_id"),
            "reviewer_id": row.get("reviewer_id"),
            "stars": row.get("stars"),
            "product_category": row.get("product_category"),
            "sentiment": row.get("sentiment"),
            "language_column": row.get("language") or row.get("lang"),
            # Generic/news fields
            "generic_id": row.get("id"),
            "generic_label": row.get("label"),
            # Provenance
            "source_csv": os.path.abspath(csv_path),
            "row_index": row_idx,
        }
        yield record

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate prompts from multilingual templates and CSV datasets.")
    parser.add_argument("--templates", default=TEMPLATES_FILE_DEFAULT, help=f'Path to templates JSON (default: "{TEMPLATES_FILE_DEFAULT}")')
    parser.add_argument("--data", required=True, help="CSV file, directory containing CSVs, or a glob pattern (e.g., ./data/*.csv).")
    parser.add_argument("--output", default=OUTPUT_FILE_DEFAULT, help=f'Output JSON file (array). Default: "{OUTPUT_FILE_DEFAULT}"')
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible template selection.")
    parser.add_argument("--default-lang", default=None, help='Optional default language to use when not detected (e.g., "english").')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        templates_by_lang = load_templates_grouped_by_lang(args.templates)
        eprint("[info] Loaded templates per language: " + json.dumps({k: len(v) for k, v in templates_by_lang.items()}))
    except Exception as ex:
        eprint(f"[error] Failed to load templates: {ex}")
        sys.exit(1)

    csv_files = resolve_csv_paths(args.data)
    if not csv_files:
        eprint(f"[error] No CSV files found for: {args.data}")
        sys.exit(1)

    eprint("[info] CSVs to process:")
    for p in csv_files:
        eprint("  - " + os.path.abspath(p))

    outputs: List[Dict[str, Any]] = []
    for csv_path in csv_files:
        try:
            for rec in generate_prompt_records_for_file(csv_path, templates_by_lang, default_lang=args.default_lang):
                outputs.append(rec)
        except Exception as ex:
            eprint(f"[warn] Skipping file due to error: {csv_path} -> {ex}")

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        with io.open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        eprint(f"[info] Wrote {len(outputs)} prompts -> {out_path}")
    except Exception as ex:
        eprint(f"[error] Failed to write output: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()
