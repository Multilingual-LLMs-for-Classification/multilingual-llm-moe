#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import random
import sys
from typing import Dict, List, Any

def normalize_language(raw: str) -> str:
    """Map a variety of language strings to the template 'lang' keys."""
    if not raw:
        return ""
    s = str(raw).strip().lower()

    # Common aliases
    mapping = {
        # English
        "en": "english", "eng": "english", "english": "english",
        # Dutch
        "nl": "dutch", "nld": "dutch", "dutch": "dutch", "nederlands": "dutch",
        # French
        "fr": "french", "fra": "french", "fre": "french", "français": "french", "francais": "french", "french": "french",
        # German
        "de": "german", "deu": "german", "ger": "german", "deutsch": "german", "german": "german",
        # Italian
        "it": "italian", "ita": "italian", "italiano": "italian", "italian": "italian",
        # Spanish
        "es": "spanish", "spa": "spanish", "español": "spanish", "espanol": "spanish", "spanish": "spanish",
        # Swedish
        "sv": "swedish", "swe": "swedish", "svenska": "swedish", "swedish": "swedish",
    }
    return mapping.get(s, s)  # fall back to the input if unknown


def build_templates_by_lang(templates_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group templates by their 'lang' field."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for t in templates_list:
        lang = str(t.get("lang", "")).strip().lower()
        if not lang:
            # Skip any template without language tag
            continue
        grouped.setdefault(lang, []).append(t)
    return grouped


def fill_template(template_text: str, domain: str, language: str, generated_text: str, pii_word_counts: str) -> str:
    """
    Fill placeholders in the template.
    Supports both {var} and {{var}} style placeholders.
    """
    # Two-pass replace to handle both styles.
    filled = template_text

    # First pass: double-curly style {{var}}
    replacements = {
        "{{domain}}": domain,
        "{{language}}": language,
        "{{generated_text}}": generated_text,
        "{{pii_word_counts}}": pii_word_counts,
    }
    for k, v in replacements.items():
        filled = filled.replace(k, v)

    # Second pass: single-curly style {var}
    replacements_single = {
        "{domain}": domain,
        "{language}": language,
        "{generated_text}": generated_text,
        "{pii_word_counts}": pii_word_counts,
    }
    for k, v in replacements_single.items():
        filled = filled.replace(k, v)

    return filled


def main():
    parser = argparse.ArgumentParser(description="Generate multilingual PII prompts from CSV using language-matched templates")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: domain, language, generated_text, pii_word_counts")
    parser.add_argument("--templates", required=True, help="Path to templates JSON (list of dicts with keys: id, lang, template)")
    parser.add_argument("--output", required=True, help="Output JSON file (list of prompt records)")
    parser.add_argument("--lang-col", default="language", help="CSV column name for language (default: language)")
    parser.add_argument("--domain-col", default="domain", help="CSV column name for domain (default: domain)")
    parser.add_argument("--text-col", default="generated_text", help="CSV column name for text (default: generated_text)")
    parser.add_argument("--pii-col", default="pii_word_counts", help="CSV column name for PII tuples (default: pii_word_counts)")
    parser.add_argument("--fallback-english", action="store_true", help="If set, fallback to English template when no language match is found")
    args = parser.parse_args()

    # 1) Load CSV
    print(f"Loading CSV: {args.csv}")
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Loaded {len(df)} rows")

    # 2) Load templates
    print(f"Loading templates JSON: {args.templates}")
    try:
        with open(args.templates, "r", encoding="utf-8") as f:
            templates_list = json.load(f)
        if not isinstance(templates_list, list):
            raise ValueError("Templates JSON must be a list of dictionaries")
    except Exception as e:
        print(f"❌ Failed to read templates: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Loaded {len(templates_list)} templates")

    templates_by_lang = build_templates_by_lang(templates_list)
    if not templates_by_lang:
        print("❌ No templates grouped by language. Check 'lang' keys in templates JSON.", file=sys.stderr)
        sys.exit(1)

    print("✅ Templates grouped by languages:", sorted(templates_by_lang.keys()))

    # 3) Build prompts per record
    out_records = []
    skipped = 0

    for idx, row in df.iterrows():
        raw_lang = str(row.get(args.lang_col, "")).strip()
        lang_key = normalize_language(raw_lang)

        # Try language match
        lang_templates = templates_by_lang.get(lang_key)

        # Optional: fallback to English if not found
        if not lang_templates and args.fallback_english:
            lang_templates = templates_by_lang.get("english")

        if not lang_templates:
            skipped += 1
            continue

        # Extract fields
        domain = str(row.get(args.domain_col, "") or "").strip()
        language = str(row.get(args.lang_col, "") or "").strip()

        # generated_text may contain newlines; keep them
        generated_text = row.get(args.text_col, "")
        if pd.isna(generated_text):
            generated_text = ""
        generated_text = str(generated_text)

        # pii_word_counts likely comes as a string; keep as-is
        pii_word_counts = row.get(args.pii_col, "")
        if pd.isna(pii_word_counts):
            pii_word_counts = ""
        pii_word_counts = str(pii_word_counts)

        if not generated_text or not pii_word_counts:
            skipped += 1
            continue

        # Pick a random template
        template = random.choice(lang_templates)

        # Fill placeholders
        prompt_text = fill_template(
            template_text=template.get("template", ""),
            domain=domain,
            language=language,
            generated_text=generated_text,
            pii_word_counts=pii_word_counts,
        )

        rec = {
            "template_id": template.get("id"),
            "template_lang": template.get("lang"),
            "domain": domain,
            "language": language,
            "generated_text": generated_text,
            "pii_word_counts": pii_word_counts,
            "prompt": prompt_text
        }
        out_records.append(rec)

    # 4) Save output
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(out_records, out_f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(out_records)} prompts to: {args.output}")
    if skipped:
        print(f"ℹ️ Skipped {skipped} rows (missing language match or essential fields)")

if __name__ == "__main__":
    main()
