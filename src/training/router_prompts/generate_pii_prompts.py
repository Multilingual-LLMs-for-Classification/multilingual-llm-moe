#!/usr/bin/env python3
"""
Generate multilingual PII prompts by combining CSV data with language-matched templates.

CONFIGURATION SECTION - Modify these paths as needed
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
CSV_PATH = "/home/cse/Desktop/multilingual-llm-moe/data/PII/train_final.csv"
TEMPLATES_PATH = "/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/templates/pii_new.json"

# Output file
OUTPUT_PATH = "/home/cse/Desktop/multilingual-llm-moe/src/training/router_prompts/output/pii_prompts_generated.json"

# CSV column names
LANGUAGE_COLUMN = "language"
GENERATED_TEXT_COLUMN = "generated_text"
DOCUMENT_TYPE_COLUMN = "document_type"
EXPANDED_TYPE_COLUMN = "expanded_type"
PII_JSON_COLUMN = "pii_json"

# Options
FALLBACK_TO_ENGLISH = True  # If True, use English template when language match not found
SKIP_EMPTY_TEXT = True      # If True, skip rows with empty generated_text

# ============================================================================
# END CONFIGURATION
# ============================================================================

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
        "fr": "french", "fra": "french", "fre": "french", "français": "french", "francais": "french", "french": "french", "france": "french",
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


def fill_template(template_text: str, generated_text: str) -> str:
    """
    Fill placeholders in the template.
    Supports both {var} and {{var}} style placeholders.
    """
    filled = template_text

    # Replace double-curly style {{generated_text}}
    filled = filled.replace("{{generated_text}}", generated_text)

    # Replace single-curly style {generated_text}
    filled = filled.replace("{generated_text}", generated_text)

    return filled


def main():
    print("=" * 70)
    print("PII Prompt Generator")
    print("=" * 70)
    print()

    # Display configuration
    print("Configuration:")
    print(f"  CSV Path:       {CSV_PATH}")
    print(f"  Templates Path: {TEMPLATES_PATH}")
    print(f"  Output Path:    {OUTPUT_PATH}")
    print(f"  Fallback to EN: {FALLBACK_TO_ENGLISH}")
    print(f"  Skip Empty:     {SKIP_EMPTY_TEXT}")
    print()

    # 1) Load CSV
    print(f"[1/4] Loading CSV: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ Loaded {len(df)} rows")
    print(f"    Columns: {', '.join(df.columns.tolist())}")
    print()

    # 2) Load templates
    print(f"[2/4] Loading templates JSON: {TEMPLATES_PATH}")
    try:
        with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
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

    print(f"✅ Templates grouped by languages: {sorted(templates_by_lang.keys())}")
    for lang, tmpl_list in templates_by_lang.items():
        print(f"    {lang}: {len(tmpl_list)} templates")
    print()

    # 3) Build prompts per record
    print("[3/4] Generating prompts...")
    out_records = []
    skipped = 0
    stats = {
        "no_language_match": 0,
        "empty_text": 0,
        "missing_columns": 0,
        "success": 0
    }

    for idx, row in df.iterrows():
        # Get language
        raw_lang = str(row.get(LANGUAGE_COLUMN, "")).strip()
        lang_key = normalize_language(raw_lang)

        # Try language match
        lang_templates = templates_by_lang.get(lang_key)

        # Optional: fallback to English if not found
        if not lang_templates and FALLBACK_TO_ENGLISH:
            lang_templates = templates_by_lang.get("english")
            if lang_templates:
                stats["no_language_match"] += 1

        if not lang_templates:
            skipped += 1
            stats["no_language_match"] += 1
            continue

        # Extract generated_text
        generated_text = row.get(GENERATED_TEXT_COLUMN, "")
        if pd.isna(generated_text):
            generated_text = ""
        generated_text = str(generated_text)

        if SKIP_EMPTY_TEXT and not generated_text.strip():
            skipped += 1
            stats["empty_text"] += 1
            continue

        # Extract other fields (optional, for metadata)
        document_type = str(row.get(DOCUMENT_TYPE_COLUMN, "") or "").strip()
        expanded_type = str(row.get(EXPANDED_TYPE_COLUMN, "") or "").strip()
        pii_json = row.get(PII_JSON_COLUMN, "")
        if pd.isna(pii_json):
            pii_json = ""
        pii_json = str(pii_json)

        # Pick a random template
        template = random.choice(lang_templates)

        # Fill placeholders
        prompt_text = fill_template(
            template_text=template.get("template", ""),
            generated_text=generated_text,
        )

        rec = {
            "template_id": template.get("id"),
            "template_lang": template.get("lang"),
            "language": raw_lang,
            "document_type": document_type,
            "expanded_type": expanded_type,
            "generated_text": generated_text,
            "pii_json": pii_json,
            "prompt": prompt_text
        }
        out_records.append(rec)
        stats["success"] += 1

    print(f"✅ Generated {len(out_records)} prompts")
    print()

    # 4) Save output
    print(f"[4/4] Saving output to: {OUTPUT_PATH}")
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            json.dump(out_records, out_f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully wrote {len(out_records)} prompts")
    except Exception as e:
        print(f"❌ Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    # Print statistics
    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Total rows processed:     {len(df)}")
    print(f"  Successfully generated:   {stats['success']}")
    print(f"  Skipped (total):          {skipped}")
    if stats["no_language_match"] > 0:
        print(f"    - No language match:    {stats['no_language_match']}")
    if stats["empty_text"] > 0:
        print(f"    - Empty text:           {stats['empty_text']}")
    if stats["missing_columns"] > 0:
        print(f"    - Missing columns:      {stats['missing_columns']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
