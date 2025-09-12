#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import random

def main():
    parser = argparse.ArgumentParser(description="Generate JSON prompts from ESCI dataset")
    parser.add_argument("--csv", required=True, help="Path to downsized CSV file")
    parser.add_argument("--templates", required=True, help="Path to templates JSON file (list of dicts)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    # 1) Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✅ Loaded {len(df)} rows")

    # 2) Load templates (list of dicts)
    print(f"Loading templates: {args.templates}")
    with open(args.templates, "r", encoding="utf-8") as f:
        templates_list = json.load(f)
    if not isinstance(templates_list, list):
        raise ValueError("Templates file must be a list of dictionaries")
    print(f"✅ Loaded {len(templates_list)} templates")

    # Group templates by language
    templates_by_lang = {}
    for t in templates_list:
        lang = t.get("lang")
        if not lang:
            continue
        templates_by_lang.setdefault(lang, []).append(t)
    print(f"✅ Templates grouped by languages: {list(templates_by_lang.keys())}")

    # 3) Mapping product_locale → lang
    locale_to_lang = {
        "us": "english",
        "es": "spanish",
        "jp": "japanese"
    }

    # 4) Build records
    records = []
    for _, row in df.iterrows():
        product_locale = str(row.get("product_locale", "")).strip().lower()
        lang = locale_to_lang.get(product_locale)
        if not lang:
            continue

        query = str(row.get("query", "")).strip()
        esci_label = row.get("esci_label", "")
        merged_description = " ".join(
            str(v) for v in [
                row.get("product_title", ""),
                row.get("product_description", ""),
                row.get("product_bullet_point", "")
            ] if v and str(v) != "nan"
        ).strip()

        # Skip if missing essentials
        if not query or not merged_description:
            continue

        # Pick random template for this language
        lang_templates = templates_by_lang.get(lang, [])
        if not lang_templates:
            continue
        template = random.choice(lang_templates)

        # Build prompt: replace {query} and {product_description} placeholders
        prompt_text = template["template"].replace("{query}", query).replace("{product_description}", merged_description)

        record = {
            "template_id": template["id"],
            "language_column": lang,
            "query": query,
            "esci_label": esci_label,
            "merged_description": merged_description,
            "prompt": prompt_text,
        }

        records.append(record)

    # 5) Save to JSON
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(records, out_f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote JSON file with {len(records)} records: {args.output}")

if __name__ == "__main__":
    main()
