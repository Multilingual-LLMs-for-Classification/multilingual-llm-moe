#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import random

ESCI_LABELS = {"E", "S", "C", "I"}

def parse_query_product(text: str):
    """
    Parse: 'Query: ... Product: ...' (case-insensitive).
    Returns (query, product). If parsing fails -> ("","").
    """
    if text is None:
        return "", ""
    t = str(text).strip()
    lower = t.lower()

    q_key = "query:"
    p_key = "product:"

    if q_key in lower and p_key in lower:
        p_idx = lower.find(p_key)
        q_part = t[:p_idx].strip()
        p_part = t[p_idx:].strip()

        q = q_part
        if q.lower().startswith(q_key):
            q = q[len("Query:"):].strip()

        p = p_part
        if p.lower().startswith(p_key):
            p = p[len("Product:"):].strip()

        return q, p

    return "", ""


def main():
    parser = argparse.ArgumentParser(description="Generate JSON prompts from ESCI dataset (text,label,lang,id)")
    parser.add_argument("--csv", required=True, help="Path to CSV file with columns: text,label,lang,id")
    parser.add_argument("--templates", required=True, help="Path to templates JSON file (list of dicts)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional limit for debugging")
    parser.add_argument("--use_csv_template_id", action="store_true",
                        help="If set, uses CSV 'id' (e.g., en-1) to select an exact template. "
                             "If not set, chooses a random template from the same language.")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✅ Loaded {len(df)} rows")

    if args.max_rows:
        df = df.head(args.max_rows).copy()
        print(f"🔎 Using first {len(df)} rows (--max_rows)")

    # Validate required columns
    needed = {"text", "label", "lang", "id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    print(f"Loading templates: {args.templates}")
    with open(args.templates, "r", encoding="utf-8") as f:
        templates_list = json.load(f)
    if not isinstance(templates_list, list):
        raise ValueError("Templates file must be a list of dictionaries")
    print(f"✅ Loaded {len(templates_list)} templates")

    # Group templates by language and by id
    templates_by_lang = {}
    templates_by_id = {}
    for t in templates_list:
        tid = t.get("id")
        lang = t.get("lang")
        if tid:
            templates_by_id[tid] = t
        if lang:
            templates_by_lang.setdefault(lang, []).append(t)

    print(f"✅ Templates grouped by languages: {list(templates_by_lang.keys())}")

    records = []
    skipped_parse = 0
    skipped_lang = 0
    skipped_label = 0
    skipped_template_id = 0

    for _, row in df.iterrows():
        lang = str(row.get("lang", "")).strip().lower()
        label = str(row.get("label", "")).strip().upper()
        text = row.get("text", "")
        csv_tid = str(row.get("id", "")).strip()

        if label not in ESCI_LABELS:
            skipped_label += 1
            continue

        query, product = parse_query_product(text)
        if not query or not product:
            skipped_parse += 1
            continue

        # choose template
        template = None
        if args.use_csv_template_id:
            template = templates_by_id.get(csv_tid)
            if template is None:
                skipped_template_id += 1
                continue
        else:
            lang_templates = templates_by_lang.get(lang, [])
            if not lang_templates:
                skipped_lang += 1
                continue
            template = random.choice(lang_templates)

        prompt_text = (
            template["template"]
            .replace("{query}", query)
            .replace("{product_description}", product)
        )

        records.append(
            {
                "template_id": template["id"],
                "csv_id": csv_tid,
                "lang": lang,
                "label": label,
                "query": query,
                "product_description": product,
                "text": str(text),
                "prompt": prompt_text,
            }
        )

    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(records, out_f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(records)} records: {args.output}")
    print(
        "Skipped: "
        f"bad_label={skipped_label}, parse_fail={skipped_parse}, "
        f"no_template_for_lang={skipped_lang}, missing_template_id={skipped_template_id}"
    )


if __name__ == "__main__":
    main()
