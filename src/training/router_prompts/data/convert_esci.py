#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Any, List

LANG_MAP = {
    "english": "en",
    "spanish": "es",
    "japanese": "ja",
    "chinese": "zh",
}

LABEL_MAP = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}

def main():
    ap = argparse.ArgumentParser(description="Convert ESCI prompt JSON to unified format")
    ap.add_argument("--input", required=True, help="Input JSON (list of dicts) from esci_prompts.py")
    ap.add_argument("--output", required=True, help="Output JSON file")
    ap.add_argument("--domain", default="finance", help="Value for output field 'domain' (default: finance)")
    ap.add_argument("--task", default="esci", help="Value for output field 'task' (default: esci)")
    ap.add_argument(
        "--label_style",
        choices=["full", "letter"],
        default="letter",
        help="full: Exact/Substitute/Complement/Irrelevant ; letter: E/S/C/I",
    )
    ap.add_argument(
        "--classification_text_source",
        choices=["text", "query", "product", "query_product"],
        default="text",
        help="What to put into classification_text",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")

    out: List[Dict[str, Any]] = []

    for r in data:
        prompt = r.get("prompt", "")
        lang_long = str(r.get("lang", "")).strip().lower()
        lang = LANG_MAP.get(lang_long, lang_long or "en")

        # label
        label_raw = str(r.get("label", "")).strip().upper()
        if args.label_style == "full":
            label = LABEL_MAP.get(label_raw, label_raw)
        else:
            label = label_raw

        # classification_text
        if args.classification_text_source == "text":
            classification_text = r.get("text", "")
        elif args.classification_text_source == "query":
            classification_text = r.get("query", "")
        elif args.classification_text_source == "product":
            classification_text = r.get("product_description", "")
        else:  # query_product
            q = r.get("query", "")
            p = r.get("product_description", "")
            classification_text = f"Query: {q}\nProduct: {p}".strip()

        out.append(
            {
                "prompt": prompt,
                "classification_text": classification_text,
                "language": lang,
                "task": args.task,
                "domain": args.domain,
                "label": label,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(out)} records -> {args.output}")


if __name__ == "__main__":
    main()
