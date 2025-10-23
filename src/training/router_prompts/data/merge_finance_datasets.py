# merge_finance_datasets.py  (py38/py39 compatible)
import json
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import random
from collections import defaultdict

ALLOWED_NEWS = {
    "Tax & Accounting",
    "Business & Management",
    "Finance",
    "Industry",
    "Technology",
    "Government & Controls",
}
ALLOWED_STARS = {"1", "2", "3", "4", "5"}

LANG_MAP = {
    "en": "en", "english": "en",
    "fr": "fr", "french": "fr",
    "ja": "ja", "japanese": "ja",
    "de": "de", "german": "de",
    "es": "es", "spanish": "es",
    "zh": "zh", "chinese": "zh",
}

# ---------------- I/O helpers ----------------
def load_any_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f]
        return json.load(f)

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ---------------- Normalization ----------------
def normalize_language(x: Optional[str]) -> str:
    if not x:
        return "en"
    key = x.strip().lower()
    return LANG_MAP.get(key, key[:2] if len(key) >= 2 else "en")

def render_prompt(rec: Dict[str, Any]) -> str:
    p = rec.get("prompt")
    if p:
        return p
    templ = rec.get("template_text") or ""
    text = rec.get("review_text") or ""
    if "{review_text}" in templ:
        return templ.replace("{review_text}", text)
    return (templ + ("\n" if templ and text else "") + text).strip()

def label_from_product(rec: Dict[str, Any]) -> Optional[str]:
    stars = rec.get("stars")
    if stars is not None:
        s = str(stars).strip()
        m = re.search(r"[1-5]", s)  # accepts "5", "5.0", "⭐️5", "5 stars", etc.
        if m and m.group(0) in ALLOWED_STARS:
            return m.group(0)
    # fallback via sentiment → {1,3,5}
    sent = (rec.get("sentiment") or "").strip().lower()
    sent_map = {"negative": "1", "neutral": "3", "positive": "5"}
    return sent_map.get(sent)

def normalize_news_label(lbl: Optional[str]) -> Optional[str]:
    if not lbl:
        return None
    x = lbl.strip()
    x = x.replace(" and ", " & ").replace("And", "&")
    x = x.replace("Government and Controls", "Government & Controls")
    x = x.replace("Tax and Accounting", "Tax & Accounting")
    x = x.replace("Business and Management", "Business & Management")
    if x in ALLOWED_NEWS:
        return x
    t = x.title().replace(" And ", " & ")
    if t in ALLOWED_NEWS:
        return t
    return None

def label_from_news(rec: Dict[str, Any]) -> Optional[str]:
    return normalize_news_label(rec.get("generic_label") or rec.get("label"))

# ---------------- Merge ----------------
def merge(product_path: str, news_path: str, out_path: Optional[str] = None, domain: str = "finance") -> List[Dict[str, Any]]:
    products = load_any_json(product_path)
    news = load_any_json(news_path)

    merged, skipped = [], []

    # Product reviews → task=rating
    for r in products:
        label = label_from_product(r)
        if label not in ALLOWED_STARS:
            skipped.append(("product", r.get("row_index")))
            continue
        merged.append({
            "prompt":   render_prompt(r),
            "language": normalize_language(r.get("language_column")),
            "task":     "rating",
            "domain":   domain,
            "label":    label
        })

    # News → task=news
    for r in news:
        label = label_from_news(r)
        if label not in ALLOWED_NEWS:
            skipped.append(("news", r.get("row_index")))
            continue
        merged.append({
            "prompt":   render_prompt(r),
            "language": normalize_language(r.get("language_column")),
            "task":     "news",
            "domain":   domain,
            "label":    label,
        })

    if out_path:
        # auto pick json vs jsonl by extension
        if Path(out_path).suffix.lower() == ".jsonl":
            save_jsonl(out_path, merged)
        else:
            save_json(out_path, merged)
        print(f"✅ Wrote merged: {len(merged)} rows → {out_path}")

    if skipped:
        n_prod = sum(1 for s in skipped if s[0] == "product")
        n_news = sum(1 for s in skipped if s[0] == "news")
        print(f"⚠️ Skipped {len(skipped)} rows (product={n_prod}, news={n_news}) due to missing/invalid labels.")

    return merged

# ---------------- Balanced test split ----------------
def _sample_k_per_bucket(buckets: Dict[Any, List[int]], k: int, rng: random.Random) -> List[int]:
    picked = []
    for key, idxs in buckets.items():
        if len(idxs) == 0 or k == 0:
            continue
        if len(idxs) < k:
            # if some bucket is too small, take all it has
            picked.extend(idxs)
        else:
            picked.extend(rng.sample(idxs, k))
    return picked

def balanced_split(
    rows: List[Dict[str, Any]],
    test_frac: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    - For task='news': balance test set by language → same count per language.
    - For task='rating': balance test set by (language, star) → same count per (lang, star).
    """
    rng = random.Random(seed)
    idxs_by_lang_news: Dict[str, List[int]] = defaultdict(list)
    idxs_by_lang_star: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for i, r in enumerate(rows):
        task = r.get("task")
        lang = r.get("language", "en")
        if task == "news":
            idxs_by_lang_news[lang].append(i)
        elif task == "rating":
            star = str(r.get("label"))
            idxs_by_lang_star[(lang, star)].append(i)

    # --- NEWS: equal K per language ---
    news_counts = [len(v) for v in idxs_by_lang_news.values() if len(v) > 0]
    news_k = 0
    if news_counts:
        news_min = min(news_counts)
        news_k = max(1, int(news_min * test_frac)) if news_min > 0 else 0
    news_test = _sample_k_per_bucket(idxs_by_lang_news, news_k, rng)

    # --- RATING: equal K per (language, star) ---
    rating_counts = [len(v) for v in idxs_by_lang_star.values() if len(v) > 0]
    rating_k = 0
    if rating_counts:
        rating_min = min(rating_counts)
        rating_k = max(1, int(rating_min * test_frac)) if rating_min > 0 else 0
    rating_test = _sample_k_per_bucket(idxs_by_lang_star, rating_k, rng)

    test_set = set(news_test) | set(rating_test)
    test_rows = [rows[i] for i in sorted(test_set)]
    train_rows = [r for i, r in enumerate(rows) if i not in test_set]

    # Small summary
    print(f"🧪 Test sizes → news/lang: k={news_k} each, rating/(lang,star): k={rating_k} each")
    print(f"🧪 Test total: {len(test_rows)}  |  Train total: {len(train_rows)}")

    return train_rows, test_rows

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Merge product reviews and news datasets with labels for SFT, and create balanced train/test splits."
    )
    ap.add_argument("--products", required=True, help="Path to product reviews (.json or .jsonl)")
    ap.add_argument("--news", required=True, help="Path to news dataset (.json or .jsonl)")
    ap.add_argument("--out", help="(Optional) Write merged dataset to .json or .jsonl")
    ap.add_argument("--train_out", help="Write train split to .json")
    ap.add_argument("--test_out", help="Write test split to .json (balanced)")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Fraction per bucket to allocate to test (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--domain", default="finance", help="Domain field for merged rows (default: finance)")
    args = ap.parse_args()

    merged_rows = merge(args.products, args.news, args.out, domain=args.domain)

    if args.train_out and args.test_out:
        train_rows, test_rows = balanced_split(merged_rows, test_frac=args.test_frac, seed=args.seed)
        save_json(args.train_out, train_rows)
        save_json(args.test_out, test_rows)
        print(f"✅ Saved train → {args.train_out}")
        print(f"✅ Saved test  → {args.test_out}")
