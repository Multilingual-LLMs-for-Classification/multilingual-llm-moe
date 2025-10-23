import os, sys, csv, urllib.request, pandas as pd
from pathlib import Path

CSV_IN  = "reddit_finance_topics.csv"                # change if needed
CSV_OUT = "reddit_finance_topics_with_languages.csv" # output file
MODEL_FTZ = "lid.176.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# ---------- ensure model ----------
if not Path(MODEL_FTZ).exists():
    print("Downloading fastText model (lid.176.bin)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FTZ)

# ---------- imports ----------
import fasttext
try:
    import langcodes
except Exception as e:
    print("`langcodes` not available. Install with: pip install langcodes")
    sys.exit(1)

# ---------- load data ----------
if not Path(CSV_IN).exists():
    print(f"Input CSV not found: {CSV_IN}")
    sys.exit(1)

df = pd.read_csv(CSV_IN)

if "text" not in df.columns:
    print("The CSV must include a 'text' column.")
    sys.exit(1)

# ---------- load model ----------
model = fasttext.load_model(MODEL_FTZ)

# ---------- helpers ----------
def normalize_text(x):
    if not isinstance(x, str):
        return ""
    # Min cleanup helps detection
    return x.replace("\n", " ").strip()

def code_to_language_name(code: str) -> str:
    """
    Convert ISO language code to a readable English name.
    Handles 2/3-letter codes. Falls back gracefully.
    """
    # Fix a few legacy/edge codes sometimes seen in models
    fixes = {
        "jw": "jv",   # Javanese legacy code
        "iw": "he",   # Hebrew legacy code
        "zh-cn": "zh",
        "zh-tw": "zh",
    }
    code = fixes.get(code.lower(), code.lower())
    try:
        # langcodes knows tons of 2/3-letter codes
        name = langcodes.Language.make(code).display_name("en")
        # Some display names come back capitalized already, but ensure nice casing
        return name
    except Exception:
        # Last resort: show the code itself
        return code

THRESHOLD = 0.70  # raise if you want stricter certainty

# ---------- batch predict for speed ----------
texts = [normalize_text(t) for t in df["text"].tolist()]
# fastText can predict on a list
labels, probs = model.predict(texts, k=1)

# Extract code & probability, map to full name
langs = []
for lab, pr in zip(labels, probs):
    if not lab:            # safety
        langs.append("Unknown")
        continue
    code = lab[0].replace("__label__", "").lower()
    p = float(pr[0]) if pr is not None and len(pr) else 0.0
    if (not texts[len(langs)].strip()) or p < THRESHOLD:
        langs.append("Uncertain")
    else:
        langs.append(code_to_language_name(code))

df["language"] = langs

# ---------- write out ----------
# Preserve original column order and append 'language' at the end
cols = list(df.columns)
if cols[-1] != "language":
    # ensure language comes last
    cols = [c for c in cols if c != "language"] + ["language"]

df.to_csv(CSV_OUT, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Done. Wrote {CSV_OUT}")