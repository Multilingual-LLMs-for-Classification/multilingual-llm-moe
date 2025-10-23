"""
Stage 2: Language Detection and Filtering
-----------------------------------------
pip install fasttext langdetect pandas
Download fastText model once:
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
"""

import pandas as pd
import fasttext
from langdetect import detect, DetectorFactory

INPUT_FILE = "reddit_finance_raw.csv"
OUTPUT_FILE = "reddit_finance_with_languages.csv"
TARGET_LANGS = ["en", "es", "de", "zh", "ja", "fr"]

# Load fastText model
fasttext_model = fasttext.load_model("lid.176.ftz")
DetectorFactory.seed = 0

def fasttext_detect(text):
    text = str(text or "").replace("\n", " ")[:1000]
    label, prob = fasttext_model.predict(text)
    return label[0].split("__")[-1], float(prob[0])

def langdetect_check(text):
    try:
        return detect(str(text or "")[:1000])
    except:
        return "unknown"

print("Loading raw data ...")
df = pd.read_csv(INPUT_FILE)

print("Detecting languages ...")
df["language_fasttext"] = ""
df["confidence"] = 0.0
df["language_langdetect"] = ""

for i, row in df.iterrows():
    text = f"{row['title']}\n{row['body']}"
    lang_ft, conf = fasttext_detect(text)
    lang_ld = langdetect_check(text)
    df.at[i, "language_fasttext"] = lang_ft
    df.at[i, "confidence"] = conf
    df.at[i, "language_langdetect"] = lang_ld
    if i % 500 == 0:
        print(f"Processed {i} rows...")

print("Filtering target languages ...")
df_filtered = df[
    (df["language_fasttext"].isin(TARGET_LANGS)) &
    (df["language_fasttext"] == df["language_langdetect"]) &
    (df["confidence"] > 0.8)
]

df_filtered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"✅ Saved {len(df_filtered)} filtered rows to {OUTPUT_FILE}")
