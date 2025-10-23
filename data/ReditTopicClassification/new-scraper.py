"""
Multilingual Finance Reddit Scraper with Double Language Check + Top Comment
-----------------------------------------------------------------------------
Requirements:
    pip install praw fasttext pandas langdetect
Download fastText language model once:
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
"""

import praw
import fasttext
import pandas as pd
from langdetect import detect, DetectorFactory
from datetime import datetime
import prawcore

# ---------- 1. CONFIGURATION ----------
CLIENT_ID = "GhwWRGDE4PQ0PEfS2bnz1Q"
CLIENT_SECRET = "8iW67fQoZuAT7_WoSeY2MXbVzcjloQ"
USER_AGENT = "moe"


# Finance-related subreddits across languages
SUBREDDITS = [
    # English
    "investing", "finance", "stocks", "wallstreetbets", "CryptoCurrency",
    # Spanish
    "Finanzas", "Inversiones", "CriptoMonedas",
    # German
    "Finanzen", "Aktien", "Krypto",
    # Chinese (check validity; some may not exist)
    "投资", "股票", "加密货币",
    # Japanese (check validity)
    "日本株", "投資", "仮想通貨",
    # French
    "FinanceFR", "Investir", "CryptoFR"
]

TARGET_LANGS = ["en", "es", "de", "zh", "ja", "fr"]
LIMIT_PER_SUB = 10000         # number of posts per subreddit
OUTPUT_FILE = "multilingual_finance_reddit_with_top_comment.csv"

# ---------- 2. SETUP ----------
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

fasttext_model = fasttext.load_model("lid.176.ftz")
DetectorFactory.seed = 0   # reproducible langdetect results

def second_opinion(text):
    """Detect language using langdetect as a second check."""
    try:
        return detect(text)
    except:
        return "unknown"

def get_top_comment(post):
    """Return the text of the top (first-ranked) comment or empty string."""
    try:
        post.comment_sort = "top"
        post.comments.replace_more(limit=0)
        if post.comments:
            return post.comments[0].body
    except prawcore.exceptions.PrawcoreException:
        return ""  # network / rate limit issue
    except Exception:
        return ""
    return ""

def subreddit_exists(name):
    """Check if a subreddit exists and is accessible."""
    try:
        reddit.subreddits.search_by_name(name, exact=True)
        return True
    except prawcore.exceptions.NotFound:
        return False
    except Exception:
        return False

# ---------- 3. SCRAPING ----------
records = []

for sub in SUBREDDITS:
    if not subreddit_exists(sub):
        print(f"Skipping {sub} (not found or private)")
        continue

    print(f"Scraping r/{sub} ...")
    try:
        for post in reddit.subreddit(sub).new(limit=LIMIT_PER_SUB):
            text = f"{post.title}\n{post.selftext or ''}"
            # fastText prediction (with confidence)
            label, prob = fasttext_model.predict(text.replace("\n", " ")[:1000])
            lang_fasttext = label[0].split("__")[-1]
            conf = float(prob[0])

            if lang_fasttext in TARGET_LANGS:
                # second detector
                lang_langdetect = second_opinion(text[:1000])
                # get top comment text
                top_comment = get_top_comment(post)

                records.append({
                    "id": post.id,
                    "subreddit": sub,
                    "language_fasttext": lang_fasttext,
                    "confidence": conf,
                    "language_langdetect": lang_langdetect,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z",
                    "title": post.title,
                    "body": post.selftext,
                    "top_comment": top_comment,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "url": post.url
                })
    except Exception as e:
        print(f"Error scraping r/{sub}: {e}")
        continue

# ---------- 4. SAVE TO CSV ----------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"\n✅ Saved {len(df)} rows to {OUTPUT_FILE}")

# ---------- 5. OPTIONAL QUALITY REPORT ----------
if not df.empty:
    match_rate = (df["language_fasttext"] == df["language_langdetect"]).mean()
    print(f"Language match rate (fastText vs langdetect): {match_rate:.2%}")
    print("\nSample mismatches:")
    print(df[df["language_fasttext"] != df["language_langdetect"]].head())
else:
    print("No records saved. Check subreddit list or scraping limits.")
