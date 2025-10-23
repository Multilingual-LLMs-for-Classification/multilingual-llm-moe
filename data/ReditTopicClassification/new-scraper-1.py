"""
Stage 1: Fast Reddit Scraper (no language detection)
----------------------------------------------------
pip install praw pandas
"""

import praw
import pandas as pd
from datetime import datetime
import prawcore

CLIENT_ID = "GhwWRGDE4PQ0PEfS2bnz1Q"
CLIENT_SECRET = "8iW67fQoZuAT7_WoSeY2MXbVzcjloQ"
USER_AGENT = "moe"


SUBREDDITS = [
    "investing", "finance", "stocks", "wallstreetbets", "CryptoCurrency",
    "Finanzas", "Inversiones", "CriptoMonedas",
    "Finanzen", "Aktien", "Krypto",
    "FinanceFR", "Investir", "CryptoFR"
    # add or adjust, but remove non-existent subs for speed
]

LIMIT_PER_SUB = 2000
OUTPUT_RAW = "reddit_finance_raw.csv"

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

def get_top_comment(post):
    try:
        post.comment_sort = "top"
        post.comments.replace_more(limit=0)
        if post.comments:
            return post.comments[0].body
    except prawcore.exceptions.PrawcoreException:
        return ""
    except Exception:
        return ""
    return ""

records = []
for sub in SUBREDDITS:
    print(f"Scraping r/{sub} ...")
    try:
        for post in reddit.subreddit(sub).new(limit=LIMIT_PER_SUB):
            records.append({
                "id": post.id,
                "subreddit": sub,
                "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z",
                "title": post.title,
                "body": post.selftext,
                "top_comment": get_top_comment(post),
                "score": post.score,
                "num_comments": post.num_comments,
                "url": post.url
            })
    except Exception as e:
        print(f"Error scraping r/{sub}: {e}")
        continue

pd.DataFrame(records).to_csv(OUTPUT_RAW, index=False, encoding="utf-8")
print(f"\n✅ Saved {len(records)} rows to {OUTPUT_RAW}")
