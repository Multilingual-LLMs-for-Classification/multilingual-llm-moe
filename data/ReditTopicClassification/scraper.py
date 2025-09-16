import praw
import pandas as pd
import csv

# Authenticate Reddit API
reddit = praw.Reddit(
    client_id="GhwWRGDE4PQ0PEfS2bnz1Q",
    client_secret="8iW67fQoZuAT7_WoSeY2MXbVzcjloQ",
    user_agent="moe"
)

# Finance-related subreddits (topics = labels)
subreddits = {
    "stocks": "Stocks",
    "StockMarket": "Stocks",
    "pennystocks": "Stocks",
    "wallstreetbets": "Speculation",
    "investing": "Investing",
    "Bitcoin": "Crypto",
    "Ethereum": "Crypto",
    "CryptoMarkets": "Crypto",
    "CryptoCurrency": "Crypto",
    "personalfinance": "Personal Finance",
    "daytrading": "Trading",
    "AlgoTrading": "Trading"
}

data = []
MAX_TEXT_SIZE = 10 * 1024  # 10 KB

for sub, label in subreddits.items():
    subreddit = reddit.subreddit(sub)
    for post in subreddit.top(limit=1000000):
        title = post.title if post.title else ""
        body = post.selftext if post.selftext else ""
        full_text = title + "\n" + body
        # Remove overly long posts
        if len(full_text.encode('utf-8')) > MAX_TEXT_SIZE:
            continue
        # Replace real line breaks with literal \n
        full_text = full_text.replace("\n", "\\n").replace("\r", "\\n")
        data.append({
            "text": full_text,
            "label": label,
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc
        })
    print(f"Collected {len(data)} posts so far for {sub}")

df = pd.DataFrame(data)

# Save CSV safely
df.to_csv(
    "reddit_finance_topics.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
    escapechar="\\"
)

print("✅ Dataset saved with", len(df), "records")
