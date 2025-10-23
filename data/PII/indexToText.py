import pandas as pd
import ast
from collections import Counter

# Load CSV
df = pd.read_csv("synthetic_pii_finance_test.csv")

# Assuming your text is in a column called 'text' and spans are in 'pii_spans'
def extract_pii_info(row):
    try:
        spans = ast.literal_eval(row['pii_spans'])  # safely convert string to list of dicts
    except (ValueError, SyntaxError):
        return []
    
    text = str(row['generated_text'])
    words = []
    
    for span in spans:
        start, end, label = span['start'], span['end'], span['label']
        substring = text[start:end]  # extract using indices
        words.append((substring.strip(), label))
    
    # Count frequencies
    counter = Counter(words)  # keys are (word, label)
    
    # Convert to list of tuples (word, count, label)
    result = [(word, count, label) for (word, label), count in counter.items()]
    return result

# Apply function to each row
df['pii_word_counts'] = df.apply(extract_pii_info, axis=1)

df = df[['domain','language','generated_text', 'pii_word_counts']]

# 1. Remove rows where list is empty
df = df[df['pii_word_counts'].apply(lambda x: len(x) > 0)]

# 2. Sample up to 1500 rows per language
df = (
    df.groupby("language", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 250), random_state=42))
)

# 3. Keep only the required columns
df = df[['domain', 'language', 'generated_text', 'pii_word_counts']]

# 4. Save results
df.to_csv("pii_test.csv", index=False)

print(f"✅ Saved {len(df)} rows → pii_test.csv")
print(df['language'].value_counts())



