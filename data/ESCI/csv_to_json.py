import pandas as pd

# Load the CSV
csv_file = "esci_small_filtered_min5.csv"
df = pd.read_csv(csv_file)

# Maximum records per locale
max_records = 8000

# Function to downsize a locale
def downsize_locale(df_locale, max_records):
    # Shuffle unique queries for randomness
    unique_queries = df_locale['query'].drop_duplicates().sample(frac=1, random_state=42)
    
    # Keep adding queries until max_records is reached
    kept_queries = []
    total_rows = 0
    for q in unique_queries:
        query_rows = df_locale[df_locale['query'] == q]
        if total_rows + len(query_rows) > max_records:
            break
        kept_queries.append(q)
        total_rows += len(query_rows)
    
    # Return only rows with the kept queries
    return df_locale[df_locale['query'].isin(kept_queries)]

# Process each locale separately
df_result = pd.DataFrame()

for locale in df['product_locale'].unique():
    df_locale = df[df['product_locale'] == locale]
    
    if len(df_locale) > max_records:
        df_locale_downsized = downsize_locale(df_locale, max_records)
        df_result = pd.concat([df_result, df_locale_downsized])
    else:
        df_result = pd.concat([df_result, df_locale])

# Reset index
df_result.reset_index(drop=True, inplace=True)

# Save the downsized dataset
df_result.to_csv("esci_min5_downsized.csv", index=False)
print("Saved downsized dataset as esci_min5_downsized.csv")
print(df_result['product_locale'].value_counts())
