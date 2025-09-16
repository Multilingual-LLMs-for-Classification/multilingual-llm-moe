import pandas as pd

# Path to the full ESCI dataset
csv_file = "esci_all_data.csv"

# Read the full dataset
df = pd.read_csv(csv_file)

# Step 1: Keep only rows where small_version == 1
df_small = df[df['small_version'] == 1]

# Step 2: Filter rows where query has at least 5 words
# Remove leading/trailing spaces and count words
df_small_filtered = df_small[df_small['query'].str.strip().str.split().str.len() >= 5]

# Optional: check how many rows matched
print("Number of rows with small_version=1 and query >= 6 words:", len(df_small_filtered))

# Save the filtered dataset
df_small_filtered.to_csv("esci_small_filtered_min5.csv", index=False)
print("Saved esci_small_filtered_min6.csv")
