import pandas as pd

# Path to the full ESCI dataset
csv_file = "esci_all_data.csv"

# Read the full dataset
df = pd.read_csv(csv_file)

# Step 1: Keep only rows where small_version == 1
df_small = df[df['small_version'] == 1]

# Step 2: Filter rows where query has 3 words or fewer
# Remove leading/trailing spaces and count words
df_small_filtered = df_small[df_small['query'].str.strip().str.split().str.len() <= 3]

# Optional: check how many rows matched
print("Number of rows with small_version=1 and query <= 3 words:", len(df_small_filtered))

# Save the filtered dataset
df_small_filtered.to_csv("esci_small_filtered.csv", index=False)
print("Saved esci_small_filtered.csv")
import pandas as pd

# Path to the full ESCI dataset
csv_file = "esci_all_data.csv"

# Read the full dataset
df = pd.read_csv(csv_file)

# Step 1: Keep only rows where small_version == 1
df_small = df[df['small_version'] == 1]

# Step 2: Filter rows where query has 3 words or fewer
# Remove leading/trailing spaces and count words
df_small_filtered = df_small[df_small['query'].str.strip().str.split().str.len() <= 3]

# Optional: check how many rows matched
print("Number of rows with small_version=1 and query <= 3 words:", len(df_small_filtered))

# Save the filtered dataset
df_small_filtered.to_csv("esci_small_filtered.csv", index=False)
print("Saved esci_small_filtered.csv")
