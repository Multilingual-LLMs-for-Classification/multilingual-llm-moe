import pandas as pd

# Load the CSV file
df = pd.read_csv("unified.csv")

# Replace language names with codes
df["language"] = df["language"].replace({
    "english": "en",
    "spanish": "es"
})

# Save back to CSV
df.to_csv("unified.csv", index=False)

print("✅ Languages replaced in 'unified.csv'")
