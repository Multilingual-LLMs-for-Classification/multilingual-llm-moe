#!/usr/bin/env python3
import pandas as pd

# Input and output file names
input_file = "combined_pii_without_labels.csv"
output_file = "combined_pii_without_labels.csv"

# Load CSV
df = pd.read_csv(input_file)

# Make the 'language' column lowercase (if it exists)
if "language" in df.columns:
    df["language"] = df["language"].astype(str).str.lower()
else:
    raise ValueError("CSV does not have a 'language' column")

# Save to a new CSV
df.to_csv(output_file, index=False)

print(f"✅ Saved new CSV with lowercase language column to {output_file}")
