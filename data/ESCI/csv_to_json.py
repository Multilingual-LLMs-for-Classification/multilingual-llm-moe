import csv
import json

input_csv = "train_truncated.csv"      # path to your CSV file
output_json = "train_truncated.json"  # path to write JSON

with open(input_csv, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

with open(output_json, "w", encoding="utf-8") as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=2)

print(f"Converted {len(data)} rows from CSV to JSON → {output_json}")
