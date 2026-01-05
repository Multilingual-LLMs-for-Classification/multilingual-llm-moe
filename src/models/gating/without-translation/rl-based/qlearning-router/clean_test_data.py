#!/usr/bin/env python3
"""
Clean test2_updated.json by removing category suffix from classification_text
"""
import json
import re

# Read original data
with open('test2_updated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Processing {len(data)} items...")

# Clean each item
cleaned_count = 0
for item in data:
    if 'classification_text' in item:
        original = item['classification_text']
        # Remove category suffix like "(Category: office_product)"
        cleaned = re.sub(r'\s*\(Category:\s*[^)]+\)\s*$', '', original)
        if cleaned != original:
            item['classification_text'] = cleaned
            cleaned_count += 1

print(f"Cleaned {cleaned_count} items")

# Save cleaned data
with open('test2_updated_clean.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Saved to test2_updated_clean.json")

# Verify
print("\nSample before/after:")
with open('test2_updated.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)
print(f"Original: {original_data[0]['classification_text'][-80:]}")
print(f"Cleaned:  {data[0]['classification_text'][-80:]}")
