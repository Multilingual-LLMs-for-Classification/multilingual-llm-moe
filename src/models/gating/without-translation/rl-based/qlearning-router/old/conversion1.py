import json
from collections import defaultdict

# load json (list of objects)
with open("test2_updated_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

grouped = defaultdict(list)

# group by language (preserves original order within each language)
for obj in data:
    lang = obj.get("language", "unknown")
    grouped[lang].append(obj)

# choose language order
language_order = sorted(grouped.keys())  
# or explicitly:
# language_order = ["zh", "ja", "en"]

# flatten back into a single list
reordered = []
for lang in language_order:
    reordered.extend(grouped[lang])

# save back
with open("test2_grouped_languages_flat.json", "w", encoding="utf-8") as f:
    json.dump(reordered, f, ensure_ascii=False, indent=2)
