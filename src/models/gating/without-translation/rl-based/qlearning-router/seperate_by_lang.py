import json
from pathlib import Path
from collections import defaultdict

def main():
    input_path = Path(__file__).resolve().parent / "test_combined.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    out_dir = input_path.parent / "test_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected test_combined.json to contain a JSON list of objects.")

    # Filter by task == "rating"
    rating_data = [item for item in data if isinstance(item, dict) and item.get("task") == "rating"]

    grouped = defaultdict(list)
    missing_language = 0

    for item in rating_data:
        lang = item.get("language")
        if not lang:
            missing_language += 1
            continue
        grouped[str(lang)].append(item)

    languages = sorted(grouped.keys())

    for lang in languages:
        out_path = out_dir / f"test_{lang}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(grouped[lang], f, ensure_ascii=False, indent=2)

    print(f"Total items in file: {len(data)}")
    print(f"Items with task='rating': {len(rating_data)}")
    print(f"Found languages ({len(languages)}): {languages}")
    if missing_language:
        print(f"Warning: {missing_language} rating items had no 'language' field and were skipped.")
    print(f"Output written to folder: {out_dir}")

if __name__ == "__main__":
    main()
