import json
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "test_combined.json"
    output_path = base_dir / "test_combined_esci.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected test_combined.json to contain a JSON list.")

    rating_data = [
        item for item in data
        if isinstance(item, dict) and item.get("task") == "esci"
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rating_data, f, ensure_ascii=False, indent=2)

    print(f"Total items: {len(data)}")
    print(f"Rating items: {len(rating_data)}")
    print(f"Wrote: {output_path}")

if __name__ == "__main__":
    main()
