#!/usr/bin/env python3
"""
Combine rating, news, and PII datasets into unified format for router training/evaluation.

Output format matches what the routing system expects:
{
    "prompt": "...",
    "classification_text": "..." or "generated_text": "...",
    "language": "en",
    "task": "rating" | "news" | "pii",
    "domain": "finance",
    "label": "5" | "Finance" | '[{"text": "...", "label": "...", "occurrence": 1}]'
}
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Language code mapping to standardize language names
LANGUAGE_MAPPING = {
    # Ratings task languages
    'de': 'de',
    'en': 'en',
    'es': 'es',
    'fr': 'fr',
    'ja': 'ja',
    'zh': 'zh',

    # News task languages
    'Danish': 'da',
    'English': 'en',
    'Polish': 'po',
    'Spanish': 'es',
    'Turkish': 'tu',

    # PII task languages
    'Dutch': 'nl',
    'English': 'en',
    'France': 'fr',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Spanish': 'es',
    'Swedish': 'sv'
}

def process_rating_sample(item: Dict) -> Dict:
    """Convert rating dataset sample to unified format."""
    # Get language code
    lang = item.get('language_column', item.get('template_lang', 'en'))
    lang_code = LANGUAGE_MAPPING.get(lang, lang)

    return {
        'prompt': item['prompt'],
        'classification_text': item['review_text'],
        'language': lang_code,
        'task': 'rating',
        'domain': 'finance',
        'label': str(item['stars'])  # Convert to string: "1", "2", "3", "4", "5"
    }

def process_news_sample(item: Dict) -> Dict:
    """Convert news dataset sample to unified format."""
    # Get language code
    lang = item.get('language_column', item.get('template_lang', 'English'))
    lang_code = LANGUAGE_MAPPING.get(lang, lang)

    # Extract the news category from generic_label
    label = item.get('generic_label', 'Unknown')

    return {
        'prompt': item['prompt'],
        'classification_text': item['review_text'],
        'language': lang_code,
        'task': 'news',
        'domain': 'finance',
        'label': label
    }

def process_pii_sample(item: Dict) -> Dict:
    """Convert PII dataset sample to unified format."""
    # Get language code
    lang = item.get('language', item.get('template_lang', 'English'))
    lang_code = LANGUAGE_MAPPING.get(lang, lang)

    return {
        'prompt': item['prompt'],
        'generated_text': item['generated_text'],
        'language': lang_code,
        'task': 'pii',
        'domain': 'finance',
        'label': item['pii_json']  # Already in JSON string format
    }

def combine_datasets(ratings_path: str, news_path: str, pii_path: str) -> List[Dict]:
    """Combine all three datasets into unified format."""

    # Load datasets
    print("Loading datasets...")
    with open(ratings_path) as f:
        ratings = json.load(f)
    with open(news_path) as f:
        news = json.load(f)
    with open(pii_path) as f:
        pii = json.load(f)

    print(f"  Ratings: {len(ratings)} samples")
    print(f"  News: {len(news)} samples")
    print(f"  PII: {len(pii)} samples")

    # Process each dataset
    print("\nProcessing datasets...")
    combined = []

    for item in ratings:
        combined.append(process_rating_sample(item))

    for item in news:
        combined.append(process_news_sample(item))

    for item in pii:
        combined.append(process_pii_sample(item))

    print(f"  Total combined: {len(combined)} samples")

    return combined

def split_train_test(data: List[Dict], test_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Split data into train and test sets while maintaining task balance."""

    random.seed(seed)

    # Group by task
    by_task = {'rating': [], 'news': [], 'pii': []}
    for item in data:
        by_task[item['task']].append(item)

    train_data = []
    test_data = []

    # Split each task separately to maintain balance
    for task, items in by_task.items():
        random.shuffle(items)
        split_idx = int(len(items) * (1 - test_ratio))
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])

        print(f"  {task}: {len(items[:split_idx])} train, {len(items[split_idx:])} test")

    # Shuffle combined datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data

def print_statistics(data: List[Dict], name: str):
    """Print dataset statistics."""
    from collections import Counter

    print(f"\n{name} Statistics:")
    print("=" * 60)

    # Task distribution
    task_counts = Counter([item['task'] for item in data])
    print(f"Total samples: {len(data)}")
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count} ({count/len(data)*100:.1f}%)")

    # Language distribution per task
    print("\nLanguage distribution per task:")
    for task in sorted(task_counts.keys()):
        task_items = [item for item in data if item['task'] == task]
        lang_counts = Counter([item['language'] for item in task_items])
        print(f"\n  {task}:")
        for lang, count in sorted(lang_counts.items()):
            print(f"    {lang}: {count}")

def main():
    """Main function to combine and split datasets."""

    # Input paths
    base_dir = Path(__file__).parent
    ratings_path = base_dir / "ratings_test.json"
    news_path = base_dir / "generated_prompts_news.json"
    pii_path = base_dir / "pii_prompts_generated.json"

    # Output paths
    output_dir = Path("/home/cse/Desktop/multilingual-llm-moe/src/models/gating/without-translation/rl-based/qlearning-router")
    train_output = output_dir / "train_combined.json"
    test_output = output_dir / "test_combined.json"
    full_output = output_dir / "full_combined.json"

    print("=" * 60)
    print("COMBINING DATASETS FOR ROUTER TRAINING")
    print("=" * 60)
    print()

    # Combine datasets
    combined = combine_datasets(ratings_path, news_path, pii_path)

    # Split into train/test
    print("\nSplitting into train/test sets (80/20 split)...")
    train_data, test_data = split_train_test(combined, test_ratio=0.2, seed=42)

    # Print statistics
    print_statistics(train_data, "TRAINING SET")
    print_statistics(test_data, "TEST SET")

    # Save datasets
    print("\n" + "=" * 60)
    print("Saving datasets...")
    print("=" * 60)

    with open(full_output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"✅ Full dataset: {full_output} ({len(combined)} samples)")

    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Training set: {train_output} ({len(train_data)} samples)")

    with open(test_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Test set: {test_output} ({len(test_data)} samples)")

    print("\n" + "=" * 60)
    print("DATASET COMBINATION COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Update router_config.json to use train_combined.json and test_combined.json")
    print("  2. Run training: python main.py --mode train")
    print("  3. Run evaluation: python main.py --mode eval")

if __name__ == "__main__":
    main()
