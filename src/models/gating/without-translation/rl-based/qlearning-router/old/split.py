import pandas as pd

# Load the unified dataset
df = pd.read_csv("unified.csv")

# Define language codes
languages = ["en", "es", "de", "fr", "ja", "zh"]
tasks = ["news", "rating"]

# Initialize containers
test_parts = []
train_parts = []

# For each language, sample 50 per task for test
for lang in languages:
    lang_subset = df[df["language"] == lang]
    test_subset_parts = []

    for task in tasks:
        task_subset = lang_subset[lang_subset["task"] == task]
        n_sample = min(50, len(task_subset))  # make sure there are enough rows
        test_task_sample = task_subset.sample(n=n_sample, random_state=42)
        test_subset_parts.append(test_task_sample)

    # Combine task samples for this language
    test_sample_lang = pd.concat(test_subset_parts)
    train_sample_lang = lang_subset.drop(test_sample_lang.index)

    test_parts.append(test_sample_lang)
    train_parts.append(train_sample_lang)

# Combine all languages
test_df = pd.concat(test_parts).reset_index(drop=True)
train_df = pd.concat(train_parts).reset_index(drop=True)

# Save as JSON
test_df.to_json("test1.json", orient="records", indent=4, force_ascii=False)
train_df.to_json("train1.json", orient="records", indent=4, force_ascii=False)

print(f"✅ Train size: {len(train_df)} | Test size: {len(test_df)}")
print("💾 Files saved as train.json and test.json")
