# Prompt Generation

## Run on a single CSV

```bash
python prompt_generation.py \
  --templates templates/news_headline_classification.json \
  --data ../../../data/train_news_headlines.csv \
  --output output/generated_prompts_news.json \
  --seed 42
```

## PII Prompts Generation

```bash
python3 gen_pii_prompts.py \
  --csv /path/to/your_dataset.csv \
  --templates /path/to/multilingual_pii_router_templates.json \
  --output /path/to/out_prompts.json \
  --fallback-english
```

- Use ```merge_finance_dataset.py``` to merge ```Star Ratings``` and ```News Topics``` datasets.

```bash
python merge_finance_datasets.py \
  --products ratings.json \
  --news news.json \
  --out merged.json \
  --train_out train.json \
  --test_out test.json \
  --test_frac 0.2 \
  --seed 13
```
