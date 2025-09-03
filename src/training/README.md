# Prompt Generation

## Run on a single CSV

```bash
python prompt_generation.py \
  --templates templates/news_headline_classification.json \
  --data data/train_news_headlines.csv \
  --output output/generated_prompts_news.json \
  --seed 42
```
