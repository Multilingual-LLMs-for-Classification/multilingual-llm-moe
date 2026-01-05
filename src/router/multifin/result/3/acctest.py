import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv("llm_news_test_outputs.csv")

for lang, group in df.groupby("language"):
    print(lang)
    y_true = group["true_category"]
    y_pred = group["pred_category"]
    acc = accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(acc, precision_weighted, recall_weighted, f1_weighted)