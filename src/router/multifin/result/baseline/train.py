import os
import re
import gc
import csv
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ==============================
# 1. Configuration
# ==============================
NEWS_CATEGORIES = [
    "Technology", "Industry", "Tax & Accounting",
    "Finance", "Government & Controls", "Business & Management"
]

MODEL_NAMES = [
    "Qwen/Qwen2.5-3B",
    "CohereLabs/aya-expanse-8b",
    "meta-llama/Llama-3.1-8B"
]

HF_TOKEN = os.getenv("HF_TOKEN")

INPUT_CSV = "../../data/test.csv"
OUTPUT_CSV = "llm_inference_results.csv"
CSV_BATCH_SIZE = 50   # 👈 adjust as needed

# ==============================
# 2. LLM Manager (Inference Only)
# ==============================
class LLMManager:
    def __init__(self, model_names, categories):
        self.model_names = model_names
        self.categories = categories
        self.models = {}
        self.tokenizers = {}

    def load_model(self, name):
        if name in self.models:
            return self.models[name], self.tokenizers[name]

        torch.cuda.empty_cache()
        gc.collect()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            quantization_config=bnb,
            token=HF_TOKEN
        )

        self.models[name] = model
        self.tokenizers[name] = tokenizer
        return model, tokenizer

    def extract_label(self, decoded):
        prompt_tags = [
            "Output (Category only):", "Output (Kun kategori):",
            "Salida (Categoría solamente):", "Wyjście (Tylko kategoria):",
            "Çıktı (Sadece Kategori):"
        ]

        raw_output = decoded
        for tag in prompt_tags:
            if tag in raw_output:
                raw_output = raw_output.split(tag)[-1].strip()
                break
        else:
            raw_output = decoded.strip().split("\n")[-1].strip()

        cleaned = raw_output.lower().replace('"', '').replace("'", '').strip()

        for cat in sorted(self.categories, key=len, reverse=True):
            if cat.lower() == cleaned or cat.lower() in cleaned:
                return cat

        cleaned_text = re.sub(r"[^a-zA-Z\s]", "", cleaned)
        return cleaned_text if cleaned_text else "Unknown"

    def predict(self, model_name, text):
        model, tokenizer = self.load_model(model_name)

        prompt = (
            f"You are a strict news classifier. "
            f"Output ONLY one category from {self.categories}. "
            f"No explanation. No punctuation. "
            f"Article: {text} Output (Category only):"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False
            )

        decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        prediction = self.extract_label(decoded)
        return prediction, decoded

# ==============================
# 3. Run Inference (Batch CSV)
# ==============================
def main():
    df = pd.read_csv(INPUT_CSV)

    llm = LLMManager(MODEL_NAMES, NEWS_CATEGORIES)

    # CSV header
    header = ["text", "language", "true_label"]
    for model_name in MODEL_NAMES:
        header.append(f"{model_name}_pred")
        header.append(f"{model_name}_decode")

    # Write header once
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    csv_buffer = []

    print("🚀 Running inference on test set...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        record = [
            row["text"],
            row["lang"],
            row["label"]
        ]

        for model_name in MODEL_NAMES:
            pred, decoded = llm.predict(model_name, row["text"])
            record.extend([pred, decoded])

        csv_buffer.append(record)

        # Flush batch
        if len(csv_buffer) >= CSV_BATCH_SIZE:
            with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(csv_buffer)
            csv_buffer.clear()

    # Flush remaining rows
    if csv_buffer:
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_buffer)

    print(f"\n✅ Inference complete. Results saved to {OUTPUT_CSV}")

# ==============================
# 4. Entry Point
# ==============================
if __name__ == "__main__":
    main()
