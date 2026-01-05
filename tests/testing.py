import os
import torch
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

# ===============================
# 1. Config
# ===============================

MODEL_NAMES = [
    "Qwen/Qwen2.5-3B",
    "CohereLabs/aya-expanse-8b",
    "meta-llama/Llama-3.1-8B"
]

LABELS = [
    "Technology",
    "Industry",
    "Tax & Accounting",
    "Finance",
    "Government & Controls",
    "Business & Management"
]

PROMPT = "Classify the following text into the given categories. Give only a single label."

INPUT_CSV = "../src/router/multifin/data/test.csv"
OUTPUT_CSV = "results.csv"

MAX_NEW_TOKENS = 20
BATCH_SIZE = 10   # 🔹 SAVE EVERY N ROWS

# ===============================
# 2. Pydantic Schemas
# ===============================
class ClassificationInput(BaseModel):
    text: str
    prompt: str
    labels: List[str]

class ClassificationOutput(BaseModel):
    label: str
    confidence: float

# ===============================
# 3. Quantization Config
# ===============================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# ===============================
# 4. Load All Models Once
# ===============================
def load_all_models(model_names: List[str]) -> Dict[str, any]:
    pipelines = {}

    for model_name in model_names:
        print(f"🔹 Loading {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True
        )

        model.eval()

        pipelines[model_name] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    return pipelines

# ===============================
# 5. Classification Function
# ===============================
def classify_text(pipe, user_input: ClassificationInput) -> ClassificationOutput:
    labels_str = ", ".join(user_input.labels)

    full_prompt = (
        f"{user_input.prompt}\n"
        f"Text: {user_input.text}\n"
        f"Labels: {labels_str}\n"
        f"Answer:"
    )

    result = pipe(
        full_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    generated_text = result[0]["generated_text"]
    print(generated_text)
    answer_only = generated_text[len(full_prompt):].strip().lower()

    found_labels = [
        label for label in user_input.labels
        if label.lower() in answer_only
    ]

    predicted_label = found_labels[0] if found_labels else user_input.labels[0]

    return ClassificationOutput(
        label=predicted_label,
        confidence=1.0
    )

# ===============================
# 6. Main Loop (Batch-wise Save)
# ===============================
def main():
    df = pd.read_csv(INPUT_CSV)

    # Load models once
    model_pipelines = load_all_models(MODEL_NAMES)

    # Remove old output if exists
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    batch_rows = []
    header_written = False

    for idx, row in df.iterrows():
        result_row = {
            "text": row["text"],
            "true_label": row.get("true_label", "")
        }

        user_input = ClassificationInput(
            text=row["text"],
            prompt=PROMPT,
            labels=LABELS
        )

        for model_name, pipe in model_pipelines.items():
            output = classify_text(pipe, user_input)
            print(f"Predicted Label: {output.label} \n\n")
            result_row[model_name] = output.label

        batch_rows.append(result_row)

        # 🔹 Save batch
        if len(batch_rows) >= BATCH_SIZE:
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(
                OUTPUT_CSV,
                mode="a",
                index=False,
                header=not header_written
            )
            header_written = True
            batch_rows.clear()
            print(f"✅ Saved batch ending at row {idx}")

    # 🔹 Save remaining rows
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            index=False,
            header=not header_written
        )

    print(f"\n✅ Final results saved to {OUTPUT_CSV}")

# ===============================
# 7. Entry Point
# ===============================
if __name__ == "__main__":
    main()
