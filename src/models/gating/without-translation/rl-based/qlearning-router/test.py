import fasttext
import requests
from pathlib import Path

# 1. Download model if not exists
model_path = Path("lid.176.bin")
if not model_path.exists():
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    print("Downloading model...")
    r = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

# 2. Load model
model = fasttext.load_model(str(model_path))

# 3. Text to detect
text = """Imaginez qu’il s’agit d’un avis Amazon. Quelle note sur 5 est la plus probable ?
Nappe qui conserve encore ses pliures de transport même plusieurs semaines après,
aspect peu fluide donc avec des plis apparents. Dommage"""

# 4. Clean (FastText requires single line)
cleaned_text = text.replace("\n", " ").strip()

# 5. Predict
labels, scores = model.predict(cleaned_text, k=3)

print("Predictions:")
for l, s in zip(labels, scores):
    print(l, round(s, 3))
