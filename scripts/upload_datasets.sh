#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"config/dataset_storage.json"}
python - <<'PY'
import sys
from pathlib import Path

from src.utils import upload_datasets_from_config

config_path = Path(sys.argv[1])
print(f"Uploading datasets using {config_path}...")
upload_datasets_from_config(config_path)
print("✅ Dataset upload completed")
PY
"$CONFIG_PATH"
