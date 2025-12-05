#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"config/dataset_storage.json"}

python - <<'PY'
import sys
from pathlib import Path

from src.utils import download_datasets_from_config

config_path = Path(sys.argv[1])
print(f"Downloading datasets using {config_path}...")
download_datasets_from_config(config_path)
print("✅ Dataset download completed")
PY
