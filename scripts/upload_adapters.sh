#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"config/adapter_storage.json"}

python - <<'PY'
import sys
from pathlib import Path

from src.utils import upload_adapters_from_config

config_path = Path(sys.argv[1])
print(f"Uploading adapters using {config_path}...")
upload_adapters_from_config(config_path)
print("✅ Upload completed")
PY
