#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-"config/adapter_storage.json"}

python - <<'PY'
import sys
from pathlib import Path

from src.utils import download_adapters_from_config

config_path = Path(sys.argv[1])
print(f"Downloading adapters using {config_path}...")
download_adapters_from_config(config_path)
print("✅ Download completed")
PY
