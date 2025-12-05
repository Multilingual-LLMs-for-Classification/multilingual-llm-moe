#!/usr/bin/env bash
set -euo pipefail

FILE_ID=${1:-}
DEST_DIR=${2:-}
ARTIFACT_PATH=${3:-}
TMP_DIR=${4:-}
PARENT_ID=${5:-}

if [[ -z "${FILE_ID}" ]]; then
  echo "download_dataset.sh: FILE_ID is required" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_PATH}"

if command -v gdrive >/dev/null 2>&1 && [[ -n "${FILE_ID}" ]]; then
  gdrive download --path "${ARTIFACT_PATH}" "${FILE_ID}"
  exit 0
fi

echo "[download_dataset.sh] No gdrive CLI detected; creating placeholder file"
>"${ARTIFACT_PATH}/placeholder"
exit 0
