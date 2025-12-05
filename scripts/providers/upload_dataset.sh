#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:-}
REMOTE_PATH=${2:-}
ARCHIVE_PATH=${3:-}
TMP_DIR=${4:-}
PARENT_ID=${5:-}

if [[ -z "${REMOTE_PATH}" || -z "${ARCHIVE_PATH}" ]]; then
  echo "upload_dataset.sh: REMOTE_PATH and ARCHIVE_PATH are required" >&2
  exit 1
fi

if command -v gdrive >/dev/null 2>&1 && [[ -n "${PARENT_ID}" ]]; then
  gdrive upload --name "${REMOTE_PATH}.zip" -p "${PARENT_ID}" "${ARCHIVE_PATH}"
  exit 0
fi

# Fallback placeholder: copy into tmp staging area
DEST_DIR="${TMP_DIR}/uploaded_datasets/${REMOTE_PATH}"
mkdir -p "$(dirname "${DEST_DIR}")"
cp "${ARCHIVE_PATH}" "${DEST_DIR}.zip"
echo "[upload_dataset.sh] Staged dataset archive at ${DEST_DIR}.zip"
