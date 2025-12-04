"""Utilities for synchronising dataset archives with local storage."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..experts.adapters.download_adapters import (
    AdapterArtifact,
    AdapterSyncConfig,
    download_adapters,
)
from ..experts.adapters.download_adapters import _load_structured_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def download_datasets_from_config(
    config_path: Path,
    *,
    target_root: Optional[Path] = None,
) -> List[Path]:
    """Download datasets described in a JSON/YAML configuration file."""

    if target_root is None:
        target_root = DEFAULT_DATA_ROOT

    config = load_dataset_sync_config(config_path, section="download")
    return download_datasets(config, target_root=target_root)


def download_datasets(
    config: AdapterSyncConfig,
    *,
    target_root: Optional[Path] = None,
) -> List[Path]:
    if target_root is None:
        target_root = DEFAULT_DATA_ROOT

    return download_adapters(config, target_root=target_root)


def load_dataset_sync_config(config_path: Path, section: Optional[str] = None) -> AdapterSyncConfig:
    data = _load_structured_file(config_path)

    section_data = _resolve_section(data, section)

    provider = section_data.get("provider")
    if not provider:
        raise ValueError("Dataset config missing 'provider'")

    credentials = section_data.get("credentials", {})
    artifacts_raw = section_data.get("artifacts", [])
    artifacts = [
        AdapterArtifact(
            name=item["name"],
            source=item["source"],
            destination=item["destination"],
            unpack=bool(item.get("unpack", False)),
        )
        for item in artifacts_raw
    ]

    return AdapterSyncConfig(provider=provider, credentials=credentials, artifacts=artifacts)


def _resolve_section(config_data, section: Optional[str]):
    if section is None:
        return config_data

    if isinstance(config_data, dict) and section in config_data:
        return config_data[section]

    if isinstance(config_data, dict) and "provider" in config_data:
        return config_data

    raise ValueError(f"Section '{section}' not found in dataset config")


__all__ = [
    "download_datasets_from_config",
    "download_datasets",
    "DEFAULT_DATA_ROOT",
    "load_dataset_sync_config",
]
