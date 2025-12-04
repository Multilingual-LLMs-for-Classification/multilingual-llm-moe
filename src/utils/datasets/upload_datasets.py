"""Utilities for exporting datasets to remote storage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..experts.adapters.download_adapters import AdapterArtifact, AdapterSyncConfig
from ..experts.adapters.upload_adapters import upload_adapters
from .download_datasets import DEFAULT_DATA_ROOT, load_dataset_sync_config


def discover_local_datasets(root: Path = DEFAULT_DATA_ROOT) -> Iterable[Path]:
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def build_dataset_upload_config(
    *,
    provider: str,
    credentials: dict,
    datasets_root: Path = DEFAULT_DATA_ROOT,
) -> AdapterSyncConfig:
    artifacts: list[AdapterArtifact] = []
    for dataset_dir in sorted(discover_local_datasets(datasets_root)):
        rel_path = dataset_dir.relative_to(datasets_root)
        artifacts.append(
            AdapterArtifact(
                name=rel_path.as_posix().replace("/", "-"),
                source=str(rel_path),
                destination=str(rel_path),
                unpack=False,
            )
        )
    return AdapterSyncConfig(provider=provider, credentials=credentials, artifacts=artifacts)


def upload_datasets_from_config(
    config_path: Path,
    *,
    datasets_root: Path = DEFAULT_DATA_ROOT,
    remote_prefix: Optional[Path] = None,
) -> list[Path]:
    config = load_dataset_sync_config(config_path, section="upload")
    return upload_datasets(
        config,
        adapters_root=datasets_root,
        remote_prefix=remote_prefix,
    )


def upload_datasets(
    config: AdapterSyncConfig,
    *,
    datasets_root: Path = DEFAULT_DATA_ROOT,
    remote_prefix: Optional[Path] = None,
) -> list[Path]:
    return upload_adapters(
        config,
        adapters_root=datasets_root,
        remote_prefix=remote_prefix,
    )


__all__ = [
    "discover_local_datasets",
    "build_dataset_upload_config",
    "upload_datasets_from_config",
    "upload_datasets",
]
