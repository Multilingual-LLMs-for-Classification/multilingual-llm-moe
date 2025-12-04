"""Utility for exporting locally cached adapter artifacts to remote storage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .download_adapters import (
    DEFAULT_ADAPTERS_ROOT,
    AdapterSyncConfig,
    AdapterArtifact,
    get_storage_provider,
    load_adapter_sync_config,
)


def discover_local_adapters(root: Path = DEFAULT_ADAPTERS_ROOT) -> Iterable[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("adapter_config.json")]


def build_upload_config(
    *,
    provider: str,
    credentials: dict,
    adapters_root: Path = DEFAULT_ADAPTERS_ROOT,
) -> AdapterSyncConfig:
    artifacts = []
    for adapter_dir in sorted(
        {path.parent for path in discover_local_adapters(adapters_root)}
    ):
        rel_path = adapter_dir.relative_to(adapters_root)
        artifact = AdapterArtifact(
            name=rel_path.as_posix().replace("/", "-"),
            source=str(rel_path),
            destination=str(rel_path),
            unpack=False,
        )
        artifacts.append(artifact)

    return AdapterSyncConfig(
        provider=provider,
        credentials=credentials,
        artifacts=artifacts,
    )


def upload_adapters_from_config(
    config_path: Path,
    *,
    adapters_root: Path = DEFAULT_ADAPTERS_ROOT,
    remote_prefix: Optional[Path] = None,
) -> list[Path]:
    config = load_adapter_sync_config(config_path)
    return upload_adapters(
        config,
        adapters_root=adapters_root,
        remote_prefix=remote_prefix,
    )


def upload_adapters(
    config: AdapterSyncConfig,
    *,
    adapters_root: Path = DEFAULT_ADAPTERS_ROOT,
    remote_prefix: Optional[Path] = None,
) -> list[Path]:
    provider = get_storage_provider(config.provider, config.credentials)
    uploaded_paths: list[Path] = []

    for artifact in config.artifacts:
        local_path = (adapters_root / artifact.source).resolve()
        if not local_path.exists():
            continue
        if local_path.is_file():
            local_dir = local_path.parent
        else:
            local_dir = local_path

        remote_path = Path(artifact.destination)
        uploaded = provider.upload_directory(
            local_dir,
            remote_path,
            remote_prefix=remote_prefix,
        )
        uploaded_paths.append(uploaded)

    return uploaded_paths


__all__ = [
    "discover_local_adapters",
    "upload_adapters_from_config",
    "upload_adapters",
    "build_upload_config",
]

