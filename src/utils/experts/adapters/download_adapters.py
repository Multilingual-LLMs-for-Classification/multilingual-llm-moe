"""Utilities for pulling QLoRA adapter artifacts from remote storage.

This module implements a configurable download pipeline that can ingest adapter
artifacts stored on external providers (e.g., Google Drive, local mirrors) and
materialise them inside the repository's `adapters/` directory structure.

The downloader is driven by a declarative configuration file so that open-source
consumers can mount their own storage endpoints without modifying code.  A sample
JSON configuration looks like:

```
{
  "provider": "google_drive",
  "credentials": {
    "service_account_file": "./service-account.json",
    "parent_folder_id": "1AbCdEf..."   // optional, used for uploads
  },
  "artifacts": [
    {
      "name": "finance-news-mistral",
      "source": "1baSe64GoogleDriveFileId",
      "destination": "finance/news_classification/mistral",
      "unpack": true
    }
  ]
}
```

The same provider factory is reused by the upload utilities so credentials only
need to be declared once.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTERS_ROOT = PROJECT_ROOT / "src" / "models" / "experts" / "llms" / "adapters"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdapterArtifact:
    """Description of a single adapter artifact in remote storage."""

    name: str
    source: str
    destination: str
    unpack: bool = False


@dataclass(slots=True)
class AdapterSyncConfig:
    """Configuration block describing a storage provider and its artifacts."""

    provider: str
    credentials: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[AdapterArtifact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider interfaces
# ---------------------------------------------------------------------------


class StorageProvider(ABC):
    """Abstract interface implemented by all adapter storage providers."""

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        self.credentials = credentials or {}

    # Download -----------------------------------------------------------------
    @abstractmethod
    def download_artifact(
        self,
        artifact: AdapterArtifact,
        target_root: Path,
    ) -> Path:
        """Download *artifact* into *target_root*; returns the materialised path."""

    # Upload -------------------------------------------------------------------
    def upload_directory(
        self,
        local_dir: Path,
        remote_path: Path,
        *,
        archive_format: str = "zip",
        remote_prefix: Optional[Path] = None,
    ) -> Path:
        """Upload *local_dir* contents to *remote_path* (optional override).

        Upload semantics differ across providers.  The default implementation
        raises ``NotImplementedError`` so subclasses can opt-in to upload
        support.
        """

        raise NotImplementedError(
            f"Upload not implemented for provider {self.__class__.__name__}"
        )


class LocalStorageProvider(StorageProvider):
    """Copies artifacts from/to a directory on the local filesystem."""

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials)
        base_path = self.credentials.get("base_path")
        if not base_path:
            raise ValueError(
                "Local storage provider requires a 'base_path' credential"
            )
        self.base_path = Path(base_path).expanduser().resolve()

    def download_artifact(
        self,
        artifact: AdapterArtifact,
        target_root: Path,
    ) -> Path:
        source_path = self.base_path / artifact.source
        if not source_path.exists():
            raise FileNotFoundError(f"Local artifact not found: {source_path}")

        destination_path = (target_root / artifact.destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
            return destination_path

        if destination_path.exists() and destination_path.is_dir():
            destination_file = destination_path / source_path.name
        else:
            destination_file = destination_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_file)
        return destination_file

    def upload_directory(
        self,
        local_dir: Path,
        remote_path: Path,
        *,
        archive_format: str = "zip",
        remote_prefix: Optional[Path] = None,
    ) -> Path:
        if not local_dir.is_dir():
            raise ValueError(f"Expected directory to upload, got {local_dir}")

        remote_base = self.base_path
        if remote_prefix:
            remote_base = remote_base / remote_prefix

        remote_base.mkdir(parents=True, exist_ok=True)

        destination_dir = remote_base / remote_path.parent
        destination_dir.mkdir(parents=True, exist_ok=True)

        if archive_format:
            archive_name = f"{remote_path.name}.{archive_format}"
            archive_path = destination_dir / archive_name
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_base = Path(tmpdir) / local_dir.name
                shutil.copytree(local_dir, tmp_base)
                shutil.make_archive(
                    base_name=str(archive_path.with_suffix("")),
                    format=archive_format,
                    root_dir=tmp_base.parent,
                    base_dir=tmp_base.name,
                )
            return archive_path

        target_dir = destination_dir / remote_path.name
        shutil.copytree(local_dir, target_dir, dirs_exist_ok=True)
        return target_dir


class GoogleDriveStorageProvider(StorageProvider):
    """Provider implementation backed by Google Drive."""

    DOWNLOAD_URL = "https://drive.google.com/uc?export=download"

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials)
        self._drive_service = None

    # Download ---------------------------------------------------------------
    def download_artifact(
        self,
        artifact: AdapterArtifact,
        target_root: Path,
    ) -> Path:
        destination_path = (target_root / artifact.destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact.unpack:
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / f"{artifact.name}.zip"
                self._download_google_drive_file(artifact.source, archive_path)
                _extract_archive(archive_path, destination_path)
            return destination_path

        self._download_google_drive_file(artifact.source, destination_path)
        return destination_path

    # Upload ---------------------------------------------------------------
    def upload_directory(
        self,
        local_dir: Path,
        remote_path: Path,
        *,
        archive_format: str = "zip",
        remote_prefix: Optional[Path] = None,
    ) -> Path:
        service = self._ensure_drive_service()
        parent_folder_id = self.credentials.get("parent_folder_id")
        if not parent_folder_id:
            raise ValueError(
                "Google Drive upload requires 'parent_folder_id' in credentials"
            )

        target_parts: Iterable[str]
        if remote_prefix:
            target_parts = (*remote_prefix.parts, *remote_path.parts[:-1])
        else:
            target_parts = remote_path.parts[:-1]

        parent_id = parent_folder_id
        for part in target_parts:
            parent_id = self._ensure_folder(service, parent_id, part)

        file_name = remote_path.name
        if archive_format:
            file_name = f"{file_name}.{archive_format}"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_base = Path(tmpdir) / local_dir.name
                shutil.copytree(local_dir, tmp_base)
                archive_path = Path(tmpdir) / file_name
                shutil.make_archive(
                    base_name=str(archive_path.with_suffix("")),
                    format=archive_format,
                    root_dir=tmp_base.parent,
                    base_dir=tmp_base.name,
                )
                media_path = archive_path
        else:
            raise ValueError("Google Drive upload currently requires an archive format")

        from googleapiclient.http import MediaFileUpload  # type: ignore

        file_metadata = {"name": file_name, "parents": [parent_id]}
        media = MediaFileUpload(str(media_path), resumable=True)

        existing_id = self._find_existing_file(service, parent_id, file_name)
        if existing_id:
            service.files().update(fileId=existing_id, media_body=media).execute()
            return Path(file_name)

        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        return Path(file_name)

    # Helper methods -------------------------------------------------------
    def _download_google_drive_file(self, file_id: str, destination: Path) -> None:
        session = requests.Session()
        response = session.get(self.DOWNLOAD_URL, params={"id": file_id}, stream=True)
        token = _extract_confirm_token(response.headers)
        if token:
            response = session.get(
                self.DOWNLOAD_URL,
                params={"id": file_id, "confirm": token},
                stream=True,
            )
        _save_response_content(response, destination)

    def _ensure_drive_service(self):
        if self._drive_service is not None:
            return self._drive_service

        try:
            from google.oauth2.service_account import Credentials  # type: ignore
            from googleapiclient.discovery import build  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-api-python-client and google-auth are required for "
                "Google Drive uploads"
            ) from exc

        scopes = self.credentials.get(
            "scopes",
            ["https://www.googleapis.com/auth/drive.file"],
        )

        if "service_account_file" in self.credentials:
            creds = Credentials.from_service_account_file(
                self.credentials["service_account_file"], scopes=scopes
            )
        elif "service_account_info" in self.credentials:
            creds = Credentials.from_service_account_info(
                self.credentials["service_account_info"], scopes=scopes
            )
        else:
            raise ValueError(
                "Google Drive credentials must include 'service_account_file' "
                "or 'service_account_info'"
            )

        self._drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._drive_service

    def _ensure_folder(self, service, parent_id: str, name: str) -> str:
        query = (
            f"'{parent_id}' in parents and name = '{name}' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
            .execute()
        )
        files = response.get("files", [])
        if files:
            return files[0]["id"]

        file_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = service.files().create(body=file_metadata, fields="id").execute()
        return folder["id"]

    def _find_existing_file(self, service, parent_id: str, name: str) -> Optional[str]:
        query = (
            f"'{parent_id}' in parents and name = '{name}' and trashed = false"
        )
        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
            .execute()
        )
        files = response.get("files", [])
        if files:
            return files[0]["id"]
        return None


PROVIDER_REGISTRY = {
    "local": LocalStorageProvider,
    "google_drive": GoogleDriveStorageProvider,
}


def get_storage_provider(name: str, credentials: Optional[Dict[str, Any]] = None) -> StorageProvider:
    try:
        provider_cls = PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unsupported storage provider: {name}") from exc
    return provider_cls(credentials)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_adapter_sync_config(config_path: Path) -> AdapterSyncConfig:
    data = _load_structured_file(config_path)
    if "provider" not in data and any(isinstance(v, dict) for v in data.values()):
        raise ValueError(
            "Top-level config must contain 'provider' unless loaded via section helper"
        )
    provider = data.get("provider")
    if not provider:
        raise ValueError("Config missing 'provider'")

    credentials = data.get("credentials", {})
    artifacts_data = data.get("artifacts", [])
    artifacts = [
        AdapterArtifact(
            name=item["name"],
            source=item["source"],
            destination=item["destination"],
            unpack=bool(item.get("unpack", False)),
        )
        for item in artifacts_data
    ]

    return AdapterSyncConfig(provider=provider, credentials=credentials, artifacts=artifacts)


def _load_structured_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML is required to parse YAML configuration files") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Download driver
# ---------------------------------------------------------------------------


def download_adapters_from_config(
    config_path: Path,
    *,
    target_root: Optional[Path] = None,
) -> List[Path]:
    """Download all artifacts defined in *config_path*.

    Parameters
    ----------
    config_path:
        Path to the JSON/YAML configuration file.
    target_root:
        Optional override of the adapters root. Defaults to the repository's
        canonical adapters directory.
    """

    config = load_adapter_sync_config(config_path)
    return download_adapters(config, target_root=target_root)


def download_adapters(
    config: AdapterSyncConfig,
    *,
    target_root: Optional[Path] = None,
) -> List[Path]:
    if target_root is None:
        target_root = DEFAULT_ADAPTERS_ROOT
    target_root.mkdir(parents=True, exist_ok=True)

    provider = get_storage_provider(config.provider, config.credentials)

    materialised: List[Path] = []
    for artifact in config.artifacts:
        path = provider.download_artifact(artifact, target_root)
        materialised.append(path)
    return materialised


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _extract_confirm_token(headers: Dict[str, str]) -> Optional[str]:
    for key, value in headers.items():
        if key.lower().startswith("set-cookie") and "download_warning" in value:
            parts = value.split(";")
            for part in parts:
                if part.strip().startswith("download_warning"):
                    _, token = part.split("=", 1)
                    return token
    return None


def _save_response_content(response, destination: Path, chunk_size: int = 32 * 1024) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download artifact (status={response.status_code}): {response.text[:200]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)


def _extract_archive(archive_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


__all__ = [
    "AdapterArtifact",
    "AdapterSyncConfig",
    "DEFAULT_ADAPTERS_ROOT",
    "download_adapters",
    "download_adapters_from_config",
    "get_storage_provider",
    "load_adapter_sync_config",
    "StorageProvider",
]
