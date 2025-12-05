"""Utilities for pulling QLoRA adapter artifacts from remote storage.

This module implements a configurable download pipeline that can ingest adapter
artifacts stored on external providers (e.g., Google Drive via the ``gdrive``
CLI, local mirrors) and materialise them inside the repository's
``adapters/`` directory structure.

The downloader is driven by a declarative configuration file so that
open-source consumers can mount their own storage endpoints without modifying
code.  A minimal JSON configuration using the gdrive CLI looks like:

```
{
  "provider": "gdrive_cli",
  "credentials": {
    "binary": "gdrive",
    "parent_folder_id": "1AbCdEf..."  // optional, used for uploads
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
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


class GoogleDriveCLIStorageProvider(StorageProvider):
    """Storage provider backed by the `gdrive` command-line client.

    Authentication is delegated to the user.  Run ``gdrive about`` once the
    CLI is installed to complete the OAuth flow and cache credentials locally.
    """

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials)
        binary = self.credentials.get("binary", "gdrive")
        resolved = shutil.which(str(binary))
        if not resolved:
            raise FileNotFoundError(
                f"gdrive CLI not found: expected '{binary}'. Install it from "
                "https://github.com/prasmussen/gdrive or adjust the 'binary' "
                "credential."
            )
        self.binary = resolved
        self.download_args = list(self.credentials.get("download_args", []))
        self.upload_args = list(self.credentials.get("upload_args", []))
        self.parent_folder_id = self.credentials.get("parent_folder_id")

    # Download ---------------------------------------------------------------
    def download_artifact(
        self,
        artifact: AdapterArtifact,
        target_root: Path,
    ) -> Path:
        destination_path = (target_root / artifact.destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            before = set(tmpdir.iterdir()) if tmpdir.exists() else set()
            cmd = [self.binary, "download", "--path", str(tmpdir)]
            cmd.extend(self.download_args)
            cmd.append(artifact.source)
            self._run_cli(cmd, "download")

            downloaded = [p for p in tmpdir.iterdir() if p not in before]
            if not downloaded:
                raise RuntimeError(
                    f"gdrive download for artifact '{artifact.name}' produced no files"
                )
            downloaded.sort(key=lambda p: p.stat().st_mtime)
            materialised = downloaded[-1]

            if artifact.unpack:
                _extract_archive(materialised, destination_path)
                return destination_path

            if materialised.is_dir():
                shutil.copytree(materialised, destination_path, dirs_exist_ok=True)
                return destination_path

            if destination_path.exists() and destination_path.is_dir():
                target_file = destination_path / materialised.name
            else:
                target_file = destination_path
            shutil.copy2(materialised, target_file)
            return target_file

    # Upload ---------------------------------------------------------------
    def upload_directory(
        self,
        local_dir: Path,
        remote_path: Path,
        *,
        archive_format: str = "zip",
        remote_prefix: Optional[Path] = None,
    ) -> Path:
        if archive_format is None:
            raise ValueError("gdrive uploads require an archive format (zip recommended)")

        if not self.parent_folder_id:
            raise ValueError(
                "gdrive upload requires 'parent_folder_id' in credentials"
            )

        if not local_dir.is_dir():
            raise ValueError(f"Expected directory to upload, got {local_dir}")

        archive_stem = "__".join(
            [*remote_prefix.parts] if remote_prefix else []
            + list(remote_path.parts)
        ) or remote_path.name

        archive_name = f"{archive_stem}.{archive_format}"

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            tmp_base = tmpdir / local_dir.name
            shutil.copytree(local_dir, tmp_base)
            archive_path = tmpdir / archive_name
            shutil.make_archive(
                base_name=str(archive_path.with_suffix("")),
                format=archive_format,
                root_dir=tmp_base.parent,
                base_dir=tmp_base.name,
            )

            cmd = [self.binary, "upload", "--name", archive_name, "-p", self.parent_folder_id]
            cmd.extend(self.upload_args)
            cmd.append(str(archive_path))
            self._run_cli(cmd, "upload")

        return Path(archive_name)

    # Helpers ---------------------------------------------------------------
    def _run_cli(self, cmd: List[str], action: str) -> None:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on CLI availability
            stdout = exc.stdout.decode("utf-8", errors="ignore") if exc.stdout else ""
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            raise RuntimeError(
                f"gdrive {action} command failed (exit {exc.returncode}).\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            ) from exc



class ShellCommandStorageProvider(StorageProvider):
    """Delegate downloads/uploads to user-defined shell commands.

    Provide commands via credentials:

    - ``download_command``: string or list executed for downloads.
    - ``upload_command``: string or list executed for uploads.

    Placeholders (expanded using ``str.format``):

    Download → ``{source}``, ``{name}``, ``{destination}``, ``{destination_dir}``, ``{tmp_dir}``
    Upload → ``{source_dir}``, ``{archive_path}``, ``{remote_path}``, ``{remote_name}``,
    ``{tmp_dir}``, ``{archive_format}``
    """

    def __init__(self, credentials: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(credentials)
        self.download_command = self._normalise_command(
            self.credentials.get("download_command")
        )
        self.upload_command = self._normalise_command(
            self.credentials.get("upload_command")
        )
        self._placeholder_values = {
            key: value
            for key, value in (self.credentials or {}).items()
            if isinstance(value, (str, int, float, bool))
        }

    def download_artifact(
        self,
        artifact: AdapterArtifact,
        target_root: Path,
    ) -> Path:
        if not self.download_command:
            raise NotImplementedError(
                "Shell provider requires 'download_command' for downloads"
            )

        destination_path = (target_root / artifact.destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            command = self._expand_command(
                self.download_command,
                source=artifact.source,
                name=artifact.name,
                destination=str(destination_path),
                destination_dir=str(destination_path.parent),
                tmp_dir=str(tmpdir),
            )
            self._run_cli(command, "download")

            produced = self._determine_output(destination_path, tmpdir)
            if artifact.unpack:
                if produced.resolve() == destination_path.resolve():
                    return destination_path
                if produced.is_dir():
                    shutil.copytree(produced, destination_path, dirs_exist_ok=True)
                    return destination_path
                _extract_archive(produced, destination_path)
                return destination_path

            if produced == destination_path:
                return destination_path
            if produced.is_dir():
                shutil.copytree(produced, destination_path, dirs_exist_ok=True)
                return destination_path

            final_path = destination_path
            if destination_path.exists() and destination_path.is_dir():
                final_path = destination_path / produced.name
            shutil.copy2(produced, final_path)
            return final_path

    def upload_directory(
        self,
        local_dir: Path,
        remote_path: Path,
        *,
        archive_format: str = "zip",
        remote_prefix: Optional[Path] = None,
    ) -> Path:
        if not self.upload_command:
            raise NotImplementedError(
                "Shell provider requires 'upload_command' for uploads"
            )
        if not local_dir.is_dir():
            raise ValueError(f"Expected directory to upload, got {local_dir}")

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            archive_name = remote_path.name
            if remote_prefix:
                archive_name = "__".join((*remote_prefix.parts, remote_path.name))
            archive_name = (
                f"{archive_name}.{archive_format}"
                if archive_format
                else archive_name
            )

            archive_path = tmpdir / archive_name
            if archive_format:
                shutil.make_archive(
                    base_name=str(archive_path.with_suffix("")),
                    format=archive_format,
                    root_dir=local_dir.parent,
                    base_dir=local_dir.name,
                )
            else:
                shutil.copytree(local_dir, archive_path)

            command = self._expand_command(
                self.upload_command,
                source_dir=str(local_dir),
                archive_path=str(archive_path),
                remote_path=str(remote_path),
                remote_name=remote_path.name,
                tmp_dir=str(tmpdir),
                archive_format=archive_format,
            )
            self._run_cli(command, "upload")

        return Path(archive_name)

    @staticmethod
    def _normalise_command(command: Optional[Any]) -> Optional[List[str]]:
        if command is None:
            return None
        if isinstance(command, str):
            return shlex.split(command)
        if isinstance(command, (list, tuple)):
            return [str(part) for part in command]
        raise TypeError("Shell provider commands must be strings or sequences of strings")

    def _expand_command(self, command: List[str], **kwargs: Any) -> List[str]:
        context = {**self._placeholder_values, **kwargs}
        return [part.format(**context) for part in command]

    @staticmethod
    def _run_cli(command: List[str], action: str) -> None:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on external tool
            raise RuntimeError(
                f"Shell provider {action} command failed with exit code {exc.returncode}."
            ) from exc

    @staticmethod
    def _determine_output(destination_path: Path, tmpdir: Path) -> Path:
        if destination_path.exists():
            return destination_path
        if not tmpdir.exists():
            raise RuntimeError("Shell download command produced no output")
        candidates = list(tmpdir.iterdir())
        if not candidates:
            raise RuntimeError("Shell download command produced no files")
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]


PROVIDER_REGISTRY = {
    "local": LocalStorageProvider,
    "gdrive_cli": GoogleDriveCLIStorageProvider,
    "shell": ShellCommandStorageProvider,
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
