# Utilities

The `src/utils` package contains tooling for synchronising QLoRA adapter
artifacts between the local repository and remote storage providers.

## Modules

- `experts/adapters/download_adapters.py` – Loads adapter sync configurations
  (JSON/YAML) and downloads artifacts from supported providers. The registry
  includes a local filesystem provider, a [`gdrive` CLI](https://github.com/prasmussen/gdrive)
  helper, and a generic shell provider that executes user-defined commands.
- `experts/adapters/upload_adapters.py` – Discovers local adapters, builds
  upload configs, and exports adapter directories to the configured provider.
  Reuses the same provider registry defined in the download module.
- `datasets/download_datasets.py` – Reuses the adapter download plumbing to
  fetch dataset archives and extract them under the repository `data/` folder.
- `datasets/upload_datasets.py` – Mirrors the upload helpers so local datasets
  can be exported to remote storage when needed.
- `__init__.py` – Convenience exports for the public utility surface.

## Configuration

Adapter sync behaviour is driven by configuration files in the repository
`config/` directory.  Two templates are provided:

- `config/adapter_storage.json` – maps remote adapter artifacts to
  `src/models/experts/llms/adapters/`.
- `config/dataset_storage.json` – maps dataset archives to the repository
  `data/` folder.

Both follow the same schema. Example entries:

```json
{
  "provider": "gdrive_cli",
  "credentials": {
    "binary": "gdrive",
    "parent_folder_id": "1ExampleFolderId"
  },
  "artifacts": [
    {
      "name": "finance-news-mistral",
      "source": "1DriveFileId",
      "destination": "finance/news_classification/mistral",
      "unpack": true
    }
  ]
}

{
  "provider": "shell",
  "credentials": {
    "download_command": [
      "bash",
      "scripts/providers/download_adapter.sh",
      "{source}",
      "{destination_dir}",
      "{name}"
    ],
    "upload_command": [
      "bash",
      "scripts/providers/upload_adapter.sh",
      "{source_dir}",
      "{remote_path}",
      "{archive_path}"
    ]
  },
  "artifacts": [
    {
      "name": "finance-news-mistral",
      "source": "1DriveFileId",
      "destination": "finance/news_classification/mistral",
      "unpack": true
    }
  ]
}
```

To download or upload adapters, use the shell wrappers in `scripts/`:

```bash
bash scripts/download_adapters.sh          # uses config/adapter_storage.json
bash scripts/upload_adapters.sh            # uploads using the same config
bash scripts/download_adapters.sh custom.json
bash scripts/download_datasets.sh          # populates ./data from dataset config
bash scripts/upload_datasets.sh            # pushes ./data back to remote storage
```

For local mirrors, switch `provider` to `local` and set `credentials.base_path`
to the external directory containing the adapter archives.
