# Utilities

The `src/utils` package contains tooling for synchronising QLoRA adapter
artifacts between the local repository and remote storage providers.

## Modules

- `experts/adapters/download_adapters.py` – Loads adapter sync configurations
  (JSON/YAML) and downloads artifacts from supported providers (currently
  Google Drive and local directories).  A declarative config allows
  contributors to mount their own storage locations without code changes.
- `experts/adapters/upload_adapters.py` – Discovers local adapters, builds
  upload configs, and exports adapter directories to the configured provider.
  Reuses the same provider registry defined in the download module.
- `__init__.py` – Convenience exports for the public utility surface.

## Configuration

Adapter sync behaviour is driven by `config/adapter_storage.json` at the
repository root.  The file specifies the storage provider, credentials, and
artifact mappings.  Example (Google Drive):

```json
{
  "provider": "google_drive",
  "credentials": {
    "service_account_file": "./service-account.json",
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
```

To download or upload adapters, use the shell wrappers in `scripts/`:

```bash
bash scripts/download_adapters.sh          # uses config/adapter_storage.json
bash scripts/upload_adapters.sh            # uploads using the same config
bash scripts/download_adapters.sh custom.json
```

For local mirrors, switch `provider` to `local` and set `credentials.base_path`
to the external directory containing the adapter archives.
