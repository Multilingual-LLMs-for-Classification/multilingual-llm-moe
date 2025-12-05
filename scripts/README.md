# Sync Scripts

This directory hosts thin wrappers around the utility helpers in
`src/utils`. They read the sync configuration files under `config/` and invoke
the appropriate download or upload pipeline.

## Available wrappers

- `download_adapters.sh` – fetch adapter artifacts based on
  `config/adapter_storage.json` (or a custom path passed as the first
  argument).
- `upload_adapters.sh` – export adapters according to the same configuration
  schema.
- `download_datasets.sh` – populate the local `data/` directory from
  `config/dataset_storage.json`.
- `upload_datasets.sh` – package datasets and push them to remote storage.

Each script accepts an optional path to a configuration file. Example:

```bash
# Use the default config
bash scripts/download_adapters.sh

# Supply a custom configuration
bash scripts/upload_datasets.sh path/to/custom_dataset_config.json
```

The scripts simply forward the configuration to the utilities—any failures or
status messages will come from the underlying provider implementation.

## Provider-specific helpers

When using the `shell` provider in configuration files, the sync logic delegates
the actual transfer work to helper scripts located in `scripts/providers/`.
These helpers are intentionally minimal and should be customised to match the
storage backend you use.

### Google Drive via gdrive CLI

1. Install and authenticate the CLI:

   ```bash
   wget -O gdrive https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive-linux-x64
   chmod +x gdrive
   mv gdrive ~/.local/bin/
   gdrive about  # completes OAuth in the browser
   ```

2. Update `scripts/providers/upload_dataset.sh` (and any other helper) to call
   `gdrive upload` with the desired folder ID. Example snippet:

   ```bash
   PARENT_ID="15hldwt3XwGZmyYwF8F-xHCVkEYUzKGxi"
   gdrive upload --name "${REMOTE_PATH}.zip" -p "${PARENT_ID}" "${ARCHIVE_PATH}"
   ```

3. Adjust `config/dataset_storage.json` or `config/adapter_storage.json` to pass
   any extra parameters (folder IDs, etc.) via placeholders or environment
   variables.

### Google Drive via rclone

1. Install `rclone` and run `rclone config` to create a `gdrive` remote.
2. In the helper script, replace the placeholder copy with:

   ```bash
   rclone copy "${ARCHIVE_PATH}" "gdrive:your-folder/${REMOTE_PATH}.zip"
   ```

### AWS S3 (or S3-compatible storage)

1. Configure credentials with `aws configure` (or environment variables).
2. Update the helper to run:

   ```bash
   aws s3 cp "${ARCHIVE_PATH}" "s3://your-bucket/${REMOTE_PATH}.zip"
   ```

### Azure Blob Storage

1. Install the Azure CLI (`az`) and authenticate.
2. Use `az storage blob upload` inside the helper, e.g.:

   ```bash
   az storage blob upload --file "${ARCHIVE_PATH}" \
     --container-name my-container --name "${REMOTE_PATH}.zip"
   ```

### Custom providers

The shell helper can run any command, including FTP clients or bespoke REST
upload scripts. Expand the config placeholders (`{source_dir}`, `{archive_path}`
etc.) or read environment variables to pass additional information.

## Testing changes

After editing a provider helper:

1. Ensure the required CLI is on `PATH` and authenticated.
2. Run the relevant wrapper script (e.g. `bash scripts/upload_datasets.sh`).
3. Verify the command exits with status 0 and the remote service receives the
   artifact. Any non-zero exit code will surface in the script output.

Remember to document new dependencies or environment variables so other users
can replicate your setup.

