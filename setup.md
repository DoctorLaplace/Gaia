# Gaia Setup & Credentials

Welcome to the Gaia project! To get the model training and data acquisition running, you'll need to set up your own credentials for **Earthdata** and **S3**.

## 1. Unified Credentials Setup
The project uses a clean, unified way to manage your secrets.

1.  Create a file at `src/credentials.py`. (This file is ignored by Git and will not be shared).
2.  Add the following template and fill in your keys:
    ```python
    # 1. NASA Earthdata
    EARTHDATA_USERNAME = "your_username"
    EARTHDATA_PASSWORD = "your_password"

    # 2. NRP Nautilus S3 (Ceph)
    S3_ACCESS_KEY = "your_access_key"
    S3_SECRET_KEY = "your_secret_key"
    S3_ENDPOINT = "https://s3-west.nrp-nautilus.io"
    ```

## 2. NRP Nautilus Access
If you are training on the supercomputer:
- Use `coder ssh nasa-gaia` to access your workspace.
- S3 access will automatically work if you have the `credentials.py` file or environment variables set.

## 3. Directory Structure
The repository includes tracked, empty directories for:
- `data/`: Hyperspectral granules and site labels.
- `checkpoints/`: Model weights (`.pth` files).
- `tmp/`: Temporary processing files.

These folders are tracked by Git but their contents are ignored to keep the repository size small.
