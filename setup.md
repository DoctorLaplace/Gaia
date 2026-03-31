# Gaia Setup & Credentials

Welcome to the Gaia project! To get the model training and data acquisition running, you'll need to set up your own credentials for **Earthdata** and **S3**.

## 1. Earthdata Credentials
The project uses `earthaccess` to download hyperspectral data from NASA. 

To automate your login:
1.  Create a file at `src/credentials.py`. (This file is ignored by Git and will not be shared).
2.  Add the following content:
    ```python
    EARTHDATA_USERNAME = "your_username"
    EARTHDATA_PASSWORD = "your_password"
    ```
The project scripts will automatically check this file for authentication.

## 2. S3 Storage Configuration
If you are syncing data to a Ceph S3 bucket (like in our production environment), you need to configure `s3cmd`.

1.  Install `s3cmd` (`pip install s3cmd`).
2.  Generate a configuration file (usually `~/.s3cfg` or specified in the code).
3.  The project defaults to checking `C:\Users\silve\AppData\Roaming\s3cmd.ini` for local runs, but you can override this in `src/smart_sync.py`.

## 3. Directory Structure
The repository includes empty directories for:
- `data/`: Where raw hyperspectral granules and site metadata are stored.
- `checkpoints/`: Where model weights (`.pth` files) are saved during training.
- `tmp/`: Temporary processing files.

These folders are tracked by Git but their contents are ignored to keep the repository size small.
