from src.s3_utils import get_s3_fs
import os
import sys

def check_size(path):
    print(f"[*] Connecting to S3 to check size of {path} ...")
    try:
        fs = get_s3_fs()
        files = fs.ls(path)
        total_size_bytes = sum(f.get('size', 0) for f in files)
        total_size_gb = total_size_bytes / (1024**3)
        print(f"[OK] Total size of {path} is {total_size_gb:.2f} GB ({len(files)} files)")
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    check_size("s3://gaia-datasets/30m/")
    check_size("s3://gaia-datasets/5m/")
