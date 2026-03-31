import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.s3_utils import get_s3_fs

def s3_clear(path=None):
    if not path:
        path = "s3://gaia-datasets/"
    
    # Ensure path starts with s3:// for fsspec
    if not path.startswith("s3://"):
        path = f"s3://{path}"
    
    # Safety Check: Don't allow clearing "s3://" or "s3://" followed by nothing
    if len(path.strip("s3:/")) == 0:
        print("[!] Error: You cannot clear the entire S3 root. Please specify a bucket.")
        return

    try:
        fs = get_s3_fs()
        print(f"[*] Target Bucket/Path: {path}")
        print(f"[*] S3 Endpoint: {fs.storage_options.get('client_kwargs', {}).get('endpoint_url')}")
        
        # Double Confirmation
        print(f"\n[!] WARNING: This will DESTRUCTIVELY delete EVERYTHING in {path}")
        confirm = input(f"    Type 'YES' to confirm absolute destruction: ")
        
        if confirm == "YES":
            print(f"[*] Purging contents of {path}...")
            # To avoid deleting the bucket itself, we list and remove children
            try:
                items = fs.ls(path, detail=False)
                if not items:
                    print(f"[*] {path} is already empty.")
                else:
                    for item in items:
                        # item often doesn't have the protocol from fs.ls
                        full_item = f"s3://{item}" if not item.startswith("s3://") else item
                        print(f"  > Removing: {full_item}")
                        fs.rm(full_item, recursive=True)
                    print(f"[✔] S3 Clear Complete: {path} is now empty.")
            except Exception as e:
                # Fallback to direct rm if ls fails or if path is a file
                fs.rm(path, recursive=True)
                print(f"[✔] S3 Clear Complete (Direct RM).")
        else:
            print("[*] Aborted. No files were harmed.")
            
    except Exception as e:
        print(f"[!] S3 Error: {e}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    s3_clear(target)
