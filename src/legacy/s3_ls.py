import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.s3_utils import get_s3_fs

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # Ensure path starts with s3:// for fsspec
    if path and not path.startswith("s3://"):
        path = f"s3://{path}"
        
    try:
        fs = get_s3_fs()
        print(f"[*] Connecting to S3 Endpoint: {fs.storage_options.get('client_kwargs', {}).get('endpoint_url')}")
        
        if not path or path == "s3://":
            # List buckets
            buckets = fs.ls("")
            print("\nAvailable Buckets:")
            for b in buckets:
                print(f"  - {b}")
        else:
            # List contents
            print(f"[*] Listing: {path}")
            items = fs.ls(path, detail=True)
            
            print(f"\n{'NAME':<50} {'SIZE':<12} {'TYPE':<10}")
            print("-" * 72)
            
            total_size_bytes = 0
            file_count = 0
            
            for item in items:
                name = os.path.basename(item['Key'])
                if item['type'] == 'file':
                    total_size_bytes += item['Size']
                    file_count += 1
                    size = f"{item['Size'] / 1024 / 1024:.2f} MB"
                else:
                    size = "-"
                print(f"{name:<50} {size:<12} {item['type']:<10}")
            
            if file_count > 0:
                print("-" * 72)
                total_mb = total_size_bytes / 1024 / 1024
                total_gb = total_mb / 1024
                
                if total_gb >= 1.0:
                    size_str = f"{total_gb:.2f} GB"
                else:
                    size_str = f"{total_mb:.2f} MB"
                    
                print(f"{'TOTAL':<50} {size_str:<12} {file_count} files")
                
    except Exception as e:
        print(f"[!] S3 Error: {e}")
        print("\nTip: Ensure your credentials are in configs/config.yaml or ~/.s3cfg")

if __name__ == "__main__":
    main()
