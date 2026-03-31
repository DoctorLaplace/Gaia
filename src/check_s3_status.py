import subprocess
import os
import sys

# Import unified credentials from src/credentials.py (local only)
try:
    from .credentials import S3_ACCESS_KEY, S3_SECRET_KEY
except ImportError:
    try:
        from credentials import S3_ACCESS_KEY, S3_SECRET_KEY
    except ImportError:
        S3_ACCESS_KEY = S3_SECRET_KEY = None

# --- PATHS ---
S3CMD_PATH = r"C:\Users\silve\AppData\Roaming\Python\Python312\Scripts\s3cmd"
CONFIG_PATH = r"C:\Users\silve\AppData\Roaming\s3cmd.ini"

# --- ANSI COLORS ---
PURPLE = "\033[38;5;99m"
GOLD = "\033[38;5;220m"
RESET = "\033[0m"
BOLD = "\033[1m"

def check_s3():
    print(f"\n{BOLD}{PURPLE}--- S3 BUCKET AUDIT: s3://gaia-datasets ---{RESET}")
    
    try:
        # Ensure S3 credentials from credentials.py are set for the subprocess
        s3_env = os.environ.copy()
        if S3_ACCESS_KEY: s3_env['AWS_ACCESS_KEY_ID'] = S3_ACCESS_KEY
        if S3_SECRET_KEY: s3_env['AWS_SECRET_ACCESS_KEY'] = S3_SECRET_KEY

        # 1. Get Count (ls)
        ls_cmd = ['python', S3CMD_PATH, '-c', CONFIG_PATH, '--no-preserve', '--no-progress', '--quiet', 'ls', 's3://gaia-datasets/']
        result_ls = subprocess.run(ls_cmd, stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL, env=s3_env)
        
        files = [line for line in result_ls.stdout.splitlines() if line.strip()]
        count = len(files)
        
        # 2. Get Size (du)
        du_cmd = ['python', S3CMD_PATH, '-c', CONFIG_PATH, '--no-preserve', '--no-progress', '--quiet', 'du', 's3://gaia-datasets/']
        result_du = subprocess.run(du_cmd, stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL, env=s3_env)
        
        # Parse bytes from: "6736252904       5 objects s3://gaia-datasets/"
        total_bytes = 0
        if result_du.stdout:
            parts = result_du.stdout.split()
            if parts:
                total_bytes = int(parts[0])
        
        gb_size = total_bytes / (1024**3)
        
        print(f"\n  {GOLD}Total Granules:{RESET} {count}")
        print(f"  {GOLD}Total Volume:  {RESET} {gb_size:.8f} GB ({total_bytes} bytes)")
        
        if count > 0:
            print(f"\n{PURPLE}Latest Granules:{RESET}")
            for line in files[-5:]: # Show last 5
                parts = line.split()
                if len(parts) >= 4:
                    print(f"  - {os.path.basename(parts[3])}")
        
        print(f"\n{BOLD}{PURPLE}Audit Complete.{RESET}\n")
        
    except Exception as e:
        print(f"\n[!] Error auditing bucket: {e}")

if __name__ == "__main__":
    check_s3()
