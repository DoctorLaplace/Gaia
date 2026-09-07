import os
import shutil

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corrupted_txt = os.path.join(project_root, "data", "eagle", "corrupted_files.txt")
    source_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")
    repair_dir = os.path.join(project_root, "data", "eagle", "repair")
    
    if not os.path.exists(corrupted_txt):
        print(f"[!] corrupted_files.txt not found at {corrupted_txt}")
        print("Please copy the corrupted_files.txt from the supercomputer to your local data/eagle/ folder first.")
        return
        
    if not os.path.exists(source_dir):
        print(f"[!] Source directory not found: {source_dir}")
        return
        
    # Read files to repair
    with open(corrupted_txt, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    if not filenames:
        print("[*] No files listed in corrupted_files.txt.")
        return
        
    print(f"[*] Preparing {len(filenames)} files for repair...")
    os.makedirs(repair_dir, exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    for name in filenames:
        src_file = os.path.join(source_dir, name)
        dst_file = os.path.join(repair_dir, name)
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            print(f"  [!] Missing locally: {name}")
            missing_count += 1
            
    print(f"\n[OK] Copy complete!")
    print(f"  - Successfully copied to repair/: {copied_count} files")
    if missing_count > 0:
        print(f"  - Missing locally: {missing_count} files")
    print(f"[*] You can now transfer/upload only the files inside: {repair_dir}")

if __name__ == "__main__":
    main()
