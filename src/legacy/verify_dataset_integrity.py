import os
import rasterio
from tqdm import tqdm

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tif_dir = os.path.join(project_root, "data", "eagle", "30m_10nm")
    
    # Check if we are on supercomputer and use absolute path
    alt_dir = "/home/jovyan/Gaia/data/eagle/30m_10nm"
    if os.path.exists(alt_dir):
        tif_dir = alt_dir
        
    print(f"[*] Scanning files in: {tif_dir}")
    if not os.path.exists(tif_dir):
        print("[!] Directory not found!")
        return
        
    files = sorted([os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith('.tif')])
    print(f"[*] Found {len(files)} TIFF files.")
    
    corrupted = []
    for f in tqdm(files, desc="Verifying Integrity"):
        try:
            with rasterio.open(f) as src:
                # Force GDAL to read file headers and properties
                _ = src.bounds
                _ = src.crs
                _ = src.transform
        except Exception as e:
            corrupted.append((os.path.basename(f), os.path.getsize(f), str(e)))
            
    print(f"\n[*] Scan complete. Found {len(corrupted)} corrupted files out of {len(files)}.")
    if corrupted:
        print("\n[!] List of corrupted files:")
        for name, size, err in corrupted[:30]:
            print(f"  - {name} ({size/1024/1024:.2f} MB) | Error: {err}")
        if len(corrupted) > 30:
            print(f"  ... and {len(corrupted) - 30} more files.")
            
        # Save corrupted filenames to corrupted_files.txt
        out_txt = os.path.join(os.path.dirname(tif_dir), "corrupted_files.txt")
        try:
            with open(out_txt, "w", encoding="utf-8") as f_out:
                for name, _, _ in corrupted:
                    f_out.write(name + "\n")
            print(f"\n[OK] Saved list of corrupted filenames to: {out_txt}")
        except Exception as e:
            print(f"[!] Failed to save list of corrupted files: {e}")
    else:
        print("[OK] All files are healthy and valid!")

if __name__ == "__main__":
    main()
