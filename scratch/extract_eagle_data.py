import os
import zipfile
import shutil
from tqdm import tqdm

def main():
    base_dir = r"E:\Git Repositories\Laboratory\Models\Gaia\data\eagle"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Copy the RGB TIFF
    rgb_src = r"C:\Users\silve\Downloads\ANG_V02_L3_RFL_Mosaic_EAGLE30m_centeredGaussian_FWHM30m_10nmFWHM_rgb.tif"
    rgb_dst = os.path.join(base_dir, os.path.basename(rgb_src))
    if os.path.exists(rgb_src):
        print("[*] Copying RGB TIFF...")
        shutil.copy2(rgb_src, rgb_dst)
        print(f"[OK] Copied RGB TIFF to {rgb_dst}")
    else:
        print(f"[!] RGB TIFF not found at {rgb_src}")

    # 2. Extract Zip 2 (v2 native) -> data/eagle/30m_native
    native_zip = r"C:\Users\silve\Downloads\ANG_V02_L3_RFL_Mosaic_EAGLE30m_centeredGaussian_FWHM30m_v2.zip"
    native_dst_dir = os.path.join(base_dir, "30m_native")
    os.makedirs(native_dst_dir, exist_ok=True)
    
    if os.path.exists(native_zip):
        print("[*] Extracting native mosaic tiles (Zip 2)...")
        with zipfile.ZipFile(native_zip) as z:
            members = z.infolist()
            # Extract each file directly to native_dst_dir
            for member in tqdm(members, desc="Native Tiles"):
                filename = os.path.basename(member.filename)
                if not filename: # Skip directories
                    continue
                source = z.open(member)
                target_path = os.path.join(native_dst_dir, filename)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        print(f"[OK] Extracted native tiles to {native_dst_dir}")
    else:
        print(f"[!] Native zip not found at {native_zip}")

    # 3. Extract Zip 1 (10nm FWHM) -> data/eagle/30m_10nm
    fwhm_zip = r"C:\Users\silve\Downloads\ANG_V02_L3_RFL_Mosaic_EAGLE30m_centeredGaussian_FWHM30m_10nmFWHM.zip"
    fwhm_dst_dir = os.path.join(base_dir, "30m_10nm")
    os.makedirs(fwhm_dst_dir, exist_ok=True)
    
    if os.path.exists(fwhm_zip):
        print("[*] Extracting 10nm FWHM mosaic tiles (Zip 1, 1517 files)...")
        with zipfile.ZipFile(fwhm_zip) as z:
            members = z.infolist()
            # Extract each file stripping any parent directory prefix
            for member in tqdm(members, desc="10nm FWHM Tiles"):
                filename = os.path.basename(member.filename)
                if not filename or not filename.endswith(".tif"):
                    continue
                source = z.open(member)
                target_path = os.path.join(fwhm_dst_dir, filename)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        print(f"[OK] Extracted 10nm FWHM tiles to {fwhm_dst_dir}")
    else:
        print(f"[!] FWHM zip not found at {fwhm_zip}")

if __name__ == "__main__":
    main()
