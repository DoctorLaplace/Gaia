import earthaccess
import os

def generate_full_batch():
    print("[*] Logging in to Earthdata...")
    earthaccess.login()
    
    print("[*] Searching for all BioSCape granules (Oct-Nov 2023)...")
    # Broad search for the collection
    results = earthaccess.search_data(
        short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
        temporal=("2023-10-01", "2023-11-30")
    )
    
    print(f"[✔] Found {len(results)} granules.")
    
    with open('mega_batch.txt', 'w') as f:
        for r in results:
            # We want the granule name (ID)
            # mega_download.py uses: results = earthaccess.search_data(..., granule_name=gid)
            # The granule ID in Earthdata usually looks like BioSCape_AVNG_L2B_BRDF_GCFR.ang20231022t092801_000_L2B_OE_0b4f48b4_RFL_ORT_tbg_v1.nc
            gid = r.info()['title']
            f.write(f"{gid}\n")
            
    print(f"[✔] Wrote {len(results)} entries to mega_batch.txt")

if __name__ == "__main__":
    generate_full_batch()
