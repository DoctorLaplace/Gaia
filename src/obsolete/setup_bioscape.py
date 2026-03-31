import earthaccess
import os

def setup_bioscape():
    """
    Automates the acquisition of BioSCape AVIRIS-NG imagery and BioSoundSCape richness labels.
    """
    print("--- Gaia BioSCape Data Setup ---")
    
    # Automating login with provided credentials
    try:
        auth = earthaccess.login(strategy="interactive", persist=True)
    except:
        # Fallback if persist/interactive fails in this environment
        auth = earthaccess.login(strategy="environment") 
        
    # Standard login doesn't take user/pass directly in the call easily without env vars
    # We will set them as environment variables temporarily for the session
    os.environ['EARTHDATA_USERNAME'] = "valifor"
    os.environ['EARTHDATA_PASSWORD'] = "Earthdataskulblaka1!"
    auth = earthaccess.login(strategy="environment")

    # Ensure data_dir is always in the project root/data, not src/data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "bioscape")
    os.makedirs(data_dir, exist_ok=True)
    
    def download_if_missing(granules, dest):
        # Helper to skip 2.7GB downloads if already present
        for g in granules:
            fname = g['umm']['DataGranule']['ArchiveAndDistributionInformation'][0]['Name']
            if os.path.exists(os.path.join(dest, fname)):
                print(f"[!] Skipping {fname} (Already exists)")
            else:
                earthaccess.download(g, dest)
    
    # 1. Acquire BioSoundSCape Species Richness Metadata
    print("\n[*] Resolving BioSoundSCape Collection...")
    # Collection: BioSCape: BioSoundSCape Acoustic Recordings, South Africa, 2023
    acoustic_collections = earthaccess.search_datasets(keyword="BioSoundSCape Acoustic")
    
    if acoustic_collections:
        c_id = acoustic_collections[0]['meta']['concept-id']
        print(f"[*] Found Collection: {c_id}")
        print("[*] Searching specifically for 'BioSoundSCape_site_metadata.csv'...")
        
        # Searching by filename is much more efficient
        acoustic_results = earthaccess.search_data(
            concept_id=c_id,
            filename="BioSoundSCape_site_metadata.csv"
        )
        
        if acoustic_results:
            print(f"[*] Found metadata CSV. Checking...")
            download_if_missing(acoustic_results, data_dir)
        else:
            print("[!] Warning: Could not find 'BioSoundSCape_site_metadata.csv' in collection.")
    else:
        print("[!] Error: Could not find BioSoundSCape collection by title.")

    # 2. Acquire a sample AVIRIS-NG L2B Flightline
    print("\n[*] Resolving AVIRIS-NG L2B Collection...")
    aviris_collections = earthaccess.search_datasets(keyword="BioSCape AVIRIS-NG L2B")
    
    if aviris_collections:
        c_id = aviris_collections[0]['meta']['concept-id']
        print(f"[*] Found Collection: {c_id}")
        aviris_results = earthaccess.search_data(
            concept_id=c_id,
            count=1
        )
        if aviris_results:
            print(f"[*] Found granule. Checking...")
            download_if_missing(aviris_results, data_dir)
        else:
            print("[!] Warning: No granules found in AVIRIS-NG collection.")
    else:
        print("[!] Error: Could not find AVIRIS-NG collection by title.")
    
    if aviris_results:
        print(f"[*] Found {len(aviris_results)} granules. Downloading one sample (will take some time)...")
        earthaccess.download(aviris_results[0], data_dir)
    else:
        print("[!] Warning: AVIRIS-NG BioSCape granules not found.")

    print("\n[✔] Setup Script Complete. Files will be in data/bioscape/")

if __name__ == "__main__":
    setup_bioscape()
