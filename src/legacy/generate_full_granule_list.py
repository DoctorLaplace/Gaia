"""
NASA DATA DISCOVERY: Find every available flightline for the BioSCape project.
This script generates a list of "Native IDs" which can be pasted into mega_batch.txt.
"""
import earthaccess
import os

def list_all_available_granules():
    print("\n" + "="*60)
    print("GAIA DATA DISCOVERY: Querying NASA Earthdata for BioSCape...")
    print("="*60)
    
    # Login (if needed)
    earthaccess.login(persist=True)
    
    # 1. Search for EVERYTHING in this collection
    print("[*] Contacting NASA CMR for 'BioSCape_AVNG_L2B_BRDF_GCFR_2385'...")
    results = earthaccess.search_data(
        short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
        count=5000 # BioSCape has ~3647 granules per my recent search
    )
    
    if not results:
        print("[!] No granules found. Is the short_name correct?")
        return
        
    print(f"[✔] Found {len(results)} total granules available on server.")
    
    # 2. Extract Native IDs
    native_ids = [g['meta']['native-id'] for g in results]
    native_ids = sorted(list(set(native_ids))) # Deduplicate and sort
    
    # 3. Save to full_inventory.txt
    output_path = "full_inventory.txt"
    with open(output_path, 'w') as f:
        for nid in native_ids:
            f.write(nid + "\n")
            
    print(f"\n[SUCCESS] Master Inventory saved to: {output_path}")
    print(f"To download everything, you can copy the contents of {output_path} into mega_batch.txt.")
    print("="*60)

if __name__ == "__main__":
    list_all_available_granules()
