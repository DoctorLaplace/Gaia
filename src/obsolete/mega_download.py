import earthaccess
import os
import time

def mega_download():
    # 1. Login
    print("\n" + "="*50)
    print("GAIA MEGA-DOWNLOADER: 54GB BIOCAPE BATCH")
    print("="*50)
    print("[*] Accessing NASA Earthdata...")
    earthaccess.login(persist=True)
    
    # 2. Load IDs
    if not os.path.exists('mega_batch.txt'):
        print("[!] mega_batch.txt not found. Please run src/list_granules.py first.")
        return
        
    with open('mega_batch.txt', 'r') as f:
        batch_ids = [l.strip() for l in f if l.strip()]
        
    print(f"[*] Batch Sync: {len(batch_ids)} flightlines identified.")
    print("[*] Destination: data/bioscape/\n")
    
    # 3. Search and Download
    print(f"[*] Searching for {len(batch_ids)} specific flightlines in batches...")
    granules = []
    chunk_size = 10
    for i in range(0, len(batch_ids), chunk_size):
        chunk = batch_ids[i:i + chunk_size]
        print(f"    > Searching batch {i//chunk_size + 1} ({len(chunk)} IDs)...")
        try:
            batch_results = earthaccess.search_data(
                short_name="BioSCape_AVNG_L2B_BRDF_GCFR_2385",
                granule_name=chunk
            )
            granules.extend(batch_results)
        except Exception as e:
            print(f"    [!] Batch search failed: {e}")
            time.sleep(2) # Backoff
    
    print(f"[✔] Found {len(granules)} total matching granules on server.\n")

    # Get session for direct requests
    session = earthaccess.get_requests_https_session()

    for i, g in enumerate(granules):
        native_id = g['meta']['native-id']
        url = g.data_links()[0]
        filename = os.path.basename(url)
        dest_path = os.path.join("data/bioscape", filename)
        
        print(f"--- [{i+1}/{len(granules)}] SYNCING: {filename} ---")
        
        # Check if already exists and is complete (approx > 1GB)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000 * 1024 * 1024:
            print(f"[✔] Found existing complete file. Skipping.")
            continue
            
        print(f"[*] URL: {url}")
        print(f"[*] Status: DOWNLOADING (Visible Progress Every 10%)...")
        
        start_time = time.time()
        try:
            with session.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                last_print = 0
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024*10): # 10MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                            if percent - last_print >= 10:
                                print(f"  > Progress: {percent:.0f}% ({downloaded/(1024**3):.1f} GB / {total_size/(1024**3):.1f} GB)")
                                last_print = percent
            
            duration = time.time() - start_time
            print(f"[✔] SUCCESS: {filename} acquired in {duration/60:.1f} minutes.\n")
        except Exception as e:
            print(f"[!] FAILED: {filename} | Error: {e}\n")
            if os.path.exists(dest_path) and os.path.getsize(dest_path) < 100 * 1024 * 1024:
                os.remove(dest_path) # Clean up tiny/broken files
            
    print("="*50)
    print("MEGA-DOWNLOAD COMPLETE.")
    print("Run 'python src/train_production.py' to start training.")
    print("="*50)

if __name__ == "__main__":
    mega_download()
