import earthaccess
import os
import time
import subprocess

def sync_targeted():
    print('\n' + '='*50)
    print('GAIA TARGETED SYNC: OCTOBER 2023')
    print('='*50)
    
    print('[*] Logging in to Earthdata...')
    try:
        earthaccess.login()
    except Exception as e:
        print(f'[!] Login failed: {e}')
        return
    
    # Target dates from richness CSV: Oct 13, 25, 26
    target_dates = [
        ('2023-10-10', '2023-10-14'),
        ('2023-10-22', '2023-10-28'),
        ('2023-10-29', '2023-11-02')
    ]
    
    granules = []
    for start, end in target_dates:
        print(f'[*] Searching BioSCape ({start} to {end})...')
        results = earthaccess.search_data(
            short_name='BioSCape_AVNG_L2B_BRDF_GCFR_2385',
            temporal=(start, end)
        )
        granules.extend(results)
    
    print(f'\n[✔] Found {len(granules)} targeted granules.')
    
    session = earthaccess.get_requests_https_session()
    
    for i, g in enumerate(granules):
        url = g.data_links()[0]
        filename = os.path.basename(url)
        s3_path = f's3://gaia-datasets/{filename}'
        
        print(f'\n--- [{i+1}/{len(granules)}] SYNCING: {filename} ---')
        
        # Check if exists
        try:
            check = subprocess.run(['s3cmd', 'ls', s3_path], capture_output=True, text=True)
            if filename in check.stdout:
                print(f'[✔] Found existing file in S3. Skipping.')
                continue
        except:
            pass
            
        print(f'[*] Target: {s3_path}')
        
        start_time = time.time()
        try:
            with session.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Pipe to s3cmd put
                proc = subprocess.Popen(['s3cmd', 'put', '-', s3_path], stdin=subprocess.PIPE)
                
                downloaded = 0
                last_print = 0
                for chunk in r.iter_content(chunk_size=1024*1024*10): # 10MB chunks
                    if chunk:
                        proc.stdin.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        if percent - last_print >= 25:
                            print(f'  > Progress: {percent:.0f}% ({downloaded/(1024**3):.2f} GB / {total_size/(1024**3):.2f} GB)')
                            last_print = percent
                            
                proc.stdin.close()
                proc.wait()
                
            duration = time.time() - start_time
            print(f'[✔] SUCCESS: {filename} in {duration/60:.1f} min.')
        except Exception as e:
            print(f'[!] FAILED: {filename} | {e}')
            
    print('\n' + '='*50)
    print('SYNC COMPLETE.')
    print('='*50)

if __name__ == '__main__':
    sync_targeted()
