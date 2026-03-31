"""
INCREMENTAL SCALING: Add N new flightlines to the download batch.
This script compares 'mega_batch.txt' with 'full_inventory.txt' and 
appends N new, unique IDs to the batch.
"""
import os, sys

def incremental_add(n=10):
    batch_file = "mega_batch.txt"
    inventory_file = "full_inventory.txt"
    
    if not os.path.exists(inventory_file):
        print(f"[!] {inventory_file} not found. Run 'python src/generate_full_granule_list.py' first.")
        return

    # 1. Load current batch IDs
    current_batch = set()
    if os.path.exists(batch_file):
        with open(batch_file, 'r') as f:
            current_batch = set(line.strip() for line in f if line.strip())
    
    # 2. Load full inventory IDs
    with open(inventory_file, 'r') as f:
        full_inventory = [line.strip() for line in f if line.strip()]
    
    # 3. Find candidates (in inventory but NOT in current batch)
    candidates = [id for id in full_inventory if id not in current_batch]
    
    if not candidates:
        print("[✔] All flightlines from inventory are already in your batch!")
        return
        
    # 4. Take the next N
    to_add = candidates[:n]
    
    # 5. Append to mega_batch.txt
    with open(batch_file, 'a') as f:
        for item in to_add:
            f.write(item + "\n")
            
    print(f"\n" + "="*50)
    print(f"INCREMENTAL UPDATE: +{len(to_add)} NEW FLIGHTLINES")
    print(f"="*50)
    for i, item in enumerate(to_add):
        print(f"[{i+1:>2}] Added: {item}")
        
    print(f"\n[SUMMARY]")
    print(f"Current Batch Size: {len(current_batch) + len(to_add)}")
    print(f"Remaining in Inventory: {len(candidates) - len(to_add)}")
    print(f"\nNext Step: Run 'python src/mega_download.py' to fetch these new files.")
    print("="*50)

if __name__ == "__main__":
    # Default to 10 if no argument provided
    n_count = 10
    if len(sys.argv) > 1:
        try:
            n_count = int(sys.argv[1])
        except ValueError:
            print(f"[!] Invalid number: {sys.argv[1]}. Using default of 10.")
            
    incremental_add(n_count)
