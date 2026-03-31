import xarray as xr
import h5py
import sys

def inspect_file(path):
    print(f"--- Inspecting: {path} ---")
    
    # 1. Try h5py to see groups
    with h5py.File(path, 'r') as f:
        print("\nh5py Groups/Datasets:")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  [D] {name} - Shape: {obj.shape}")
            else:
                print(f"  [G] {name}")
        f.visititems(print_keys)

    # 2. Try xarray
    print("\nxarray (Root):")
    try:
        ds = xr.open_dataset(path)
        print(ds)
    except Exception as e:
        print(f"  Error opening root: {e}")

    # 3. Try xarray with 'reflectance' group
    print("\nxarray (Group: reflectance):")
    try:
        ds = xr.open_dataset(path, group='reflectance')
        print(ds)
    except Exception as e:
        print(f"  Error opening group 'reflectance': {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_file(sys.argv[1])
    else:
        print("Provide a path to an .nc file")
