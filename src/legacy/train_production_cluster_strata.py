import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import fsspec
import yaml
import threading

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bioscape_dataset import BioScapeNetCDFDataset
from src.vit_spatial_spectral import ViTSpatialSpectral
try:
    from .s3_utils import get_s3_fs, get_cached_s3_fs, get_s3_bytes_pulled, get_worker_statuses
except ImportError:
    from s3_utils import get_s3_fs, get_cached_s3_fs, get_s3_bytes_pulled, get_worker_statuses

class GaiaTransferModel(nn.Module):
    def __init__(self, num_targets=1, patch_size=16):
        super().__init__()
        # Backbone (Frozen)
        self.encoder = ViTSpatialSpectral(
            image_size=patch_size, spatial_patch_size=1, spectral_patch_size=10, 
            num_classes=num_targets, dim=96, depth=4, heads=8, mlp_dim=64, 
            dropout=0.1, emb_dropout=0.1, channels=200, spectral_pos=torch.arange(20), 
            spectral_pos_embed=True, blockwise_patch_embed=True, spectral_only=False, pixelwise=False
        )
        # Small MLP head (96 -> 64 -> 1) with GELU
        self.encoder.mlp_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_targets)
        )

    def load_foundation_weights(self, checkpoint_path, device):
        print(f"{Colors.OKBLUE}[*] Loading foundation weights from {checkpoint_path}{Colors.ENDC}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'mlp_head' in k or 'mask_token' in k: continue
            new_key = k.replace("encoder.", "")
            if new_key == "pos_embed":
                ckpt_grid = int(v.shape[1]**0.5)
                target_grid = int(self.encoder.pos_embed.shape[1]**0.5)
                if ckpt_grid != target_grid:
                    v = v.permute(0, 2, 1).reshape(1, v.shape[2], ckpt_grid, ckpt_grid)
                    v = torch.nn.functional.interpolate(v, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
                    v = v.reshape(1, v.shape[1], target_grid * target_grid).permute(0, 2, 1)
            new_state_dict[new_key] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # 1. Feature extraction: returns (B, 16, 16, num_targets)
        out = self.encoder(x)
        # 2. Spatial Average Pooling (averaging predicted richness across 16x16 patch)
        return out.mean(dim=(1, 2))

class MultiFlightBioScapeDataset(Dataset):
    def __init__(self, nc_paths, richness_csv, patch_size=16, augment=False, 
                 cache_path=None, mask_water_vapor=True):
        self.patch_size = patch_size
        self.augment = augment
        self.nc_paths = nc_paths
        self.richness_csv = richness_csv
        self.mappings = []
        self.dataset_cache = {}
        self.cache_lock = threading.Lock()
        self.mask_water_vapor = mask_water_vapor
        # Global per-band normalization stats (set after construction)
        self.band_mean = None  # shape: (200, 1, 1)
        self.band_std = None   # shape: (200, 1, 1)
        
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            import json
            print(f"{Colors.OKBLUE}[*] Loading mapping from cache: {cache_path}{Colors.ENDC}")
            with open(cache_path, 'r') as f:
                self.mappings = json.load(f)
            print(f"{Colors.OKGREEN}[OK] Loaded {len(self.mappings)} sites from cache.{Colors.ENDC}")
            return

        self._initialize_mapping(richness_csv, nc_paths, patch_size, cache_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'cache_lock' in state:
            del state['cache_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cache_lock = threading.Lock()

    def _initialize_mapping(self, richness_csv, nc_paths, patch_size, cache_path=None):
        self.richness_df = pd.read_csv(richness_csv)
        self.richness_df = self.richness_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness', 'Richness': 'richness'})
        
        # Load metadata to merge CLUSTER_ID
        metadata_csv = os.path.join(os.path.dirname(richness_csv), "biosoundscape_site_metadata.csv")
        if os.path.exists(metadata_csv):
            print(f"{Colors.OKBLUE}[*] Loading metadata cluster mappings from {metadata_csv}...{Colors.ENDC}")
            meta_df = pd.read_csv(metadata_csv)[['SiteID', 'CLUSTER_ID']]
            self.richness_df = self.richness_df.merge(meta_df, on='SiteID', how='left')
            self.richness_df['CLUSTER_ID'] = self.richness_df['CLUSTER_ID'].fillna(-1).astype(int)
        else:
            print(f"{Colors.WARNING}[!] Metadata file not found at {metadata_csv}. Defaulting all CLUSTER_ID to -1.{Colors.ENDC}")
            self.richness_df['CLUSTER_ID'] = -1
            
        print(f"{Colors.OKBLUE}[*] Mapping {len(self.richness_df)} potential sites across {len(nc_paths)} flightlines...{Colors.ENDC}")
        
        # Parallel mapping with spatial filtering
        from concurrent.futures import ThreadPoolExecutor
        
        def process_one(nc):
            local_mappings = []
            try:
                ds = BioScapeNetCDFDataset(nc, richness_csv=None, patch_size=patch_size, 
                                          quiet=True, use_cache=False, mask_water_vapor=self.mask_water_vapor)
                # Fast spatial filter using bounding box
                min_lon, min_lat, max_lon, max_lat = ds.get_bounds()
                mask = (self.richness_df['lat'] >= min_lat) & (self.richness_df['lat'] <= max_lat) & \
                       (self.richness_df['lon'] >= min_lon) & (self.richness_df['lon'] <= max_lon)
                relevant_sites = self.richness_df[mask]
                
                if len(relevant_sites) > 0:
                    for _, row in relevant_sites.iterrows():
                        easting, northing = ds.transformer.transform(row['lon'], row['lat'])
                        x_idx, y_idx = int((easting-ds.origin_x)/ds.pixel_w), int((northing-ds.origin_y)/ds.pixel_h)
                        p = ds.patch_size // 2
                        if y_idx >= p and y_idx < ds.height-p and x_idx >= p and x_idx < ds.width-p:
                            # Verify if the patch is valid (not mostly no-data)
                            ds._ensure_open()
                            cube = ds.h5_file['reflectance/reflectance'] if 'reflectance/reflectance' in ds.h5_file else ds.h5_file['reflectance']
                            patch_band = cube[50, y_idx-p : y_idx+p, x_idx-p : x_idx+p]
                            nodata_ratio = (patch_band < 0.0).mean()
                            if nodata_ratio < 0.5:
                                local_mappings.append((
                                    nc, 
                                    float(row['lat']), 
                                    float(row['lon']), 
                                    float(row['richness']),
                                    int(row.get('CLUSTER_ID', -1))
                                ))
                ds.close()
            except Exception as e: 
                pass
            return local_mappings

        with ThreadPoolExecutor(max_workers=10) as executor:
            all_results = list(tqdm(executor.map(process_one, nc_paths), total=len(nc_paths), desc="Mapping Sites"))
            
        seen_coords = set()
        for res_list in all_results:
            for item in res_list:
                coord_key = (round(item[1], 5), round(item[2], 5))
                if coord_key not in seen_coords:
                    self.mappings.append(item)
                    seen_coords.add(coord_key)

        # Only save cache if we processed all granules (not a test run)
        if cache_path and len(nc_paths) > 2:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.mappings, f)
            print(f"{Colors.OKGREEN}[OK] Mapping cached to {cache_path}{Colors.ENDC}")
                    
        print(f"{Colors.OKGREEN}[OK] Dataset Ready: {len(self.mappings)} unique sites.{Colors.ENDC}")

    def compute_band_stats(self, cache_path=None):
        """Compute global per-band mean/std across ALL patches. Results are cached to disk."""
        if cache_path and os.path.exists(cache_path):
            import json
            with open(cache_path, 'r') as f:
                stats = json.load(f)
            self.band_mean = torch.tensor(stats['mean']).reshape(200, 1, 1)
            self.band_std = torch.tensor(stats['std']).reshape(200, 1, 1)
            print(f"{Colors.OKBLUE}[*] Loaded band stats from cache: {cache_path}{Colors.ENDC}")
            return
        
        print(f"{Colors.OKBLUE}[*] Computing global band statistics across {len(self)} patches...{Colors.ENDC}")
        
        running_sum = torch.zeros(200)
        running_sq_sum = torch.zeros(200)
        n_pixels = 0
        
        def process_patch(i):
            nc_path, lat, lon, _, _ = self.mappings[i]
            with self.cache_lock:
                if nc_path not in self.dataset_cache:
                    self.dataset_cache[nc_path] = BioScapeNetCDFDataset(nc_path, richness_csv=None, patch_size=self.patch_size, quiet=True)
                ds = self.dataset_cache[nc_path]
            
            patch, _ = ds.get_patch_at_latlon(lat, lon)
            if patch is None: return None
            return patch.sum(dim=(1, 2)), (patch**2).sum(dim=(1, 2)), patch.shape[1] * patch.shape[2]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_patch, i): i for i in range(len(self))}
            results = [None] * len(self)
            for future in tqdm(as_completed(futures), total=len(self), desc="Band Stats"):
                idx = futures[future]
                results[idx] = future.result()
            
        for res in results:
            if res:
                p_sum, p_sq_sum, p_pixels = res
                running_sum += p_sum
                running_sq_sum += p_sq_sum
                n_pixels += p_pixels
                
        if n_pixels == 0:
            print(f"{Colors.FAIL}[!] Could not compute band stats (no valid pixels found).{Colors.ENDC}")
            return
            
        global_mean = running_sum / n_pixels
        global_std = ((running_sq_sum / n_pixels) - global_mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
        
        self.band_mean = global_mean.reshape(200, 1, 1)
        self.band_std = global_std.reshape(200, 1, 1)
        
        if cache_path:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({'mean': global_mean.tolist(), 'std': global_std.tolist()}, f)
            print(f"{Colors.OKGREEN}[OK] Band stats cached to {cache_path}{Colors.ENDC}")
    
    def set_band_stats(self, band_mean, band_std):
        self.band_mean = torch.tensor(band_mean).reshape(200, 1, 1) if not isinstance(band_mean, torch.Tensor) else band_mean.reshape(200, 1, 1)
        self.band_std = torch.tensor(band_std).reshape(200, 1, 1) if not isinstance(band_std, torch.Tensor) else band_std.reshape(200, 1, 1)

    def __len__(self): return len(self.mappings)
    
    def __getitem__(self, idx):
        nc_path, lat, lon, richness, cluster_id = self.mappings[idx]
        
        with self.cache_lock:
            if nc_path not in self.dataset_cache:
                self.dataset_cache[nc_path] = BioScapeNetCDFDataset(
                    nc_path, richness_csv=None, patch_size=self.patch_size, 
                    quiet=True, mask_water_vapor=self.mask_water_vapor
                )
            ds = self.dataset_cache[nc_path]
        
        patch, bounds = ds.get_patch_at_latlon(lat, lon)
        if patch is None:
            patch = torch.zeros(200, self.patch_size, self.patch_size)
            
        if self.augment:
            patch = torch.rot90(patch, k=np.random.randint(0, 4), dims=(1, 2))
            if np.random.random() > 0.5: patch = torch.flip(patch, dims=[2])
        return patch, torch.tensor([float(richness)])

def train_production(nc_dir=None, richness_csv=None, epochs=None, batch_size=None, 
                     test_run=False, freeze_encoder=True, unfreeze_epoch=None,
                     use_mosaic=False, mask_water_vapor=True, leave_out=15, seed=42, patience=25):
    # Load config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config['bioscape']
    if use_mosaic:
        nc_dir = nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else "data/bioscape/30m_v2")
        print(f"{Colors.WARNING}[*] MOSAIC MODE: Using Level 3 Tiles from {nc_dir}{Colors.ENDC}")
    else:
        nc_dir = nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])
    
    richness_csv = richness_csv or os.path.join(project_root, b_cfg['richness_csv'])
    epochs = epochs or b_cfg['epochs']
    batch_size = batch_size or b_cfg['batch_size']
    patch_size = b_cfg.get('patch_size', 16)
    
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"{Colors.HEADER}===================================================={Colors.ENDC}\n{Colors.HEADER}Gaia Fine-Tuning (Cluster-Stratified) on {device}{Colors.ENDC}")
    
    if test_run:
        print(f"{Colors.FAIL}[!] TEST RUN ENABLED: Using only 2 granules, 1 epoch, leaving out {min(leave_out, 3)} clusters.{Colors.ENDC}")
        epochs = 1
        leave_out = min(leave_out, 3)
    
    # Use fsspec for S3 directory listing
    if nc_dir.startswith("s3://"):
        fs = get_s3_fs()
        nc_paths = ["s3://" + f for f in fs.ls(nc_dir) if f.endswith('.nc')]
    else:
        nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
        
    if use_mosaic and not nc_dir.startswith("s3"):
        # Filter by tile_names.json if in mosaic mode
        tile_path = os.path.join(project_root, "tile_names.json")
        if os.path.exists(tile_path):
            import json
            try:
                with open(tile_path, 'r', encoding='utf-8') as f:
                    tiles = {t['tile'] for t in json.load(f)['tiles']}
            except (UnicodeDecodeError, json.JSONDecodeError):
                with open(tile_path, 'r', encoding='utf-16') as f:
                    tiles = {t['tile'] for t in json.load(f)['tiles']}
            
            nc_paths = [p for p in nc_paths if any(t in p for t in tiles)]
            print(f"{Colors.OKBLUE}[*] Filtered to {len(nc_paths)} tiles from tile_names.json{Colors.ENDC}")
        
    if not nc_paths:
        print(f"{Colors.FAIL}[!] No NetCDF files found in {nc_dir}{Colors.ENDC}")
        return
        
    if test_run:
        nc_paths = nc_paths[:2]
        
    cache_suffix = f"_p{patch_size}_strata.json"
    if use_mosaic: cache_suffix = "_mosaic" + cache_suffix
    cache_name = ("s3_mapping" if nc_dir.startswith("s3") else "local_mapping") + cache_suffix
    mapping_cache = os.path.join(project_root, "data", "bioscape", cache_name)
    
    full_dataset = MultiFlightBioScapeDataset(
        nc_paths, richness_csv, patch_size=patch_size, 
        augment=True, cache_path=mapping_cache, mask_water_vapor=mask_water_vapor
    )
    if len(full_dataset) == 0:
        print(f"{Colors.FAIL}[!] No training samples found. Stopping.{Colors.ENDC}")
        return
    
    # Compute global per-band normalization stats (cached to disk)
    band_stats_name = ("s3_band_stats" if nc_dir.startswith("s3") else "local_band_stats") + cache_suffix
    band_stats_cache = os.path.join(project_root, "data", "bioscape", band_stats_name)
    full_dataset.compute_band_stats(cache_path=band_stats_cache)
    
    # Clear cache before DataLoader (Windows Multiprocessing)
    for ds in full_dataset.dataset_cache.values():
        ds.close()
    full_dataset.dataset_cache.clear()
    
    # Compute target normalization stats from the mapping
    all_richness = np.array([m[3] for m in full_dataset.mappings])
    richness_mean = float(all_richness.mean())
    richness_std = float(all_richness.std()) + 1e-6
    print(f"{Colors.OKBLUE}[*] Richness Stats: mean={richness_mean:.1f}, std={richness_std:.1f}, min={all_richness.min():.0f}, max={all_richness.max():.0f}{Colors.ENDC}")
        
    # --- Cluster-Stratified Train/Validation Split ---
    mappings = full_dataset.mappings
    sample_clusters = np.array([m[4] for m in mappings])
    unique_clusters = np.unique(sample_clusters)
    
    # Exclude noise (-1) from validation split options (will always stay in training)
    val_candidates = unique_clusters[unique_clusters != -1]
    
    if leave_out >= len(val_candidates):
        print(f"{Colors.WARNING}[!] Warning: Requested leaving out {leave_out} clusters, but only {len(val_candidates)} candidates exist. Leaving out {len(val_candidates)-1} instead.{Colors.ENDC}")
        leave_out = len(val_candidates) - 1
        
    # Set seed for random selection (but randomizes if seed is changed)
    np.random.seed(seed)
    val_clusters = np.random.choice(val_candidates, size=leave_out, replace=False)
    
    train_idx = np.where(~np.isin(sample_clusters, val_clusters))[0]
    val_idx = np.where(np.isin(sample_clusters, val_clusters))[0]
    
    print(f"\n{Colors.OKGREEN}[OK] Cluster-Stratified Split completed (using seed {seed}):")
    print(f"    - Held out {len(val_clusters)} clusters for validation: {sorted(list(val_clusters))}")
    print(f"    - Training set: {len(train_idx)} samples from {len(unique_clusters) - len(val_clusters)} clusters")
    print(f"    - Validation set: {len(val_idx)} samples from {len(val_clusters)} clusters{Colors.ENDC}\n")
    
    # Load num_workers from config
    try:
        with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
            cfg = yaml.safe_load(f).get('bioscape', {})
            num_workers = cfg.get('num_workers', 8)
    except:
        num_workers = 8
        
    if os.name == 'nt' and not nc_dir.startswith("s3"):
        num_workers = 0
    
    print(f"{Colors.OKBLUE}[*] Enabling {num_workers} workers for {'S3' if nc_dir.startswith('s3') else 'Local'} streaming...{Colors.ENDC}")
        
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
    foundation_ckpt = os.path.join(project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth")
    if os.path.exists(foundation_ckpt):
        model.load_foundation_weights(foundation_ckpt, device)
    
    # Freeze encoder
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"{Colors.OKBLUE}[*] Encoder FROZEN: {trainable}/{total} params trainable ({trainable/total*100:.1f}%){Colors.ENDC}")
    else:
        print(f"{Colors.OKBLUE}[*] Full fine-tuning: all parameters trainable{Colors.ENDC}")
        
    lr = b_cfg['learning_rate']
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    best_r2 = -float('inf')
    patience_counter = 0
    patience = patience
    
    for epoch in range(1, epochs + 1):
        # Progressive unfreezing
        if freeze_encoder and unfreeze_epoch and epoch == unfreeze_epoch:
            print(f"\n{Colors.WARNING}[*] Epoch {epoch}: UNFREEZING encoder for fine-tuning{Colors.ENDC}")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.encoder.mlp_head.parameters(), 'lr': lr},
                {'params': [p for n, p in model.encoder.named_parameters() if 'mlp_head' not in n], 'lr': lr * 0.3},
            ], weight_decay=0.05)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs - epoch + 1,
                steps_per_epoch=len(train_loader), pct_start=0.05
            )
        
        model.train(); train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        last_s3_bytes = get_s3_bytes_pulled()
        last_s3_time = time.time()
        
        for images, labels in pbar:
            images = images.to(device)
            labels_norm = ((labels - richness_mean) / richness_std).to(device).float()
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            # Update Instrumentation
            now = time.time()
            dt = now - last_s3_time
            if dt >= 2.0:
                curr_s3_bytes = get_s3_bytes_pulled()
                speed_mb = (curr_s3_bytes - last_s3_bytes) / (1024 * 1024 * dt)
                last_s3_bytes = curr_s3_bytes
                last_s3_time = now
                
                w_status = get_worker_statuses(num_workers)
                status_str = " | ".join([f"W{i}:{s[:8]}" for i, s in enumerate(w_status) if s])
                pbar.set_description(f"Epoch {epoch} [{speed_mb:.1f} MB/s] [{status_str}]")

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            scheduler.step()
            
        model.eval(); val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds_norm = model(images.to(device))
                preds_real = preds_norm.cpu().numpy().flatten() * richness_std + richness_mean
                val_preds.extend(preds_real)
                val_targets.extend(labels.numpy().flatten())
                
        epoch_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
        avg_loss = train_loss / len(train_loader)
        print(f"{Colors.OKCYAN}[*] Epoch {epoch} - Loss: {avg_loss:.4f} | Val R²: {epoch_r2:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}{Colors.ENDC}")
        
        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            patience_counter = 0
            ckpt_data = {
                'model_state_dict': model.state_dict(),
                'richness_mean': richness_mean,
                'richness_std': richness_std,
                'band_mean': full_dataset.band_mean.flatten().tolist(),
                'band_std': full_dataset.band_std.flatten().tolist(),
                'val_clusters': val_clusters.tolist()
            }
            torch.save(ckpt_data, os.path.join(project_root, "checkpoints", "gaia_bioscape_strata_best.pth"))
            print(f"{Colors.OKGREEN}[OK] New Best Model Saved (R²: {best_r2:.4f}){Colors.ENDC}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{Colors.WARNING}[*] Early stopping at epoch {epoch} (no improvement for {patience} epochs){Colors.ENDC}")
                break
            
    print(f"\n{Colors.OKGREEN}[OK] Training Complete. Best R²: {best_r2:.4f}{Colors.ENDC}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, help="Override S3/Local path")
    parser.add_argument("--richness_csv", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--test-run", action="store_true", help="1 epoch on 2 granules")
    parser.add_argument("--freeze", action="store_true", default=True,
                        help="Freeze encoder, train only head (default)")
    parser.add_argument("--no-freeze", dest="freeze", action="store_false",
                        help="Fine-tune all parameters")
    parser.add_argument("--unfreeze-epoch", type=int, default=None,
                        help="Unfreeze encoder at this epoch for progressive fine-tuning")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--mosaic", action="store_true", help="Use Level 3 Mosaic tiles")
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Disable water vapor masking")
    parser.set_defaults(mask=True)
    parser.add_argument("--leave-out", type=int, default=15, help="Number of clusters to hold out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for cluster split")
    
    args = parser.parse_args()
    train_production(args.nc_dir, args.richness_csv, epochs=args.epochs, 
                     test_run=args.test_run, freeze_encoder=args.freeze,
                     unfreeze_epoch=args.unfreeze_epoch, use_mosaic=args.mosaic,
                     mask_water_vapor=args.mask, leave_out=args.leave_out, seed=args.seed,
                     patience=args.patience)
