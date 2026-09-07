import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel

class LOFODataset(Dataset):
    """Dataset for a specific fold of LOFO."""
    def __init__(self, samples):
        self.samples = samples # List of (patch_tensor, richness)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, richness = self.samples[idx]
        return patch, torch.tensor([float(richness)])

def discover_all_samples(nc_dir, richness_csv):
    """
    Finds every site-flightline intersection across the entire dataset.
    Returns a dictionary: { flightline_name: [(patch, richness, site_id), ...] }
    """
    richness_df = pd.read_csv(richness_csv)
    richness_df = richness_df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon', 'richness': 'richness'})
    
    nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.nc')]
    flightline_samples = {}

    print(f"[*] Discovering samples across {len(nc_files)} flightlines...")
    for nc_name in tqdm(nc_files):
        nc_path = os.path.join(nc_dir, nc_name)
        try:
            ds = BioScapeNetCDFDataset(nc_path, richness_csv=None)
            samples = []
            for idx, row in richness_df.iterrows():
                patch = ds.get_patch_at_latlon(row['lat'], row['lon'])
                if patch is not None:
                    # We store the patch, target, and the unique SiteID to prevent leakage
                    samples.append((patch, row['richness'], row.get('SiteID', f"site_{idx}")))
            
            if samples:
                flightline_samples[nc_name] = samples
        except Exception as e:
            print(f"[!] Error reading {nc_name}: {e}")
            
    return flightline_samples

def run_lofo_validation(nc_dir, richness_csv, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting LOFO Validation on {device}")
    
    flightline_samples = discover_all_samples(nc_dir, richness_csv)
    flightlines = list(flightline_samples.keys())
    
    if not flightlines:
        print("[!] No samples found in any flightlines.")
        return

    print(f"[✔] Found {sum(len(s) for s in flightline_samples.values())} total intersections across {len(flightlines)} flightlines.")
    
    results = []
    
    # LOFO Loop
    for test_fl in flightlines:
        print(f"\n>>> Fold: Testing on {test_fl} ({len(flightline_samples[test_fl])} samples)")
        
        # Prepare Data
        test_data = flightline_samples[test_fl]
        test_site_ids = set(s[2] for s in test_data)
        
        train_samples = []
        for fl_name, samples in flightline_samples.items():
            if fl_name == test_fl: continue
            # Filter out any sites that are in our test set to ensure OOD
            for s in samples:
                if s[2] not in test_site_ids:
                    train_samples.append((s[0], s[1]))
        
        if not train_samples:
            print(f"[!] Skipping fold {test_fl}: No training samples available after filtering.")
            continue
            
        print(f"[*] Training on {len(train_samples)} samples from {len(flightlines)-1} other flightlines.")
        
        # Initialize Model and Optimizer
        model = GaiaTransferModel(num_targets=1).to(device)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        foundation_ckpt = os.path.join(project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth")
        if os.path.exists(foundation_ckpt):
            model.load_foundation_weights(foundation_ckpt, device)
            
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(LOFODataset(train_samples), batch_size=8, shuffle=True)
        test_loader = DataLoader(LOFODataset([(s[0], s[1]) for s in test_data]), batch_size=8)
        
        # Quick Fine-tune
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device).float()
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                out = model(images)
                preds.extend(out.cpu().numpy().flatten())
                targets.extend(labels.cpu().numpy().flatten())
        
        # Metrics
        mse = mean_squared_error(targets, preds)
        try:
            r2 = r2_score(targets, preds)
        except:
            r2 = np.nan
            
        print(f"[Fold Result] MSE: {mse:.4f} | R²: {r2:.4f}")
        results.append({'flightline': test_fl, 'mse': mse, 'r2': r2, 'samples': len(targets)})

    # Summary
    df_results = pd.DataFrame(results)
    print("\n" + "="*40)
    print("LOFO VALIDATION SUMMARY")
    print("="*40)
    print(df_results)
    print(f"\nMean R²: {df_results['r2'].dropna().mean():.4f} (+/- {df_results['r2'].dropna().std():.4f})")
    print(f"Mean MSE: {df_results['mse'].mean():.4f}")
    
    report_path = os.path.join(project_root, "reports", "lofo_validation_report.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_results.to_csv(report_path, index=False)
    print(f"[✔] LOFO report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, default="data/bioscape")
    parser.add_argument("--richness_csv", type=str, default="data/bioscape/western_cape_site_embeddings.csv")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    run_lofo_validation(args.nc_dir, args.richness_csv, epochs=args.epochs)
