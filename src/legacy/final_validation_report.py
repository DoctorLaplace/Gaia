"""
FINAL VALIDATION REPORT: Honest 48-site performance assessment.
Deduplicates by GPS and evaluates ONLY the hold-out validation split.
"""
import os, sys, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel, MultiFlightBioScapeDataset

def generate_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_dir = "data/bioscape"
    richness_csv = "data/bioscape/western_cape_site_embeddings.csv"
    checkpoint_path = "checkpoints/gaia_bioscape_best.pth"
    
    # 1. Load exact deduplicated dataset logic
    nc_paths = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')])
    ds = MultiFlightBioScapeDataset(nc_paths, richness_csv)
    
    # 2. Re-create the 60/20/20 split logic (same random_state=42)
    indices = np.arange(len(ds))
    # First split off the test set (20%) - This is the "Never Seen" set
    remaining_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    # The remaining 80% was used for train (60%) and val (20%)
    
    # 3. Load Model
    model = GaiaTransferModel(num_targets=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"\n[*] Evaluating {len(test_idx)} strictly held-out TEST sites...")
    
    results = []
    
    # 4. Infer
    for idx in tqdm(test_idx, desc="Final Testing"):
        nc, lat, lon, richness = ds.mappings[idx]
        patch, label = ds[idx]
        with torch.no_grad():
            pred = model(patch.unsqueeze(0).to(device)).item()
        
        results.append({
            'site_id': f"Site_{lat:.4f}_{lon:.4f}",
            'latitude': lat,
            'longitude': lon,
            'actual_richness': richness,
            'predicted_richness': pred,
            'error': richness - pred,
            'abs_error': abs(richness - pred),
            'flightline': nc
        })
        
    df = pd.DataFrame(results)
    
    # 5. Summary Metrics
    actuals = df['actual_richness'].values
    preds = df['predicted_richness'].values
    
    total_r2 = r2_score(actuals, preds)
    total_mse = mean_squared_error(actuals, preds)
    total_mae = df['abs_error'].mean()
    
    print(f"\n" + "="*40)
    print(f"FINAL GEOGRAPHIC VALIDATION SUMMARY")
    print(f"="*40)
    print(f"Total Unique Sites (Hold-out): {len(df)}")
    print(f"Final Validation R2:          {total_r2:.4f}")
    print(f"Mean Absolute Error:         {total_mae:.2f} species")
    print(f"Root Mean Squared Error:     {np.sqrt(total_mse):.2f} species")
    
    # 6. Save Report
    report_csv = "reports/final_validation_holdout_48.csv"
    df.to_csv(report_csv, index=False)
    print(f"\n[✔] Detailed CSV report saved to {report_csv}")
    
    # Print Top 15 worst/best
    print("\n[Sample of Individual Predictions]")
    print(df[['site_id', 'actual_richness', 'predicted_richness', 'error']].head(15))

if __name__ == "__main__":
    generate_report()
