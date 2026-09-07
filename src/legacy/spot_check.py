"""Quick spot-check: run the trained model on 10 unseen validation sites."""
import os, torch, numpy as np, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bioscape_dataset import BioScapeNetCDFDataset
from src.train_production import GaiaTransferModel, MultiFlightBioScapeDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc_dir = "data/bioscape"
nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]

ds = MultiFlightBioScapeDataset(nc_paths, "data/bioscape/western_cape_site_embeddings.csv")
indices = np.arange(len(ds))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

model = GaiaTransferModel(num_targets=1).to(device)
model.load_state_dict(torch.load("checkpoints/gaia_bioscape_best.pth", map_location=device, weights_only=True))
model.eval()

print(f"\nTotal unique sites: {len(ds)} | Val sites: {len(val_idx)}")
print(f"\n{'Site (lat, lon)':<30}  {'Actual':>7}  {'Predicted':>9}  {'Error':>7}")
print("-" * 60)

chosen = np.random.RandomState(99).choice(val_idx, size=min(10, len(val_idx)), replace=False)
preds_all, actuals_all = [], []
for i in chosen:
    nc, lat, lon, richness = ds.mappings[i]
    patch, label = ds[i]
    with torch.no_grad():
        pred = model(patch.unsqueeze(0).to(device)).item()
    err = richness - pred
    preds_all.append(pred)
    actuals_all.append(richness)
    print(f"({lat:.4f}, {lon:.4f})         {richness:>5.0f}     {pred:>6.1f}     {err:>+5.1f}")

r2 = r2_score(actuals_all, preds_all)
print(f"\n{'='*60}")
print(f"10-Site Sample R2: {r2:.4f}")
print(f"Avg Abs Error: {np.mean(np.abs(np.array(actuals_all) - np.array(preds_all))):.1f} species")
