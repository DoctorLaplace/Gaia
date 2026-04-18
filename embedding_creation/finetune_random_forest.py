"""
RANDOM FOREST BASELINE: Frozen Feature Evaluation
Extracts fixed embeddings from the foundation model and trains a Random Forest.
Usage: python -m embedding_creation.finetune_random_forest
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

# Project Root Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from embedding_creation.finetune import GaiaAvirisModel
from embedding_creation.dataset import AvirisRichnessDataset

# Visual Style
plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")

def extract_embeddings(model, loader, device):
    """Pass all data through encoder and return (embeddings, labels)."""
    model.eval()
    embeddings, targets = [], []
    
    print("[*] Extracting frozen embeddings from foundation model...")
    with torch.no_grad():
        for patches, labels in tqdm(loader, desc="Inference"):
            # The encoder is the backbone. We want the features BEFORE the MLP head.
            # In our model, forward() returns the output of the mlp_head.
            # We want the latent tokens before the head.
            
            with torch.amp.autocast('cuda', enabled=True):
                # Run feature extraction
                # Some versions return (latent, spatial_latent, spectral_latent)
                # Others might return additional attention maps. We take the first.
                features = model.encoder.forward_features(patches.to(device))
                if isinstance(features, (tuple, list)):
                    x = features[0]
                else:
                    x = features
                
                # Global pooling (same as used for the head input)
                if model.encoder.pool == "mean":
                    x = x.mean(dim=1)
                else:
                    x = x[:, 0]
            
            embeddings.append(x.cpu().numpy())
            targets.append(labels.numpy().flatten())
            
    return np.concatenate(embeddings), np.concatenate(targets)

def main():
    print("\n" + "═"*60)
    print("  GAIA EMBEDDING: RANDOM FOREST EVALUATION")
    print("═"*60)

    # 1. Load Config
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['finetune']['checkpoint_dir'])
    pretrain_ckpt_path = os.path.join(ckpt_dir, "pretrained_aviris_best.pth")

    if not os.path.exists(pretrain_ckpt_path):
        print(f"[!] No pre-trained checkpoint found at {pretrain_ckpt_path}")
        return

    # 2. Load Foundation Model (Backbone)
    model = GaiaAvirisModel(
        num_channels=cfg['data']['num_bands'],
        dim=cfg['model']['dim'],
        depth=cfg['model']['depth'],
        heads=cfg['model']['heads'],
        mlp_dim=cfg['model']['mlp_dim'],
        spectral_patch_size=cfg['model']['spectral_patch_size'],
        spatial_patch_size=cfg['model']['spatial_patch_size'],
        image_size=cfg['data']['patch_size']
    ).to(device)
    
    # Load correctly
    ckpt = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # Filter state dict (remove heads if necessary)
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    if not encoder_state: encoder_state = state_dict
    
    # We only care about the encoder part for extraction
    model.encoder.load_state_dict(encoder_state, strict=False)
    model.eval()

    # 3. Load Dataset
    nc_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'].replace('{res}', '30'))
    richness_csv = os.path.join(PROJECT_ROOT, cfg['data']['richness_csv'])
    
    # Norm stats from checkpoint
    band_mean = torch.tensor(ckpt['band_mean']).reshape(cfg['data']['num_bands'], 1, 1) if 'band_mean' in ckpt else None
    band_std = torch.tensor(ckpt['band_std']).reshape(cfg['data']['num_bands'], 1, 1) if 'band_std' in ckpt else None

    dataset = AvirisRichnessDataset(
        nc_dir=nc_dir, richness_csv=richness_csv,
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        band_mean=band_mean, band_std=band_std
    )

    # 4. Feature Extraction
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    X, y = extract_embeddings(model, loader, device)

    # 5. Split (Seed 42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg['finetune']['val_split'], random_state=42
    )

    # 6. Train Random Forest
    print(f"[*] Training Random Forest on {len(X_train)} sites...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 7. Evaluate
    train_score = rf.score(X_train, y_train)
    val_preds = rf.predict(X_val)
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # Baseline comparison (Mean Guess)
    base_preds = np.full_like(y_val, y_train.mean())
    base_r2 = r2_score(y_val, base_preds)
    base_rmse = np.sqrt(mean_squared_error(y_val, base_preds))

    print("\n" + "─"*40)
    print(f" RANDOM FOREST RESULTS (Frozen Embeddings)")
    print(f"  Train R2: {train_score:.4f}")
    print(f"  Val R2:   {val_r2:.4f}")
    print(f"  Val RMSE: {val_rmse:.2f}")
    print(f"  Baseline RMSE: {base_rmse:.2f}")
    print("─"*40)

    # 8. Plot Result
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_val, y=val_preds, alpha=0.9, s=80, color='#17a2b8', label='Validation Set')
    lims = [min(y.min(), val_preds.min())-2, max(y.max(), val_preds.max())+2]
    plt.plot(lims, lims, '--k', alpha=0.3, label='1:1 Reference')
    plt.title(f"Random Forest on Frozen Gaia Embeddings\n(Val R² = {val_r2:.4f}, RMSE = {val_rmse:.2f})")
    plt.xlabel("Actual Richness"); plt.ylabel("RF Predicted Richness")
    plt.legend()
    
    save_path = os.path.join(PROJECT_ROOT, "embedding_creation", "rf_evaluation.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[✔] RF Baseline Dashboard saved to: {save_path}")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()
