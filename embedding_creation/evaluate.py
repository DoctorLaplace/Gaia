"""
EVALUATION SUITE: Gaia Custom Embedding Model
Calculates metrics (R2, RMSE, MAE) and generates a performance dashboard.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import yaml

# Project Root Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from embedding_creation.finetune import GaiaAvirisModel
from embedding_creation.dataset import AvirisRichnessDataset

# Visual Style
plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")
COLORS = {'Model': '#007bff', 'Baseline': '#dc3545', 'Train': '#6c757d', 'Val': '#17a2b8'}

def run_inference(model, loader, richness_mean, richness_std, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for patches, labels in loader:
            with torch.amp.autocast('cuda', enabled=True):
                out = model(patches.to(device))
            real = out.cpu().numpy().flatten() * richness_std + richness_mean
            preds.extend(real)
            targets.extend(labels.numpy().flatten())
    return np.array(preds), np.array(targets)

def compute_metrics(targets, preds):
    r2 = r2_score(targets, preds) if len(targets) > 1 else 0
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'count': len(targets)}

def main():
    print("\n" + "═"*60)
    print("  GAIA EMBEDDING: FINE-TUNE EVALUATION")
    print("═"*60)

    # 1. Load Config
    config_path = os.path.join(PROJECT_ROOT, "embedding_creation", "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, cfg['finetune']['checkpoint_dir'])
    ckpt_path = os.path.join(ckpt_dir, "finetuned_aviris_best.pth")

    if not os.path.exists(ckpt_path):
        print(f"[!] No fine-tuned checkpoint found at {ckpt_path}")
        return

    # 2. Load Model & Normalization
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    richness_mean = ckpt.get('richness_mean', 23.7)
    richness_std = ckpt.get('richness_std', 6.7)
    
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
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. Load Dataset
    nc_dir = os.path.join(PROJECT_ROOT, cfg['data']['labeled_nc_dir'].replace('{res}', '30'))
    richness_csv = os.path.join(PROJECT_ROOT, cfg['data']['richness_csv'])
    
    band_mean = torch.tensor(ckpt['band_mean']).reshape(cfg['data']['num_bands'], 1, 1)
    band_std = torch.tensor(ckpt['band_std']).reshape(cfg['data']['num_bands'], 1, 1)

    dataset = AvirisRichnessDataset(
        nc_dir=nc_dir,
        richness_csv=richness_csv,
        patch_size=cfg['data']['patch_size'],
        num_bands=cfg['data']['num_bands'],
        band_mean=band_mean,
        band_std=band_std
    )

    # Split (Same seed as fine-tune to respect the val set)
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=cfg['finetune']['val_split'], random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)

    # 4. Inference
    print("[*] Running inference on Train/Val splits...")
    train_preds, train_targets = run_inference(model, train_loader, richness_mean, richness_std, device)
    val_preds, val_targets = run_inference(model, val_loader, richness_mean, richness_std, device)
    
    # 5. Baseline
    baseline_val = np.full_like(val_targets, fill_value=np.mean(train_targets))
    
    m_train = compute_metrics(train_targets, train_preds)
    m_val = compute_metrics(val_targets, val_preds)
    m_base = compute_metrics(val_targets, baseline_val)
    
    imp = (1 - (m_val['RMSE'] / m_base['RMSE'])) * 100

    print(f"\n[Validation Set vs Baseline]")
    print(f"  Model RMSE:    {m_val['RMSE']:.4f}  ({imp:+.1f}% vs baseline)")
    print(f"  Baseline RMSE: {m_base['RMSE']:.4f}")
    print(f"  Final Val R2:  {m_val['R2']:.4f}")

    # 6. Dashboard
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)

    # Scatters
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=train_targets, y=train_preds, alpha=0.3, color=COLORS['Train'], label='Train')
    sns.scatterplot(x=val_targets, y=val_preds, alpha=0.9, color=COLORS['Val'], label='Validation')
    lims = [min(train_targets.min(), val_targets.min())-2, max(train_targets.max(), val_targets.max())+2]
    ax1.plot(lims, lims, '--k', alpha=0.3)
    ax1.set_title("Predicted vs Actual Richness")
    ax1.set_xlabel("Actual Species"); ax1.set_ylabel("Predicted Species")
    ax1.legend()

    # Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = val_preds - val_targets
    sns.histplot(residuals, kde=True, color=COLORS['Val'], ax=ax2)
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Validation Error Distribution")

    # Metrics Summary Text
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    summary = (
        f"GAIA EMBEDDING RESULTS\n"
        f"----------------------\n"
        f"Validation R2:   {m_val['R2']:.4f}\n"
        f"Validation RMSE: {m_val['RMSE']:.2f}\n"
        f"Validation MAE:  {m_val['MAE']:.2f}\n\n"
        f"Improvement over 'Mean Guess' baseline: {imp:.1f}%"
    )
    ax3.text(0.5, 0.5, summary, ha='center', va='center', family='monospace', fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    plot_path = os.path.join(PROJECT_ROOT, "embedding_creation", "evaluation_dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✔] Evaluation Dashboard saved to: {plot_path}")
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()
