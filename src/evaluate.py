"""
GAIA MODEL EVALUATION & VISUALIZATION
Loads the best checkpoint and evaluates performance against a "Mean Guess" baseline.
Usage: python src/evaluate.py --nc_dir data/bioscape
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

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_production import GaiaTransferModel, MultiFlightBioScapeDataset

# Set visual style
plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {'Model': '#007bff', 'Baseline': '#dc3545', 'Train': '#6c757d', 'Val': '#17a2b8'}

def run_inference(model, loader, richness_mean, richness_std, device):
    """Run model on a dataloader and return (predictions, targets) in real units."""
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            real = out.cpu().numpy().flatten() * richness_std + richness_mean
            preds.extend(real)
            targets.extend(labels.numpy().flatten())
    return np.array(preds), np.array(targets)

def compute_metrics(targets, preds):
    r2 = r2_score(targets, preds) if len(targets) > 1 else 0
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'count': len(targets)}

def evaluate(nc_dir=None):
    print("\n" + "═"*60)
    print("  GAIA MODEL EVALUATION (PREMIUM VIZ)")
    print("═"*60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config['bioscape']
    nc_dir = nc_dir or (b_cfg['nc_dir_s3'] if b_cfg['use_s3'] else b_cfg['nc_dir_local'])
    richness_csv = os.path.join(project_root, b_cfg['richness_csv'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoint
    ckpt_path = os.path.join(project_root, "checkpoints", "gaia_bioscape_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[!] No checkpoint found at {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    richness_mean = ckpt.get('richness_mean', 23.66)
    richness_std = ckpt.get('richness_std', 6.72)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model = GaiaTransferModel(num_targets=1).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[*] Loaded Model (Richness Target: Mean={richness_mean:.2f}, Std={richness_std:.2f})")
    
    # 2. Load Dataset
    if nc_dir.startswith("s3://"):
        from src.s3_utils import get_s3_fs
        fs = get_s3_fs()
        nc_paths = ["s3://" + f for f in fs.ls(nc_dir) if f.endswith('.nc')]
    else:
        nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
    
    mapping_cache = os.path.join(project_root, "data", "bioscape", 
                                "s3_mapping.json" if nc_dir.startswith("s3") else "local_mapping.json")
    dataset = MultiFlightBioScapeDataset(nc_paths, richness_csv, augment=False, cache_path=mapping_cache)
    
    # Set stats for band norm
    if 'band_mean' in ckpt:
        dataset.set_band_stats(ckpt['band_mean'], ckpt['band_std'])
        print("[*] Band Normalization: Cached stats applied.")
        
    indices = np.arange(len(dataset))
    val_split = b_cfg.get('val_split', 0.2)
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=False)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False)
    
    # 3. Inference
    train_preds, train_targets = run_inference(model, train_loader, richness_mean, richness_std, device)
    val_preds, val_targets = run_inference(model, val_loader, richness_mean, richness_std, device)
    
    # 4. Baseline (Guessing the Average of Training Set)
    # The absolute simplest model is one that always predicts np.mean(train_targets)
    baseline_val_preds = np.full_like(val_targets, fill_value=np.mean(train_targets))
    
    # 5. Compute Metrics
    m_train = compute_metrics(train_targets, train_preds)
    m_val = compute_metrics(val_targets, val_preds)
    m_base = compute_metrics(val_targets, baseline_val_preds)
    
    # Improvement over baseline
    rmse_imp = (1 - (m_val['RMSE'] / m_base['RMSE'])) * 100
    mae_imp = (1 - (m_val['MAE'] / m_base['MAE'])) * 100
    
    print(f"\n[Validation Performance vs Baseline]")
    print(f"  Model RMSE:    {m_val['RMSE']:.4f}  (Improvement: {rmse_imp:+.1f}%)")
    print(f"  Baseline RMSE: {m_base['RMSE']:.4f}  [Guessing Average]")
    print(f"  Final Val R²:  {m_val['R2']:.4f}")
    
    # 6. Generate "Premium" Plots
    report_dir = os.path.join(project_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    
    # --- Plot 1: Scatter (Actual vs Predicted) ---
    ax1 = fig.add_subplot(gs[0, 0:2])
    sns.scatterplot(x=train_targets, y=train_preds, alpha=0.3, color=COLORS['Train'], s=40, label='Train Set (Memorized)', ax=ax1)
    sns.scatterplot(x=val_targets, y=val_preds, alpha=0.9, color=COLORS['Val'], s=80, edgecolors='white', linewidth=0.5, label='Validation Set (Honest)', ax=ax1)
    
    lims = [min(train_targets.min(), val_targets.min()) - 2, max(train_targets.max(), val_targets.max()) + 2]
    ax1.plot(lims, lims, '--k', alpha=0.3, linewidth=2, label='Perfect (1:1)')
    
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_title(f"GAIA Regression: Predicted vs Actual Richness\n(Val R² = {m_val['R2']:.4f})", fontsize=16, pad=15)
    ax1.set_xlabel("Actual Species Richness", fontsize=13)
    ax1.set_ylabel("Predicted Species Richness", fontsize=13)
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    
    # --- Plot 2: Metrics Comparison vs Baseline ---
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['RMSE', 'MAE']
    x = np.arange(len(metrics))
    width = 0.35
    
    model_ms = [m_val['RMSE'], m_val['MAE']]
    base_ms = [m_base['RMSE'], m_base['MAE']]
    
    ax2.bar(x - width/2, base_ms, width, label='Guessing Average', color='#e0e0e0')
    ax2.bar(x + width/2, model_ms, width, label='Gaia Model', color=COLORS['Model'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=12)
    ax2.set_title("Performance vs Baseline", fontsize=14, pad=10)
    ax2.set_ylabel("Error (lower is better)", fontsize=12)
    ax2.legend(loc='upper right')
    
    # Annotate improvement
    ax2.text(x[0], model_ms[0]/2, f"{rmse_imp:.1f}% Better", ha='center', color='white', fontweight='bold')
    ax2.text(x[1], model_ms[1]/2, f"{mae_imp:.1f}% Better", ha='center', color='white', fontweight='bold')
    
    # --- Plot 3: Residual Distribution ---
    ax3 = fig.add_subplot(gs[1, 0])
    residuals = val_preds - val_targets
    sns.histplot(residuals, bins=15, kde=True, color=COLORS['Val'], ax=ax3)
    ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax3.set_title("Validation Residuals (Errors)", fontsize=14)
    ax3.set_xlabel("Error (Predicted - Actual)")
    
    # --- Plot 4: Error Concentration (CDF) ---
    ax4 = fig.add_subplot(gs[1, 1])
    abs_errors = np.sort(np.abs(residuals))
    y = np.arange(1, len(abs_errors) + 1) / len(abs_errors)
    ax4.step(abs_errors, y, color=COLORS['Val'], linewidth=2.5)
    ax4.fill_between(abs_errors, y, color=COLORS['Val'], alpha=0.1)
    
    # Add some thresholds
    for threshold in [2.5, 5.0]:
        perc = np.mean(abs_errors <= threshold) * 100
        ax4.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        ax4.text(threshold+0.2, 0.2, f"{perc:.0f}% sites within {threshold} species", rotation=90, verticalalignment='bottom', alpha=0.6)

    ax4.set_title("Error Concentration (CDF)", fontsize=14)
    ax4.set_xlabel("Absolute Error (species)")
    ax4.set_ylabel("Fraction of Validation Sites")
    ax4.set_ylim([0, 1])
    
    # --- Plot 5: Summary Text ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary_text = (
        f"GAIA EVALUATION SUMMARY\n"
        f"{'─'*25}\n"
        f"Train Sites: {m_train['count']}\n"
        f"Val Sites:   {m_val['count']}\n\n"
        f"Val R²:      {m_val['R2']:.4f}\n"
        f"Val RMSE:    {m_val['RMSE']:.2f}\n"
        f"Val MAE:     {m_val['MAE']:.2f}\n\n"
        f"Baseline:\n"
        f"  RMSE: {m_base['RMSE']:.2f}\n"
        f"  (Model is {rmse_imp:.1f}% better\n"
        f"  than guessing average)"
    )
    ax5.text(0.1, 0.9, summary_text, family='monospace', fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plot_path = os.path.join(report_dir, "gaia_performance_dashboard.png")
    plt.savefig(plot_path, dpi=180, bbox_inches='tight')
    print(f"\n[✔] Performance Dashboard saved to: {plot_path}")
    print(f"[*] Per-site CSV results updated in: {os.path.join(report_dir, 'evaluation_results.csv')}")
    
    # Update CSV with baseline comparison
    results_df = pd.DataFrame({
        'split': ['train']*len(train_targets) + ['val']*len(val_targets),
        'actual': np.concatenate([train_targets, val_targets]),
        'predicted': np.concatenate([train_preds, val_preds]),
        'error': np.concatenate([train_preds - train_targets, val_preds - val_targets]),
        'baseline_guess': np.mean(train_targets)
    })
    results_df.to_csv(os.path.join(report_dir, "evaluation_results.csv"), index=False)
    
    plt.close()
    print("═"*60 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, help="Override data directory")
    args = parser.parse_args()
    evaluate(args.nc_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str, help="Override data directory")
    args = parser.parse_args()
    evaluate(args.nc_dir)

