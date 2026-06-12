import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

def create_kfold_dashboard():
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    results_dir = "reports"
    kfold_summary_path = os.path.join(results_dir, "kfold_results.csv")
    
    # We expect individual fold results named val_results_fold_1.csv etc.
    fold_files = sorted([f for f in os.listdir(results_dir) if f.startswith("val_results_fold_")])
    
    if not fold_files:
        print("[!] No fold results found in reports/. Run training first.")
        return

    # 1. Load Data
    all_data = []
    for f in fold_files:
        fold_id = f.split("_")[-1].replace(".csv", "")
        df = pd.read_csv(os.path.join(results_dir, f))
        df['fold'] = f"Fold {fold_id}"
        all_data.append(df)
    
    full_df = pd.concat(all_data)
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(20, 14))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
    
    # --- PANEL 1: REGRESSION SCATTER ---
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(data=full_df, x='actual', y='predicted', hue='fold', alpha=0.6, s=60, ax=ax1, palette='viridis')
    
    # Add identity line
    lo, hi = full_df['actual'].min(), full_df['actual'].max()
    ax1.plot([lo, hi], [lo, hi], '--', color='grey', lw=2, label='Perfect Fit')
    
    # Overall Metrics
    r2 = r2_score(full_df['actual'], full_df['predicted'])
    rmse = np.sqrt(mean_squared_error(full_df['actual'], full_df['predicted']))
    ax1.text(0.05, 0.95, f"Overall R²: {r2:.3f}\nOverall RMSE: {rmse:.2f}", 
             transform=ax1.transAxes, verticalalignment='top', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title("Pooled Regression Analysis (All Folds)", fontweight='bold', pad=15)
    ax1.set_xlabel("Actual Species Richness")
    ax1.set_ylabel("Predicted Species Richness")
    ax1.legend(loc='lower right', fontsize=10)

    # --- PANEL 2: METRICS BY FOLD ---
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists(kfold_summary_path):
        summary_df = pd.read_csv(kfold_summary_path)
        summary_df['fold_name'] = [f"Fold {i+1}" for i in range(len(summary_df))]
        
        ax2b = ax2.twinx()
        sns.barplot(data=summary_df, x='fold_name', y='r2', ax=ax2, color='#4A90E2', alpha=0.7, label='R²')
        sns.lineplot(data=summary_df, x='fold_name', y='rmse', ax=ax2b, color='#D0021B', marker='o', lw=3, label='RMSE')
        
        ax2.set_ylabel("R² Score", color='#4A90E2', fontweight='bold')
        ax2b.set_ylabel("RMSE", color='#D0021B', fontweight='bold')
        ax2.set_title("Cross-Validation Stability", fontweight='bold', pad=15)
        ax2.set_ylim(0, 1.0)
    else:
        ax2.text(0.5, 0.5, "kfold_results.csv missing", ha='center')

    # --- PANEL 3: RESIDUAL DISTRIBUTION ---
    ax3 = fig.add_subplot(gs[1, 0])
    full_df['residual'] = full_df['actual'] - full_df['predicted']
    sns.histplot(data=full_df, x='residual', kde=True, color='#50E3C2', ax=ax3)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_title("Prediction Residuals (Actual - Predicted)", fontweight='bold', pad=15)
    ax3.set_xlabel("Residual Value")

    # --- PANEL 4: PERFORMANCE BY BIN ---
    ax4 = fig.add_subplot(gs[1, 1])
    full_df['actual_bin'] = pd.qcut(full_df['actual'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    sns.boxplot(data=full_df, x='actual_bin', y='residual', palette='Pastel1', ax=ax4)
    ax4.axhline(0, color='red', linestyle='--')
    ax4.set_title("Model Bias by Richness Level", fontweight='bold', pad=15)
    ax4.set_xlabel("Biodiversity Group")
    ax4.set_ylabel("Residual Error")

    # Global Title
    plt.suptitle("GAIA FOUNDATION MODEL: K-FOLD PERFORMANCE DASHBOARD", 
                 fontsize=24, fontweight='bold', y=0.98, color='#2C3E50')
    
    # Branding
    ax4.text(1.0, -0.15, "Architecture: MS-ViT (Masked Spatial-Spectral)\nDataset: BioSCape Mosaic (30m)", 
             transform=ax4.transAxes, ha='right', fontsize=12, alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = os.path.join(results_dir, "kfold_performance_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Dashboard saved to: {out_path}")

if __name__ == "__main__":
    create_kfold_dashboard()
