import os
import sys

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["GDAL_NUM_THREADS"] = "4"

import random
import yaml
import optuna
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the production script since we are doing BioSCape Regression!
from src.train_production import MultiFlightBioScapeDataset, GaiaTransferModel
try:
    from src.s3_utils import get_s3_fs 
except ImportError:
    get_s3_fs = None

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(project_root, "configs", "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    b_cfg = config.get('bioscape', {})
    
    # Optuna suggestions (The "Knobs")
    lr = trial.suggest_float("lr", 1e-3, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    backbone_lr_mult = trial.suggest_float("backbone_lr_mult", 0.05, 0.5, log=True)
    
    # Static parameters based on user feedback
    max_epochs = 200
    patience = 25
    unfreeze_epoch = trial.suggest_int("unfreeze_epoch", 5, 50)

    wandb.init(project="gaia_optuna_tuning", config={
        "lr": lr, 
        "weight_decay": weight_decay,
        "backbone_lr_mult": backbone_lr_mult,
        "max_epochs": max_epochs,
        "unfreeze_epoch": unfreeze_epoch,
        "task": "regression",
        "resolution": "30m",
        "bands": 200
    }, reinit=True, name=f"trial_{trial.number}")

    try:
        # Set seeds for reproducibility
        seed = config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 1. Load Dataset
        nc_dir = os.path.join(project_root, b_cfg.get('nc_dir_local', 'data/'))
        if b_cfg.get('use_s3', False):
            nc_dir = b_cfg.get('nc_dir_s3', nc_dir)
            
        richness_csv = os.path.join(project_root, b_cfg.get('richness_csv', ''))
        
        if nc_dir.startswith("s3://"):
            if get_s3_fs is None:
                raise ImportError("S3 requested but src.s3_utils.get_s3_fs is unavailable")
            fs = get_s3_fs()
            nc_paths = ["s3://" + f for f in fs.ls(nc_dir) if f.endswith('.nc')]
        else:
            nc_paths = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith('.nc')]
            
        if not nc_paths:
            raise FileNotFoundError(f"No .nc files found in {nc_dir}")
            
        patch_size = b_cfg.get('patch_size', 16)
            
        cache_suffix = f"_p{patch_size}.json"
        cache_name = ("s3_mapping" if nc_dir.startswith("s3") else "local_mapping") + cache_suffix
        mapping_cache = os.path.join(project_root, "data", "bioscape", cache_name)
        
        # Note: For tuning efficiency we assume mappings and band stats are already cached by train_production.py
        dataset_train = MultiFlightBioScapeDataset(nc_paths, richness_csv, patch_size=patch_size, augment=True, cache_path=mapping_cache)
        dataset_val = MultiFlightBioScapeDataset(nc_paths, richness_csv, patch_size=patch_size, augment=False, cache_path=mapping_cache)
        
        band_stats_name = ("s3_band_stats" if nc_dir.startswith("s3") else "local_band_stats") + cache_suffix
        band_stats_cache = os.path.join(project_root, "data", "bioscape", band_stats_name)
        dataset_train.compute_band_stats(cache_path=band_stats_cache)
        dataset_val.compute_band_stats(cache_path=band_stats_cache)
        
        # Prevent multi-processing issues with h5py
        for ds in dataset_train.dataset_cache.values():
            ds.close()
        dataset_train.dataset_cache.clear()
        
        for ds in dataset_val.dataset_cache.values():
            ds.close()
        dataset_val.dataset_cache.clear()

        # 2. Split and Loaders
        indices = np.arange(len(dataset_train))
        val_split = b_cfg.get('val_split', 0.2)
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_split, 
            random_state=seed, 
            shuffle=True
        )

        # Get Normalization Targets ONLY from train_idx to prevent mild leakage
        train_richness = np.array([dataset_train.mappings[i][3] for i in train_idx], dtype=np.float32)
        richness_mean = float(train_richness.mean())
        richness_std = float(train_richness.std()) + 1e-6
        
        b_size = b_cfg.get('batch_size', 16)
        num_workers = b_cfg.get('num_workers', 4) if os.name != 'nt' or nc_dir.startswith("s3") else 0
        train_loader = DataLoader(Subset(dataset_train, train_idx), batch_size=b_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True, 
                                  prefetch_factor=4 if num_workers > 0 else None, 
                                  persistent_workers=True if num_workers > 0 else False)
        val_loader = DataLoader(Subset(dataset_val, val_idx), batch_size=b_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, 
                                prefetch_factor=4 if num_workers > 0 else None, 
                                persistent_workers=True if num_workers > 0 else False)

        # 3. Model & Optimizer
        model = GaiaTransferModel(num_targets=1, patch_size=patch_size).to(device)
        foundation_ckpt = os.path.join(project_root, "checkpoints", "pretrained_ViTSpatialSpectral_200ep_enmap.pth")
        if os.path.exists(foundation_ckpt):
            model.load_foundation_weights(foundation_ckpt, device)
            
        # Baseline: Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.encoder.mlp_head.parameters():
            param.requires_grad = True

        if torch.cuda.device_count() > 1:
            print(f"[*] utilizing {torch.cuda.device_count()} GPUs via DataParallel")
            model = nn.DataParallel(model)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        
        best_val_r2 = -float('inf')
        patience_counter = 0

        # 4. Training Loop
        epochs_pbar = tqdm(range(1, max_epochs + 1), desc=f"Trial {trial.number}")
        for epoch in epochs_pbar:
            # Get true model avoiding DataParallel wrapper logic wrapper
            core_model = model.module if isinstance(model, nn.DataParallel) else model

            # PROGRESSIVE UNFREEZING KNOB
            if epoch == unfreeze_epoch:
                for param in core_model.encoder.parameters():
                    param.requires_grad = True
                    
                head_params = []
                backbone_params = []
                
                for name, p in core_model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if name.startswith("encoder.") and "mlp_head" not in name:
                        backbone_params.append(p)
                    else:
                        head_params.append(p)
                
                optimizer = optim.AdamW(
                    [
                        {"params": head_params, "lr": lr},
                        {"params": backbone_params, "lr": lr * backbone_lr_mult},
                    ],
                    weight_decay=weight_decay,
                )

            model.train()
            train_loss = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels_norm = ((labels - richness_mean) / richness_std).to(device).float().view(-1, 1)
                
                optimizer.zero_grad()
                preds = model(images).view(-1, 1)
                loss = criterion(preds, labels_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
            avg_loss = train_loss / len(train_loader)
            
            # Validation Eval
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    preds_norm = model(images.to(device)).view(-1, 1)
                    preds_real = preds_norm.cpu().numpy().flatten() * richness_std + richness_mean
                    val_preds.extend(preds_real)
                    val_targets.extend(labels.cpu().numpy().astype(np.float32).flatten())
                    
            epoch_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0
            mae = mean_absolute_error(val_targets, val_preds) if len(val_targets) > 1 else 0
            rmse = np.sqrt(mean_squared_error(val_targets, val_preds)) if len(val_targets) > 1 else 0
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            just_unfroze = 1 if epoch == unfreeze_epoch else 0
            encoder_frozen = 1 if epoch < unfreeze_epoch else 0

            head_lr = optimizer.param_groups[0]["lr"]
            backbone_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0

            min_delta = 1e-4
            prev_best_val_r2 = best_val_r2
            best_val_r2 = max(best_val_r2, epoch_r2)

            if epoch_r2 > prev_best_val_r2 + min_delta:
                patience_counter = 0
            else:
                patience_counter += 1

            wandb.log({
                "epoch": epoch,
                "val_r2": epoch_r2,
                "val_mae": mae,
                "val_rmse": rmse,
                "train_loss": avg_loss,
                "head_lr": head_lr,
                "backbone_lr": backbone_lr,
                "unfreeze_epoch": unfreeze_epoch,
                "encoder_frozen": encoder_frozen,
                "just_unfroze": just_unfroze,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "best_val_r2_so_far": best_val_r2,
                "patience_counter": patience_counter,
            }, step=epoch)
            
                
            # Standard Early Stopping
            if epoch >= unfreeze_epoch and patience_counter >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break

        wandb.summary["best_val_r2"] = best_val_r2
        wandb.summary["final_unfreeze_epoch"] = unfreeze_epoch
        wandb.summary["final_max_epochs"] = max_epochs
        wandb.summary["final_patch_size"] = patch_size
        wandb.summary["stopped_epoch"] = epoch
        wandb.summary["patience"] = patience

        return best_val_r2
        
    finally:
        wandb.finish()

if __name__ == "__main__":
    print("[*] Starting Optuna Regression Sweep Engine...")
    
    os.makedirs("reports", exist_ok=True)
    storage_name = "sqlite:///reports/optuna_study.db"
    
    study = optuna.create_study(
        study_name="gaia_hyperparameter_optimization",
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=100)

    print("================================")
    print("Best trial:")
    trial = study.best_trial
    print(f"  R2 Score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
