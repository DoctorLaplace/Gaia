# Gaia: Biodiversity Transformation Engine

Gaia is a hyperspectral foundation model pipeline designed to predict Species Richness across the Greater Cape Floristic Region (South Africa) using NASA BioSCape data.

## QuickStart

### 1. Prerequisites
- **Python 3.10+** (Recommend a virtual env)
- **CUDA GPU** with at least 8GB VRAM.
- **NASA Earthdata Account**: [Register here](https://urs.earthdata.nasa.gov/).

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

### 3. Data Acquisition (Smart Sync)
Downloads AVIRIS-NG granules, applies Dr. Clark's Bad Band List (BBL), and optionally downsamples.
Data is automatically organized by resolution: `data/bioscape/30m/` and `data/bioscape/5m/`.

#### Option A: Targeted Sync (Recommended, ~535 labeled granules)
```bash
# 1. Generate the labeled inventory
python src/generate_labeled_inventory.py

# 2. Download 30m (default, ~35MB each, fast local training)
python src/smart_sync.py --local

# 3. Download 5m (original resolution, ~1.3GB each, for supercomputer)
python src/smart_sync.py --local --res 5
```

#### Option B: Full Dataset Sync (3,648 granules, 500GB+)
```bash
python src/smart_sync.py --inventory full_inventory.txt --local --res 30
```

#### Option C: S3 Upload (for NRP Nautilus)
```bash
python src/smart_sync.py --res 30
```

### 4. Model Training
Gaia uses a Masked Spatial-Spectral Transformer (SST) with EnMAP foundation weights.

```bash
# Train on 30m data (default, from config)
python src/train_production.py --epochs 250

# Train on 5m data (override directory)
python src/train_production.py --nc_dir data/bioscape/5m --epochs 250

# Quick test run (2 granules, 1 epoch)
python src/train_production.py --test-run
```

### 5. Evaluation
Evaluates the best checkpoint on a held-out validation set (10%, same split as training).

```bash
python src/evaluate.py --nc_dir data/bioscape/30m
```

Outputs:
- **Performance Dashboard**: `reports/gaia_performance_dashboard.png` (Styled scatter + baseline comparison).
- **CSV Results**: `reports/evaluation_results.csv` (Per-site predictions vs. Mean Baseline).
- Terminal summary of R-squared improvement over the "Mean Predictor".

### 6. K-Fold Cross-Validation (Rigorous)
Performs Group K-Fold validation using flightlines to ensure the model generalizes to entirely new geographic regions. 

```bash
# Run 5-fold CV (Takes ~100 minutes)
python src/train_kfold.py --folds 5 --freeze
```

This script computes the **Mean R² ± Std Dev** across all folds, providing a much more robust estimate of performance than a single held-out split. Results are saved to `reports/kfold_results.csv`.

### 7. Interactive Geospatial Visualizer
```bash
# 1. Update mapping from all local granules
python viz/regenerate_mapping.py

# 2. Export patches (with robust per-band normalization)
python viz/export_viz_data.py

# 3. Start the server (provides a clickable link)
python src/serve_viz.py
```
Open: [http://localhost:8000/viz/index.html](http://localhost:8000/viz/index.html)

---

## Remote Training (NRP Nautilus)

### Accessing the Workspace
```bash
coder ssh nasa-gaia
```

### S3 Storage Management (Ceph)

#### Local (Windows/Dev):
```bash
python src/s3_ls.py              # List all buckets
python src/s3_ls.py gaia-datasets  # List bucket contents
```

#### NRP Nautilus (Linux):
- **List S3**: `s3cmd ls s3://your-bucket-name/`
- **Upload**: `s3cmd put FILE s3://BUCKET/`
- **Download**: `s3cmd get s3://BUCKET/FILE .`
- **Rclone Sync**: `rclone sync /local/path nautilus-s3:bucket-name`

#### Endpoints
- **Internal (High Performance)**: `http://rook-ceph-rgw-nautiluss3.rook`
- **External (Local Access)**: `https://s3-west.nrp-nautilus.io`

---

## Project Structure
```
data/bioscape/
  30m/             Downsampled + BBL-cleaned granules (local training)
  5m/              Full-resolution granules (supercomputer)
  *.csv, *.npy     Shared labels and normalization stats
src/
  smart_sync.py              Targeted downloader with BBL + downsampling
  train_production.py        Fine-tuning with foundation weights
  train_kfold.py             5-Fold Group validation by flightline
  evaluate.py                Premium dashboard with baseline comparison
  bioscape_dataset.py        NetCDF dataset loader (5m and 30m compatible)
  generate_labeled_inventory.py   CMR spatial query for richness sites
  serve_viz.py               Visualizer server with clean CLI links
  s3_ls.py                   Local S3 bucket browser
configs/config.yaml          All training hyperparameters and paths
checkpoints/                 Foundation + fine-tuned model weights
reports/
  kfold_results.csv          Cross-validation performance table
  gaia_performance_dashboard.png  Visual evaluation summary
viz/                         Leaflet geospatial explorer
  regenerate_mapping.py      Site mapping re-scanner (syncs local data)
  export_viz_data.py         Robust patch extractor for web UI
  data/                      Binary patches and metadata
```
