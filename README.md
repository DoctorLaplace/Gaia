# Gaia: A Masked Vision Transformer for Species Richness Prediction from Hyperspectral Imagery

Gaia is a masked spatial-spectral vision transformer fine-tuned on EnMAP foundation weights to predict animal species richness (birds, frogs, and insects) from AVIRIS-NG hyperspectral imagery across the Greater Cape Floristic Region (South Africa) using NASA BioSCape data.

## QuickStart

### 1. Prerequisites
- **Python 3.10+** (Recommend a virtual env)
- **CUDA GPU** with at least 8GB VRAM.

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

### 3. Data Acquisition 
_For the old BioSCape data download instructions see README_OLD.md_
Download the three EAGLE resampled archives to ~/Downloads (or directory of your choice).

```bash
python scratch/extract_eagle_data.py
```
or
```bash
python scratch/extract_eagle_data.py --zip-dir different_directory_path
```

To upload to a remote server (NRP) there are two options.

#### A. Upload full zips then extract:
Simpler since you don't need to confirm upload multiple times, may cause issues if download breaks before its finished. You must also ensure you have space for the full zip file and the extracted data (so ~2x storage requirement). Reccomendation: try zip upload first then proceed to option B if that doesn't work. 

Once zip files are uploaded follow extraction as described previously.

#### B. Upload uncompressed files:
The extracted files will live in Gaia/data/eagle, directly upload this folder to its equivalent location. Once complete ensure all files uploaded correctly:

```bash
python scratch/verify_dataset_integrity.py
```

If any files are corrupted download corrupted_files.txt to your local Gaia/data/eagle folder. Then on your local machine:

```bash
python scratch/prepare_repair.py
```

A folder titled repair will be created in eagle containing all files to be reuploaded. Directly upload those to their corresponding folder (either 30m_native or 30m_10nm).

Repeat as needed until the whole dataset verifies cleanly. 

### 4. Model Training
Gaia uses a Masked Spatial-Spectral Transformer with EnMAP foundation weights
(`checkpoints/pretrained_ViTSpatialSpectral_200ep_enmap.pth`), fine-tuned on the
EAGLE tiles with a small MLP regression head (`GaiaTransferModel` in
`src/model_transfer.py`). The train/validation split holds out whole clusters
(geographic site groups from `CLUSTER_ID`) so validation always measures
generalization to unseen regions.

```bash
# Train the production checkpoint (10nm mode, hold out 10 clusters for validation)
python src/train_production_cluster_strata_eagle.py --mode 10nm --epochs 250

# Quick smoke test (2 tiles, 2 epochs, 1 held-out cluster)
python src/train_production_cluster_strata_eagle.py --test-run
```

Key flags: `--mode {native,10nm}`, `--leave-out N` (clusters held out, default 10),
`--freeze` / `--no-freeze` (encoder frozen by default), `--unfreeze-epoch N`
(progressive unfreeze), `--seed`, `--patience`.

The best epoch is written to `checkpoints/gaia_eagle_best_strata.pth` (bundles the
richness mean/std and per-band normalization stats needed for inference).

### 5. K-Fold Cross-Validation
Each outer fold holds out complete clusters and fits one model. Input per-band
mean/std and target mean/std are fitted only on that fold's training sites and
applied unchanged to validation sites. The best weights are selected by validation
RMSE; there is no inner split or fresh refit. **Outer validation is used for epoch
selection, so these scores are not independent nested-test estimates.**
Cluster `-1` remains training-only; every eligible site receives exactly one OOF
prediction. Grouping is not itself proof of geographic independence.

```bash
# 5-fold cluster-stratified CV
python src/train_kfold_cluster_strata_eagle.py --k 5 --mode 10nm
```

Reports two headline numbers:
- **Mean R² ± Std Dev** across defined folds, with the number used. Singleton and
  constant-target folds have missing R², but retain their RMSE/MAE and predictions.
- **Pooled R²** — all eligible held-out predictions, including singleton folds.

Every invocation creates an exclusive
`reports/<UTC timestamp>-eagle-<mode>-k<effective_k>/` directory:
- `kfold_cluster_strata_results_eagle_<mode>.csv` — outer metrics, selected epoch,
  and held-out clusters.
- `kfold_pooled_oof_eagle_<mode>.csv` — site, actual, predicted.
- `kfold_predictions_matrix_eagle_<mode>.csv` — dense per-fold predictions.
- `kfold_plots/fold_*_scatter.png` and `kfold_plots/pooled_oof_scatter_eagle_<mode>.png`.
- `run.json` — strict completed-run marker with full-precision metrics, exact
  invocation, effective k, seed, spectral mapping, elapsed time, input/source hashes,
  `exit_status: 0` and `evaluation_protocol: single_fit_outer_validation`.
  It is absent for incomplete runs.

The normal mapping cache must already exist and be audited. `--test-run` uses two
folds/two epochs and all sites from the first up to four sorted eligible clusters
plus noise, requiring at least two eligible groups, with a `-smoke` directory
suffix; it does not regenerate a smaller mapping. Audit flags for partial or
missing imagery do not authorize dropping cached sites: all 415 eligible sites,
including the two all-NaN patches, are retained; the two noise sites train only.
If a short schedule gives OneCycleLR an exactly zero-length warmup phase, that
phase alone is omitted; ordinary initial/unfreeze schedules remain unchanged.
Use `--k` equal to the live eligible cluster count for leave-one-cluster-out.

### 6. Inference / Heatmap
Slides a trained model over every tile in a directory and exports predictions as a
point layer for QGIS.

```bash
python -m src.generate_heatmap --checkpoint checkpoints/gaia_eagle_best_strata.pth --mode 10nm --format gpkg
```

Writes `reports/heatmap_<mode>.geojson` (or `.gpkg` with `--format gpkg`).

EAGLE inputs use `enmap_200_v1`: interpolation to the foundation checkpoint's 200
retained EnMAP wavelengths, not a uniform 400–2450 nm grid. Source TIFFs require
complete, finite, positive, strictly increasing wavelength metadata. Atmospheric
gap channels are omitted; the obsolete EAGLE mask switch has been removed.
This aligns channel wavelengths, not spectral response functions or pretraining
normalization; training-only per-band z-scores remain unchanged.

Heatmap inference requires `input_spectral_mapping: enmap_200_v1` in the supervised
checkpoint. Old unmarked/uniform-grid checkpoints are incompatible and rejected
before output creation; retrain rather than merely adding metadata. Existing
checkpoints remain untouched, and this restriction does not reject the foundation
checkpoint used for training initialization.


## Project Structure
```
data/
  eagle/
    30m_native/    Native wavelength resolution EAGLE mosaic tiles (.tif)
    30m_10nm/      10 nm FWHM resampled EAGLE mosaic tiles (.tif)
  bioscape/
    western_cape_site_embeddings.csv   Per-site species-richness labels
    biosoundscape_site_metadata.csv    Site -> CLUSTER_ID (geographic groups for stratified splits)
src/
  train_production_cluster_strata_eagle.py   Fine-tune on EAGLE; cluster-held-out split -> production checkpoint
  train_kfold_cluster_strata_eagle.py        Cluster-stratified K-fold CV (mean + pooled R²)
  generate_heatmap.py                        Sweep a trained model over tiles -> richness point layer for QGIS
  model_transfer.py                          GaiaTransferModel: pretrained ViT encoder + MLP regression head
  vit_spatial_spectral.py                    Masked spatial-spectral ViT backbone
  pos_embed.py                               Sin-cos positional embeddings for the backbone
  eagle_dataset.py                           EAGLE GeoTIFF dataset (tile<->site mapping, band norm, clusters)
  legacy/                                    Superseded BioSCape / S3 pipeline (kept for reference, not maintained)
  obsolete/                                  Dead one-off scripts
scratch/
  extract_eagle_data.py        Unpack downloaded EAGLE archives into data/eagle/
  verify_dataset_integrity.py  Check an uploaded dataset for corrupt tiles
  prepare_repair.py            Stage corrupt tiles for re-upload
configs/config.yaml            All training hyperparameters and paths (bioscape: block)
checkpoints/                   Foundation + fine-tuned model weights
reports/
  kfold_cluster_strata_results_eagle_*.csv   Per-fold CV metrics
  kfold_pooled_oof_eagle_*.csv               Pooled out-of-fold predictions
  kfold_plots/                               Per-fold + pooled scatter plots
  heatmap_*.geojson / .gpkg                  Richness prediction layers
viz/                           Leaflet geospatial explorer (BioSCape-era; not updated for EAGLE)
```
