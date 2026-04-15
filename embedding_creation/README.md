# AVIRIS-NG Native Foundation Model Pipeline

Self-supervised pre-training on 370 native spectral bands to build an AVIRIS-specific embedding space, then fine-tuning for richness regression.

## How It Works

1. **Pre-training (SimMIM)**: The model sees thousands of unlabeled hyperspectral patches. 60% of tokens are masked, and the model learns to reconstruct the missing spectral values. This forces the transformer to build an internal embedding space that understands spectral-spatial relationships.

2. **Fine-tuning**: The pre-trained encoder is frozen and a small regression head is trained on top using labeled richness data.

## Data Flow

```
NASA EarthData (BioSCape AVIRIS-NG L2B)
    |
    | sync_full_archive.py --limit N
    | Downloads, BBL filters (425 -> 373 bands), downsamples to target res
    v
embedding_creation/data/{res}m/*.nc   <-- NEW SSL-only granules go HERE
    +
data/bioscape/{res}m/*.nc             <-- existing labeled granules (READ-ONLY)
    |
    | pretrain.py --res {res}
    | Reads from BOTH dirs, trims to 370 bands, extracts 16x16 patches
    | Masks 60% of tokens, reconstructs via L1 loss
    v
embedding_creation/checkpoints/
    pretrained_aviris_best.pth         <-- THE EMBEDDING MODEL
    pretrain_band_stats.json           <-- per-band normalization
    |
    | finetune.py --res {res}
    | Reads ONLY from data/bioscape/{res}m/ (labeled sites)
    v
embedding_creation/checkpoints/
    finetuned_aviris_best.pth          <-- Final richness model
```

## Where Things Live

| What | Path | Notes |
|---|---|---|
| **Existing labeled data** | `data/bioscape/{res}m/` | Used by existing Gaia pipeline. **Read-only.** |
| **SSL-only data** | `embedding_creation/data/{res}m/` | Additional unlabeled granules for pre-training |
| **Richness labels** | `data/bioscape/western_cape_site_embeddings.csv` | Biodiversity sites with species richness |
| **Config** | `embedding_creation/config.yaml` | All hyperparameters |
| **Pre-trained encoder** | `embedding_creation/checkpoints/pretrained_aviris_best.pth` | The learned embedding space |
| **Band statistics** | `embedding_creation/checkpoints/pretrain_band_stats.json` | Per-band mean/std |
| **Fine-tuned model** | `embedding_creation/checkpoints/finetuned_aviris_best.pth` | Final richness model |

### Data Isolation

```
data/bioscape/30m/                  <-- existing Gaia pipeline (read-only)
data/bioscape/5m/                   <-- existing Gaia pipeline (read-only)
embedding_creation/data/30m/        <-- SSL archive, 30m
embedding_creation/data/5m/         <-- SSL archive, 5m
```

- `sync_full_archive.py` downloads into `embedding_creation/data/{res}m/` only
- The existing Gaia data directories are never written to
- `pretrain.py` reads from both directories for maximum training data
- `finetune.py` reads only from the labeled directory

---

## Commands

All commands run from `E:\Git Repositories\Laboratory\Models\Gaia`.

### Test Run (verify everything works, ~10 seconds)

```bash
python -m embedding_creation.pretrain --test-run
python -m embedding_creation.finetune --test-run
```

### Full Run at 30m (start here)

```bash
# Step 1: Build the embedding space (~1-2 hours on RTX 3060)
python -m embedding_creation.pretrain

# Step 2: Fine-tune on richness
python -m embedding_creation.finetune
```

### Full Run at 5m

```bash
# Step 0: Download 5m data (much larger files)
python -m embedding_creation.sync_full_archive --res 5 --limit 10    # test first
python -m embedding_creation.sync_full_archive --res 5                # all

# Step 1: Build embedding space at 5m
python -m embedding_creation.pretrain --res 5

# Step 2: Fine-tune at 5m
python -m embedding_creation.finetune --res 5
```

### Download More SSL Data (optional, for better embeddings)

```bash
# Download 10 more granules at 30m (test connectivity)
python -m embedding_creation.sync_full_archive --limit 10

# Download everything at 30m (~110 GB new data)
python -m embedding_creation.sync_full_archive

# Download at 5m (~750 GB new data)
python -m embedding_creation.sync_full_archive --res 5
```

---

## CLI Reference

### pretrain.py

```bash
python -m embedding_creation.pretrain [options]
```

| Flag | Effect |
|---|---|
| `--test-run` | 2 epochs, 5 granules, batch 2 |
| `--res N` | Resolution in meters (default: 30) |
| `--epochs N` | Override epoch count |
| `--batch-size N` | Override batch size |
| `--max-granules N` | Limit to first N granules |

### finetune.py

```bash
python -m embedding_creation.finetune [options]
```

| Flag | Effect |
|---|---|
| `--test-run` | 3 epochs, 5 granules, batch 2 |
| `--res N` | Resolution in meters (default: 30) |
| `--epochs N` | Override epoch count |
| `--batch-size N` | Override batch size |
| `--max-granules N` | Limit granules |
| `--no-freeze` | Train all parameters (not just head) |

### sync_full_archive.py

```bash
python -m embedding_creation.sync_full_archive [options]
```

| Flag | Effect |
|---|---|
| `--limit N` | Only download first N new granules |
| `--res N` | Target resolution in meters (default: 30) |
| `--inventory FILE` | Inventory file (default: full_inventory.txt) |

---

## Architecture

Matches Scheibenreif et al. (CVPRW 2023):

```
dim=96, depth=4, heads=8, spectral_patch_size=10, spatial_patch_size=1
Factorized attention: spatial (256 tokens) then spectral (37 tokens)
Blockwise patch embedding + spectral positional encoding
Model: ~1.9M parameters, ~3-4 GB VRAM with batch 4 + FP16
```

Our model uses 370 native AVIRIS-NG bands (37 spectral groups) instead of 200 resampled EnMAP bands (20 groups).
