# Gaia: Ideas for Improvement

When working with specialized geospatial data like BioSCape, we often face the "Labels Poor but Pixels Rich" dilemma: we have hundreds of gigabytes of imagery but only a few hundred ground-truth labels. 

To exceed an R² of 0 with the current ~450 sites, we should shift from pure Deep Learning to **Knowledge-Guided Learning**.

## 1. Synthetic Site Generation (Spatial Jitter)
**The Idea:** For each site, instead of extracting exactly one patch, take 10–20 random crops within a 50m–100m radius and assign them the same richness label.
**Why:** This multiplies the dataset size by 10x and teaches the model that richness is a property of the local ecosystem/neighborhood, not a single pixel. It also makes the model robust to GPS coordinate noise.

## 2. Multi-Task "Pre-training" (The MAE Path)
**The Idea:** Run Masked Autoencoder (MAE) pre-training on **all** 535 synced flightlines before starting the richness regression training.
**Why:** This forces the Transformer to learn the "language" of the BioSCape landscape (distinguishing forest, shrubs, soil, and water) using millions of unlabeled pixels. A model with a "PhD in BioSCape Geography" will find the richness task much easier.

## 3. Spectral Index "Hints" (Feature Engineering)
**The Idea:** Calculate standard vegetation indices (NDVI, EVI, PRI, etc.) and concatenate them to the 200 raw bands.
**Why:** It gives the model a physical shortcut. Identifying "Chlorophyll concentration" or "Water stress" explicitly helps the model converge when labels are too sparse to learn these relationships from raw reflectance.

## 4. Bayesian / Uncertainty Head
**The Idea:** Modify the regression head to output a distribution (Mean + Variance) instead of a single scalar.
**Why:** This allows the model to "down-weight" noisy or ambiguous sites during training, leading to more stable gradients and better generalization on the validation set.

## 5. Architectural "Dumbing Down"
**The Idea:** Freeze the Transformer backbone and use a much simpler, low-parameter linear head.
**Why:** A 1.8M parameter model can easily "memorize" 450 images. By restricting the "brain space" of the regressor, we force it to learn general ecological patterns instead of individual site images.

---

## Note on Patch Size and Context
The model currently trains on a **16x16 patch** (at 30m resolution).

*   **Spatial Coverage:** A 16x16 patch at 30m covers **480m x 480m** (approx. 23 hectares).
*   **Importance of Size:** Vision Transformers rely on spatial context to identify textures and structures (e.g., the difference between a uniform plantation and a diverse natural forest). 
*   **Scaling:** Increasing the patch size (e.g., to 32x32) allows the model to see even more of the surrounding environment, which is often critical for predicting biodiversity—since species richness is driven by landscape-scale diversity, not just the plants under a single pixel. However, larger patches increase GPU memory usage significantly.
