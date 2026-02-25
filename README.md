# Beyond Visible Spectrum: AI for Agriculture 2026

## Overview
This repository contains a high-performance deep learning pipeline for **Crop Disease Classification** using **multi-spectral Sentinel-2 satellite imagery**.  
The solution is optimized for the **Kaggle environment** using **dual Tesla T4 GPUs** and targets an accuracy of **0.90+**, with scalability toward **0.95+**.

The approach combines **self-supervised learning**, **transformer-based architectures**, and **spectral feature engineering** to overcome limited labeled agricultural data.

---

## Strategy and Pipeline

The pipeline follows a **three-stage learning and inference strategy** to maximize the use of both labeled and unlabeled satellite imagery.

### 1. Self-Supervised Pretraining (Masked Autoencoder – MAE)

Satellite data is abundant, but labels are scarce.  
A **Masked Autoencoder (MAE)** is used to learn robust spatial–spectral representations.

- **Input**
  - 12 raw Sentinel-2 bands
  - 4 custom vegetation indices  
  - Total: 16 spectral channels

- **Task**
  - Reconstruct masked image patches  
  - Masking ratio: 75%

- **Architecture**
  - Vision Transformer (ViT-Small)
  - Optimized for memory efficiency

---

### 2. Fine-Tuning Triple Ensemble

Three complementary architectures are fine-tuned to capture diverse feature hierarchies:

- **ViT-Base**
  - Initialized with SSL-pretrained weights
  - Strong global context modeling

- **Swin-Tiny**
  - Hierarchical transformer with shifted windows
  - Local-to-global spatial feature extraction

- **ConvNeXt-Small**
  - Modern convolutional architecture
  - Strong inductive bias for structural features

---

### 3. Advanced Inference

To improve robustness and generalization:

- **Weighted Soft Voting**
  - Ensemble weights optimized using the Nelder–Mead algorithm

- **Test-Time Augmentation (TTA)**
  - 8-way TTA  
  - 4 rotations × 2 flips

- **Layer-wise Learning Rate Decay (LLRD)**
  - Slower updates for deeper layers
  - Preserves pretrained representations

---

## Spectral Engineering

Custom vegetation indices are computed to capture physiological stress signals beyond RGB and NIR bands.

| Index | Purpose | Target Diseases |
|------|--------|----------------|
| NDVI | General vegetation health | Overall biomass |
| NDRE | Red-edge chlorophyll sensitivity | Early disease detection |
| PSRI | Plant senescence | Rust detection |
| SWIR Ratio | Moisture content | Drought and stem stress |

---

## Configuration

Key hyperparameters used in `agri2-1.ipynb`:

- Image Size: 64 × 64 (spectral patches)
- Bands: 12 Sentinel-2 bands (B1–B12)
- Classes:
  - Aphid
  - Rust
  - RPH (Rice Planthopper)
  - Blast
- Augmentations:
  - MixUp
  - Spectral Dropout
  - Spatial flips and rotations

---

## Performance Optimization

To push accuracy from **0.90 to 0.95+**, consider:

- **Pseudo-Labeling**
  - Use ensemble predictions to label the `s2a` unlabeled dataset

- **5-Fold Cross-Validation**
  - Train and ensemble all five folds instead of only fold 0

- **Higher Resolution Inputs**
  - Increase `IMG_SIZE` to 128  
  - Requires careful VRAM and batch-size management

---

## Requirements

- torch  
- torchvision  
- timm  
- rasterio (for `.tif` processing)  
- einops  
- scikit-learn
