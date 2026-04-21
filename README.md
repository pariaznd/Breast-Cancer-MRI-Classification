# Breast-Cancer-MRI-Classification

This repository contains our **TDT4265 mini-project** on breast cancer classification from **DCE-MRI** in the **ODELIA 2025 challenge**.

## Overview

Breast cancer classification from MRI is challenging because the data are:
- high-dimensional 3D volumes,
- strongly imbalanced across classes,
- and affected by **domain shift** across hospitals.

Our project investigates both **3D volumetric models** and **2D MIP-based models**, and uses **logit ensembling** to improve generalization on unseen centers.

## Method

Our final submission combines:
- a **3D ResNet18** trained on volumetric MRI inputs,
- a **2D ResNet50** trained on **Maximum Intensity Projection (MIP)** images,
- **test-time augmentation (TTA)**,
- and **logit-space ensembling**.

### Final models
- **3D ResNet18**
  - input modalities: `Pre`, `Post_1`, `Post_2`
  - isotropic resampling
  - background masking
  - light augmentation
- **2D MIP ResNet50**
  - MIP channels: `Post_1`, `Sub_1`, `Sub_2`, `Post_2`
  - weighted and unweighted variants
  - ensemble at inference time

## Main Finding

A key result of this project was that **averaging logits performed better than averaging probabilities**.  
This improved leaderboard performance without changing the underlying models.

## Results

- Final leaderboard score: **~0.57**
- Final rank: **9 / 28 teams**
- Best single 3D model: **~0.54**
- Logit ensemble improvement: **about +0.02**

## Repository Structure

```text
src/
├── analysis/
│   └── eda.py
├── baselines/
│   ├── densenet121_192.py
│   ├── densenet169_384.py
│   ├── efficientnet_192.py
│   ├── train_densenet121_384.py
│   ├── train_densenet169_384.py
│   └── train_efficientnet_384.py
├── final_models/
│   ├── resnet18_3d.py
│   └── mip_resnet50_ensemble.py
└── inference/
    └── ensemble_inference.py
