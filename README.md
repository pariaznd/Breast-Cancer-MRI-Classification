# Breast-Cancer-MRI-Classification

This repository contains our TDT4265 mini-project for breast cancer classification from DCE-MRI in the ODELIA 2025 challenge.

## Method
Our final submission uses an ensemble of:
- 3D ResNet18 on volumetric MRI inputs
- 2D ResNet50 on Maximum Intensity Projection (MIP) inputs
- Sweep models (DenseNet121, DenseNet169, efficintnet_B4) and their training

We use:
- modality-specific preprocessing
- test-time augmentation (TTA)
- logit averaging for ensemble fusion

## Main finding
Averaging logits performed better than averaging probabilities, improving leaderboard performance with no extra training cost.

## Repository structure
- `src/ensemble_inference.py`: final inference and submission script
- `requirements.txt`: dependencies
- `figures/`: plots and visuals used in the presentation

## Notes
Model weights and challenge data are not included in this repository.
