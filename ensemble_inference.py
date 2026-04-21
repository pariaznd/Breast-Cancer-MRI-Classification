import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from monai.networks.nets import resnet18
from scipy.ndimage import zoom as scipy_zoom


# ---- Config ----

DATA_ROOT = Path("/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data")
RSH_CENTER = "RSH"
PROJECT_DIR = Path("/cluster/home/pariaz/tdt4265_project")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 2 preprocessing settings (must match training exactly)
TARGET_SPACING = (1.5, 1.5, 1.5)
TARGET_SIZE = (112, 112, 40)

# MIP settings for Model 3
MIP_SIZE = (256, 256)

print(f"Device: {DEVICE}")

# Find RSH test samples
rsh_path = DATA_ROOT / RSH_CENTER / "data_unilateral"
rsh_samples = sorted([f.name for f in rsh_path.iterdir() if f.is_dir()])
print(f"Found {len(rsh_samples)} RSH test samples")


# ---- Model Definitions ----
# These must exactly match the architectures used during training

class MIPClassifier(nn.Module):
    """
    2D ResNet50 for MIP-based classification.
    Input: 4-channel MIP image (Post1, Sub1, Sub2, Post2)
    Output: 3-class logits (normal, benign, malignant)
    """
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        # modify first conv to accept 4 channels instead of the default 3
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        # classification head with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ---- Load Models ----

print("\nLoading Model 2 (ResNet18 3D isotropic, Pre+Post1+Post2)...")
model2 = resnet18(
    pretrained=False,
    spatial_dims=3,
    n_input_channels=3,
    num_classes=3
).to(DEVICE)
model2.load_state_dict(torch.load(
    PROJECT_DIR / "final_best_output/best_final_model.pth",
    weights_only=True,
    map_location=DEVICE
))
model2.eval()
print("  Model 2 loaded OK")

print("Loading Model 3 (ResNet50 2D MIP weighted)...")
model3 = MIPClassifier(in_channels=4, num_classes=3).to(DEVICE)
model3.load_state_dict(torch.load(
    PROJECT_DIR / "mip_output/best_model_A.pth",
    weights_only=True,
    map_location=DEVICE
))
model3.eval()
print("  Model 3 loaded OK")


# ---- Dataset Classes ----
# Each model was trained with different preprocessing, so we need
# separate dataset classes that reproduce those exact pipelines.

class DatasetM2(Dataset):
    """
    Preprocessing for Model 2 (ResNet18 3D):
    - Modalities: Pre, Post_1, Post_2
    - Isotropic resampling to 1.5mm spacing
    - Background masking (remove low-intensity voxels)
    - Max normalization
    - Resize to (112, 112, 40)
    """
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        folder = DATA_ROOT / RSH_CENTER / "data_unilateral" / uid
        vols = []

        for mod in ["Pre", "Post_1", "Post_2"]:
            f = folder / f"{mod}.nii.gz"
            if f.exists():
                img = nib.load(str(f))
                vol = img.get_fdata(dtype=np.float32)

                # isotropic resampling based on voxel spacing in header
                try:
                    spacing = np.array(img.header.get_zooms()[:3])
                    if np.all(spacing > 0):
                        zoom_factors = spacing / np.array(TARGET_SPACING)
                        vol = scipy_zoom(vol, zoom_factors, order=1)
                except Exception:
                    pass

                # background masking: zero out near-background voxels
                if (vol > 0).any():
                    threshold = np.percentile(vol[vol > 0], 5)
                    vol[vol <= threshold] = 0

                # max normalization to [0, 1]
                vmax = vol.max()
                if vmax > 0:
                    vol = vol / vmax

                # resize to target spatial size
                factors = [t / s for t, s in zip(TARGET_SIZE, vol.shape)]
                vol = scipy_zoom(vol, factors, order=1)
                vols.append(vol.astype(np.float32))
            else:
                vols.append(np.zeros(TARGET_SIZE, dtype=np.float32))

        return torch.FloatTensor(np.stack(vols, axis=0)), uid


class DatasetM3(Dataset):
    """
    Preprocessing for Model 3 (ResNet50 2D MIP):
    - Modalities: Pre, Post_1, Post_2 (to compute MIPs)
    - Percentile normalization (p1-p99 clipping)
    - Subtraction channels: Sub1 = Post1 - Pre, Sub2 = Post2 - Pre
    - Maximum Intensity Projection along depth axis
    - 4-channel MIP: (Post1, Sub1, Sub2, Post2)
    - Resize to (256, 256)
    """
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def load_vol(self, folder, mod):
        f = folder / f"{mod}.nii.gz"
        if f.exists():
            vol = nib.load(str(f)).get_fdata(dtype=np.float32)
            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = np.clip(vol, p1, p99)
            if p99 > p1:
                vol = (vol - p1) / (p99 - p1)
            return vol
        return np.zeros((256, 256, 32), dtype=np.float32)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        folder = DATA_ROOT / RSH_CENTER / "data_unilateral" / uid

        pre = self.load_vol(folder, "Pre")
        post1 = self.load_vol(folder, "Post_1")
        post2 = self.load_vol(folder, "Post_2")

        # subtraction images highlight contrast-enhancing tissue
        sub1 = np.clip(post1 - pre, 0, None)
        sub2 = np.clip(post2 - pre, 0, None)

        # normalize each subtraction independently
        for sub in [sub1, sub2]:
            smax = sub.max()
            if smax > 0:
                sub /= smax

        # MIP = maximum intensity projection along depth axis
        # captures the brightest signal (tumors enhance brightly)
        mips = []
        for vol in [post1, sub1, sub2, post2]:
            mip = np.max(vol, axis=-1)   # (H, W, D) -> (H, W)
            factors = [t / s for t, s in zip(MIP_SIZE, mip.shape)]
            mip = scipy_zoom(mip, factors, order=1)
            mips.append(mip.astype(np.float32))

        return torch.FloatTensor(np.stack(mips, axis=0)), uid


# ---- Prediction with Test-Time Augmentation (TTA) ----
# We run inference 5 times with different augmentations and average the logits.
# This reduces variance and improves robustness to the RSH domain.

def get_logits_tta(model, dataset_class, uids, batch_size=4, n_tta=5):
    """
    Run test-time augmentation and return averaged raw logits (pre-softmax).
    We average in logit space rather than probability space to avoid
    smoothing issues.
    """
    loader = DataLoader(
        dataset_class(uids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    all_runs = []
    final_uids = []

    for t in range(n_tta):
        run_uids = []
        run_logits = []
        model.eval()

        with torch.no_grad():
            for vols, ids in tqdm(loader, desc=f"TTA {t+1}/{n_tta}"):
                # apply different augmentations per run
                if t == 0:
                    pass  # original, no augmentation
                elif t == 1:
                    vols = torch.flip(vols, dims=[2])   # horizontal flip
                elif t == 2:
                    vols = torch.flip(vols, dims=[3])   # vertical flip
                elif t == 3:
                    vols = torch.flip(vols, dims=[2, 3])  # both flips
                elif t == 4:
                    # small gaussian noise
                    vols = torch.clamp(vols + torch.randn_like(vols) * 0.005, 0, 1)

                logits = model(vols.to(DEVICE)).cpu().numpy()
                run_uids.extend(ids)
                run_logits.append(logits)

        all_runs.append(np.concatenate(run_logits, axis=0))
        if t == 0:
            final_uids = run_uids

    # average logits across all TTA runs
    avg_logits = np.mean(all_runs, axis=0)
    return final_uids, avg_logits


# ---- Get Predictions ----

print("\nRunning TTA inference with Model 2 (ResNet18 3D)...")
uids_m2, logits_m2 = get_logits_tta(model2, DatasetM2, rsh_samples, batch_size=2)

print("\nRunning TTA inference with Model 3 (ResNet50 MIP)...")
uids_m3, logits_m3 = get_logits_tta(model3, DatasetM3, rsh_samples, batch_size=8)

# Quick sanity check on raw logits
print("\nRaw logit stats:")
for name, logits in [("M2", logits_m2), ("M3", logits_m3)]:
    probs_tmp = torch.softmax(torch.tensor(logits), dim=1).numpy()
    print(f"  {name}: normal={probs_tmp[:,0].mean():.3f}  "
          f"benign={probs_tmp[:,1].mean():.3f}  "
          f"malignant={probs_tmp[:,2].mean():.3f}")


# ---- Logit Ensemble ----
# Our key finding: combining logits is better than combining probabilities.
# When we average probabilities, the softmax has already "squashed" the
# signal. If we combine logits first and apply softmax once at the end,
# the signal from each model is better preserved.
#
# We tried all weight combinations in steps of 0.1 on the validation set.
# Best: 0.6 * M2 + 0.4 * M3 -> leaderboard score 0.5739

w_m2 = 0.6
w_m3 = 0.4

print(f"\nLogit ensemble: {w_m2:.1f} x M2 + {w_m3:.1f} x M3")
ensemble_logits = w_m2 * logits_m2 + w_m3 * logits_m3

# single softmax at the end (not per-model)
ensemble_probs = torch.softmax(torch.tensor(ensemble_logits), dim=1).numpy()

print(f"Ensemble probs: normal={ensemble_probs[:,0].mean():.3f}  "
      f"benign={ensemble_probs[:,1].mean():.3f}  "
      f"malignant={ensemble_probs[:,2].mean():.3f}")


# ---- Build Submission CSV ----

submission = pd.DataFrame({
    "ID": uids_m2,
    "normal": ensemble_probs[:, 0],
    "benign": ensemble_probs[:, 1],
    "malignant": ensemble_probs[:, 2],
})

# normalize each row to sum to exactly 1 (required by leaderboard)
row_sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
for col in ["normal", "benign", "malignant"]:
    submission[col] = (submission[col] / row_sums).round(6)

# sanity checks
sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
pred_class = submission[["normal", "benign", "malignant"]].idxmax(axis=1)

print(f"\nSanity checks:")
print(f"  All rows sum to 1: {np.allclose(sums, 1.0, atol=1e-4)}")
print(f"  Any NaN: {submission.isnull().any().any()}")
print(f"  Predicted class distribution:\n{pred_class.value_counts()}")
print(f"  Malignant prob mean: {submission['malignant'].mean():.4f}")
print(f"  Malignant prob max:  {submission['malignant'].max():.4f}")
print(f"\nPreview:\n{submission.head(5).to_string()}")

# save
out_path = PROJECT_DIR / "final_submission.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print("Upload final_submission.csv to the leaderboard!")
