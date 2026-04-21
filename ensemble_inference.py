import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from monai.networks.nets import resnet18
from scipy.ndimage import zoom as scipy_zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

TARGET_SPACING = (1.5, 1.5, 1.5)
TARGET_SIZE = (112, 112, 40)
MIP_SIZE = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble inference for breast cancer MRI classification."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing challenge data.",
    )
    parser.add_argument(
        "--test_center",
        type=str,
        default="RSH",
        help="Name of the test center folder inside data_root.",
    )
    parser.add_argument(
        "--model2_path",
        type=str,
        required=True,
        help="Path to the trained 3D ResNet18 weights (.pth).",
    )
    parser.add_argument(
        "--model3_path",
        type=str,
        required=True,
        help="Path to the trained 2D MIP ResNet50 weights (.pth).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="final_submission.csv",
        help="Output path for the submission CSV.",
    )
    parser.add_argument(
        "--batch_size_m2",
        type=int,
        default=2,
        help="Batch size for the 3D model.",
    )
    parser.add_argument(
        "--batch_size_m3",
        type=int,
        default=8,
        help="Batch size for the 2D MIP model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--n_tta",
        type=int,
        default=5,
        help="Number of test-time augmentation runs.",
    )
    parser.add_argument(
        "--weight_m2",
        type=float,
        default=0.6,
        help="Ensemble weight for the 3D ResNet18 model.",
    )
    parser.add_argument(
        "--weight_m3",
        type=float,
        default=0.4,
        help="Ensemble weight for the 2D MIP ResNet50 model.",
    )
    return parser.parse_args()


# -----------------------------
# Model Definition
# -----------------------------

class MIPClassifier(nn.Module):
    """
    2D ResNet50 for MIP-based classification.

    Input:
        4-channel MIP image (Post_1, Sub_1, Sub_2, Post_2)

    Output:
        3-class logits (normal, benign, malignant)
    """

    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# Dataset Classes
# -----------------------------

class DatasetM2(Dataset):
    """
    Preprocessing for Model 2 (3D ResNet18):
    - Modalities: Pre, Post_1, Post_2
    - Isotropic resampling to 1.5 mm spacing
    - Background masking
    - Max normalization
    - Resize to (112, 112, 40)
    """

    def __init__(self, uids, data_root: Path, test_center: str):
        self.uids = uids
        self.data_root = data_root
        self.test_center = test_center

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        folder = self.data_root / self.test_center / "data_unilateral" / uid
        vols = []

        for mod in ["Pre", "Post_1", "Post_2"]:
            f = folder / f"{mod}.nii.gz"

            if f.exists():
                img = nib.load(str(f))
                vol = img.get_fdata(dtype=np.float32)

                try:
                    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
                    if np.all(spacing > 0):
                        zoom_factors = spacing / np.array(TARGET_SPACING, dtype=np.float32)
                        vol = scipy_zoom(vol, zoom_factors, order=1)
                except Exception:
                    pass

                if (vol > 0).any():
                    threshold = np.percentile(vol[vol > 0], 5)
                    vol[vol <= threshold] = 0

                vmax = vol.max()
                if vmax > 0:
                    vol = vol / vmax

                factors = [t / s for t, s in zip(TARGET_SIZE, vol.shape)]
                vol = scipy_zoom(vol, factors, order=1)
                vols.append(vol.astype(np.float32))
            else:
                vols.append(np.zeros(TARGET_SIZE, dtype=np.float32))

        return torch.tensor(np.stack(vols, axis=0), dtype=torch.float32), uid


class DatasetM3(Dataset):
    """
    Preprocessing for Model 3 (2D MIP ResNet50):
    - Modalities: Pre, Post_1, Post_2
    - Percentile normalization (p1-p99 clipping)
    - Subtraction channels: Sub_1 = Post_1 - Pre, Sub_2 = Post_2 - Pre
    - Maximum Intensity Projection along depth axis
    - 4-channel MIP: (Post_1, Sub_1, Sub_2, Post_2)
    - Resize to (256, 256)
    """

    def __init__(self, uids, data_root: Path, test_center: str):
        self.uids = uids
        self.data_root = data_root
        self.test_center = test_center

    def __len__(self):
        return len(self.uids)

    def load_vol(self, folder: Path, mod: str):
        f = folder / f"{mod}.nii.gz"
        if f.exists():
            vol = nib.load(str(f)).get_fdata(dtype=np.float32)
            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = np.clip(vol, p1, p99)
            if p99 > p1:
                vol = (vol - p1) / (p99 - p1)
            return vol.astype(np.float32)

        return np.zeros((256, 256, 32), dtype=np.float32)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        folder = self.data_root / self.test_center / "data_unilateral" / uid

        pre = self.load_vol(folder, "Pre")
        post1 = self.load_vol(folder, "Post_1")
        post2 = self.load_vol(folder, "Post_2")

        sub1 = np.clip(post1 - pre, 0, None)
        sub2 = np.clip(post2 - pre, 0, None)

        for sub in [sub1, sub2]:
            smax = sub.max()
            if smax > 0:
                sub /= smax

        mips = []
        for vol in [post1, sub1, sub2, post2]:
            mip = np.max(vol, axis=-1)
            factors = [t / s for t, s in zip(MIP_SIZE, mip.shape)]
            mip = scipy_zoom(mip, factors, order=1)
            mips.append(mip.astype(np.float32))

        return torch.tensor(np.stack(mips, axis=0), dtype=torch.float32), uid


# -----------------------------
# Inference Utilities
# -----------------------------

def get_logits_tta(
    model,
    dataset,
    device,
    batch_size=4,
    num_workers=2,
    n_tta=5,
):
    """
    Run test-time augmentation and return averaged logits.
    Averaging is done in logit space, not probability space.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_runs = []
    final_uids = []

    for t in range(n_tta):
        run_uids = []
        run_logits = []
        model.eval()

        with torch.no_grad():
            for vols, ids in tqdm(loader, desc=f"TTA {t + 1}/{n_tta}"):
                if t == 0:
                    pass
                elif t == 1:
                    vols = torch.flip(vols, dims=[2])
                elif t == 2:
                    vols = torch.flip(vols, dims=[3])
                elif t == 3:
                    vols = torch.flip(vols, dims=[2, 3])
                elif t == 4:
                    vols = torch.clamp(vols + torch.randn_like(vols) * 0.005, 0, 1)

                logits = model(vols.to(device)).cpu().numpy()
                run_uids.extend(ids)
                run_logits.append(logits)

        all_runs.append(np.concatenate(run_logits, axis=0))
        if t == 0:
            final_uids = run_uids

    avg_logits = np.mean(all_runs, axis=0)
    return final_uids, avg_logits


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    model2_path = Path(args.model2_path)
    model3_path = Path(args.model3_path)
    output_csv = Path(args.output_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_path = data_root / args.test_center / "data_unilateral"
    if not test_path.exists():
        raise FileNotFoundError(f"Test path not found: {test_path}")

    rsh_samples = sorted([f.name for f in test_path.iterdir() if f.is_dir()])
    print(f"Found {len(rsh_samples)} test samples in center '{args.test_center}'")

    print("\nLoading 3D ResNet18 model...")
    model2 = resnet18(
        pretrained=False,
        spatial_dims=3,
        n_input_channels=3,
        num_classes=3,
    ).to(device)

    model2.load_state_dict(
        torch.load(model2_path, weights_only=True, map_location=device)
    )
    model2.eval()
    print("  3D ResNet18 loaded successfully")

    print("Loading 2D MIP ResNet50 model...")
    model3 = MIPClassifier(in_channels=4, num_classes=3).to(device)
    model3.load_state_dict(
        torch.load(model3_path, weights_only=True, map_location=device)
    )
    model3.eval()
    print("  2D MIP ResNet50 loaded successfully")

    dataset_m2 = DatasetM2(rsh_samples, data_root=data_root, test_center=args.test_center)
    dataset_m3 = DatasetM3(rsh_samples, data_root=data_root, test_center=args.test_center)

    print("\nRunning TTA inference with 3D ResNet18...")
    uids_m2, logits_m2 = get_logits_tta(
        model=model2,
        dataset=dataset_m2,
        device=device,
        batch_size=args.batch_size_m2,
        num_workers=args.num_workers,
        n_tta=args.n_tta,
    )

    print("\nRunning TTA inference with 2D MIP ResNet50...")
    uids_m3, logits_m3 = get_logits_tta(
        model=model3,
        dataset=dataset_m3,
        device=device,
        batch_size=args.batch_size_m3,
        num_workers=args.num_workers,
        n_tta=args.n_tta,
    )

    if uids_m2 != uids_m3:
        raise ValueError("Sample ordering mismatch between model outputs.")

    print("\nRaw logit stats:")
    for name, logits in [("M2", logits_m2), ("M3", logits_m3)]:
        probs_tmp = torch.softmax(torch.tensor(logits), dim=1).numpy()
        print(
            f"  {name}: normal={probs_tmp[:, 0].mean():.3f}  "
            f"benign={probs_tmp[:, 1].mean():.3f}  "
            f"malignant={probs_tmp[:, 2].mean():.3f}"
        )

    w_m2 = args.weight_m2
    w_m3 = args.weight_m3

    if not np.isclose(w_m2 + w_m3, 1.0):
        print(
            f"Warning: ensemble weights sum to {w_m2 + w_m3:.3f}, not 1.0. "
            "This is not necessarily wrong, but check if intentional."
        )

    print(f"\nLogit ensemble: {w_m2:.2f} x M2 + {w_m3:.2f} x M3")
    ensemble_logits = w_m2 * logits_m2 + w_m3 * logits_m3
    ensemble_probs = torch.softmax(torch.tensor(ensemble_logits), dim=1).numpy()

    print(
        f"Ensemble probs: normal={ensemble_probs[:, 0].mean():.3f}  "
        f"benign={ensemble_probs[:, 1].mean():.3f}  "
        f"malignant={ensemble_probs[:, 2].mean():.3f}"
    )

    submission = pd.DataFrame(
        {
            "ID": uids_m2,
            "normal": ensemble_probs[:, 0],
            "benign": ensemble_probs[:, 1],
            "malignant": ensemble_probs[:, 2],
        }
    )

    row_sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    for col in ["normal", "benign", "malignant"]:
        submission[col] = (submission[col] / row_sums).round(6)

    sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    pred_class = submission[["normal", "benign", "malignant"]].idxmax(axis=1)

    print("\nSanity checks:")
    print(f"  All rows sum to 1: {np.allclose(sums, 1.0, atol=1e-4)}")
    print(f"  Any NaN: {submission.isnull().any().any()}")
    print(f"  Predicted class distribution:\n{pred_class.value_counts()}")
    print(f"  Malignant prob mean: {submission['malignant'].mean():.4f}")
    print(f"  Malignant prob max:  {submission['malignant'].max():.4f}")
    print(f"\nPreview:\n{submission.head(5).to_string()}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)

    print(f"\nSaved submission to: {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
