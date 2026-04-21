import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from scipy.ndimage import zoom as scipy_zoom
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_CENTERS = ["CAM", "MHA", "RUMC", "UKA"]
DEFAULT_MODALITIES = ["Pre", "Post_1", "Post_2"]

NUM_CLASSES = 3
TARGET_SPACING = (1.5, 1.5, 1.5)
TARGET_SIZE = (112, 112, 40)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a 3D ResNet18 model for breast cancer MRI classification."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing center subfolders and metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving checkpoints, plots, JSON results, and submissions.",
    )
    parser.add_argument(
        "--centers",
        nargs="+",
        default=DEFAULT_CENTERS,
        help="Centers to use for train/val/test metadata loading.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=DEFAULT_MODALITIES,
        help="MRI modalities to use as channels.",
    )
    parser.add_argument(
        "--test_center",
        type=str,
        default="RSH",
        help="Unseen test center used for submission generation.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_start", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tta_runs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =============================================================================
# UTILS
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(data_root: Path, centers):
    dfs = []
    for center in centers:
        meta = data_root / center / "metadata_unilateral" / "annotation.csv"
        split = data_root / center / "metadata_unilateral" / "split.csv"

        if meta.exists() and split.exists():
            df = pd.read_csv(meta).merge(pd.read_csv(split), on="UID")
            df["Institution"] = center
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No valid metadata files found for the provided centers.")

    return pd.concat(dfs, ignore_index=True)


def compute_class_weights(train_df: pd.DataFrame, device: torch.device):
    counts = train_df["Label"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def preprocess_volume(
    img_path: Path,
    target_spacing=(1.5, 1.5, 1.5),
    target_size=(112, 112, 40),
):
    if not img_path.exists():
        return np.zeros(target_size, dtype=np.float32)

    img = nib.load(str(img_path))
    vol = img.get_fdata(dtype=np.float32)

    try:
        current_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
        if np.all(current_spacing > 0):
            zoom_factors = current_spacing / np.array(target_spacing, dtype=np.float32)
            vol = scipy_zoom(vol, zoom_factors, order=1)
    except Exception:
        pass

    if (vol > 0).any():
        threshold = np.percentile(vol[vol > 0], 5)
        vol[vol <= threshold] = 0

    vol_max = vol.max()
    if vol_max > 0:
        vol = vol / vol_max

    factors = [t / s for t, s in zip(target_size, vol.shape)]
    vol = scipy_zoom(vol, factors, order=1)

    return vol.astype(np.float32)


# =============================================================================
# DATASET
# =============================================================================

class MRIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        modalities,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.modalities = modalities
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def load_case(self, uid: str, center: str):
        folder = self.data_root / center / "data_unilateral" / uid
        vols = []

        for mod in self.modalities:
            img_path = folder / f"{mod}.nii.gz"
            vol = preprocess_volume(
                img_path=img_path,
                target_spacing=TARGET_SPACING,
                target_size=TARGET_SIZE,
            )
            vols.append(vol)

        return np.stack(vols, axis=0)

    def augment_volume(self, volume: np.ndarray):
        if np.random.rand() > 0.5:
            std = np.random.uniform(0.005, 0.015)
            noise = np.random.normal(0, std, volume.shape).astype(np.float32)
            volume = np.clip(volume + noise, 0, 1)

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)

        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()

        return volume

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        volume = self.load_case(row["UID"], row["Institution"])

        if self.augment:
            volume = self.augment_volume(volume)

        tensor = torch.tensor(volume, dtype=torch.float32)
        label = torch.tensor(row["Label"], dtype=torch.long)

        return tensor, label


class SubmissionDataset(Dataset):
    def __init__(self, uids, data_root: Path, center: str, modalities):
        self.uids = uids
        self.data_root = data_root
        self.center = center
        self.modalities = modalities

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        folder = self.data_root / self.center / "data_unilateral" / uid

        vols = []
        for mod in self.modalities:
            img_path = folder / f"{mod}.nii.gz"
            vol = preprocess_volume(
                img_path=img_path,
                target_spacing=TARGET_SPACING,
                target_size=TARGET_SIZE,
            )
            vols.append(vol)

        tensor = torch.tensor(np.stack(vols, axis=0), dtype=torch.float32)
        return tensor, uid


# =============================================================================
# TRAINING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best = 0.0

    def step(self, auroc: float):
        if auroc > self.best:
            self.best = auroc
            self.counter = 0
            return False

        self.counter += 1
        print(f"EarlyStopping: {self.counter}/{self.patience}")
        return self.counter >= self.patience


def get_lr_scale(epoch, lr_start, lr_max, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return 1.0 + (lr_max / lr_start - 1.0) * (epoch / warmup_epochs)

    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return (lr_max / lr_start) * 0.5 * (1 + np.cos(np.pi * progress))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for volumes, labels in tqdm(loader, desc="Training"):
        volumes = volumes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Evaluating"):
            volumes = volumes.to(device)
            labels = labels.to(device)

            outputs = model(volumes)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    binary = (all_labels == 2).astype(int)
    try:
        auroc = roc_auc_score(binary, all_probs[:, 2])
    except Exception:
        auroc = 0.0

    return total_loss / len(loader), correct / total, auroc, all_probs, all_labels


def run_tta_submission(model, loader, device, tta_runs=5):
    all_probs_tta = []
    final_uids = []

    for tta_idx in range(tta_runs):
        run_uids = []
        run_probs = []

        model.eval()
        with torch.no_grad():
            for volumes, uids in tqdm(loader, desc=f"TTA {tta_idx + 1}/{tta_runs}"):
                if tta_idx == 1:
                    volumes = torch.flip(volumes, dims=[2])
                elif tta_idx == 2:
                    volumes = torch.flip(volumes, dims=[3])
                elif tta_idx == 3:
                    volumes = torch.flip(volumes, dims=[2, 3])
                elif tta_idx == 4:
                    volumes = volumes + torch.randn_like(volumes) * 0.005
                    volumes = torch.clamp(volumes, 0, 1)

                outputs = model(volumes.to(device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                run_uids.extend(uids)
                run_probs.append(probs)

        all_probs_tta.append(np.concatenate(run_probs, axis=0))
        if tta_idx == 0:
            final_uids = run_uids

    avg_probs = np.mean(all_probs_tta, axis=0)
    return final_uids, avg_probs


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Modalities: {args.modalities}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR warmup: {args.lr_start} -> {args.lr_max}")

    df = load_metadata(data_root, args.centers)

    label_map = {0: 0, 2: 1, 1: 2}
    df["Label"] = df["Lesion"].map(label_map)

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['Label'].value_counts()}")

    class_weights = compute_class_weights(train_df, device)
    print(f"Class weights: {class_weights}")

    train_dataset = MRIDataset(train_df, data_root, args.modalities, augment=True)
    val_dataset = MRIDataset(val_df, data_root, args.modalities, augment=False)
    test_dataset = MRIDataset(test_df, data_root, args.modalities, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = resnet18(
        pretrained=False,
        spatial_dims=3,
        n_input_channels=len(args.modalities),
        num_classes=NUM_CLASSES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ResNet18 (3D)")
    print(f"Parameters: {total_params:,}")
    print(f"Input: {len(args.modalities)} x {TARGET_SIZE}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_start,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: get_lr_scale(
            epoch,
            lr_start=args.lr_start,
            lr_max=args.lr_max,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
        ),
    )

    early_stopping = EarlyStopping(patience=args.patience)

    best_auroc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auroc": [],
        "lr": [],
    }

    print("\n" + "=" * 60)
    print("TRAINING 3D RESNET18")
    print("=" * 60)

    checkpoint_path = output_dir / "best_final_model.pth"

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} | LR: {current_lr:.7f} ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best AUROC: {best_auroc:.4f}")

        if early_stopping.step(val_auroc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(history["val_auroc"], label="Val AUROC")
    axes[2].set_title("Validation AUROC")
    axes[2].legend()

    axes[3].plot(history["lr"])
    axes[3].set_title("LR Schedule")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_final.png", dpi=150)
    plt.close()

    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc, test_auroc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nFinal Model AUROC: {test_auroc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    print(f"\n--- Generating submission for center: {args.test_center} ---")
    submission_center_path = data_root / args.test_center / "data_unilateral"

    if not submission_center_path.exists():
        raise FileNotFoundError(f"Submission center path not found: {submission_center_path}")

    submission_uids = sorted([f.name for f in submission_center_path.iterdir() if f.is_dir()])
    print(f"Found {len(submission_uids)} samples")

    submission_dataset = SubmissionDataset(
        submission_uids,
        data_root=data_root,
        center=args.test_center,
        modalities=args.modalities,
    )
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    final_uids, avg_probs = run_tta_submission(
        model,
        submission_loader,
        device,
        tta_runs=args.tta_runs,
    )

    submission = pd.DataFrame(
        {
            "ID": final_uids,
            "normal": avg_probs[:, 0],
            "benign": avg_probs[:, 1],
            "malignant": avg_probs[:, 2],
        }
    )

    row_sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    for col in ["normal", "benign", "malignant"]:
        submission[col] = (submission[col] / row_sums).round(6)

    sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    pred_class = submission[["normal", "benign", "malignant"]].idxmax(axis=1)

    print(f"\nAll rows sum to 1: {np.allclose(sums, 1.0, atol=1e-4)}")
    print(f"No NaN: {not submission.isnull().any().any()}")
    print(f"Prediction distribution:\n{pred_class.value_counts()}")
    print(f"\nPreview:\n{submission.head(5).to_string()}")

    submission_path = output_dir / "predictions_best.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission: {submission_path}")

    results = {
        "model": "ResNet18_3D",
        "modalities": args.modalities,
        "target_size": list(TARGET_SIZE),
        "target_spacing": list(TARGET_SPACING),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr_start": args.lr_start,
        "lr_max": args.lr_max,
        "label_smoothing": args.label_smoothing,
        "tta_runs": args.tta_runs,
        "epochs_trained": len(history["val_auroc"]),
        "best_val_auroc": best_auroc,
        "test_auroc": test_auroc,
        "centers": args.centers,
        "test_center": args.test_center,
    }

    with open(output_dir / "results_final.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "history_final.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
