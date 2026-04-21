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
import torchvision.models as tv_models
from scipy.ndimage import zoom as scipy_zoom
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_CENTERS = ["CAM", "MHA", "RUMC", "UKA"]
NUM_CLASSES = 3
MIP_SIZE = (256, 256)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D MIP ResNet50 ensemble for breast cancer MRI classification."
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
        help="Centers used for train/val/test metadata loading.",
    )
    parser.add_argument(
        "--test_center",
        type=str,
        default="RSH",
        help="Unseen test center used for submission generation.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr_start", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
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
    weights = 1.0 / counts.values.astype(np.float32)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_vol(folder: Path, mod: str):
    f = folder / f"{mod}.nii.gz"
    if f.exists():
        vol = nib.load(str(f)).get_fdata(dtype=np.float32)

        if (vol > 0).any():
            threshold = np.percentile(vol[vol > 0], 5)
            vol[vol <= threshold] = 0

        vol_max = vol.max()
        if vol_max > 0:
            vol = vol / vol_max

        return vol.astype(np.float32)

    return np.zeros((256, 256, 32), dtype=np.float32)


def compute_mips(uid: str, center: str, data_root: Path):
    """
    Compute a 4-channel MIP image:
    [Post_1, Sub_1=(Post_1-Pre), Sub_2=(Post_2-Pre), Post_2]
    Returns shape: (4, H, W)
    """
    folder = data_root / center / "data_unilateral" / uid

    pre = load_vol(folder, "Pre")
    post1 = load_vol(folder, "Post_1")
    post2 = load_vol(folder, "Post_2")

    sub1 = np.clip(post1 - pre, 0, None)
    sub2 = np.clip(post2 - pre, 0, None)

    for sub in [sub1, sub2]:
        s_max = sub.max()
        if s_max > 0:
            sub /= s_max

    mips = []
    for vol in [post1, sub1, sub2, post2]:
        mip = np.max(vol, axis=-1)
        factors = [t / s for t, s in zip(MIP_SIZE, mip.shape)]
        mip = scipy_zoom(mip, factors, order=1)
        mips.append(mip.astype(np.float32))

    return np.stack(mips, axis=0)


# =============================================================================
# DATASETS
# =============================================================================

class MIPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def augment_mip(self, mip: np.ndarray):
        if np.random.rand() > 0.5:
            mip = np.flip(mip, axis=1).copy()

        if np.random.rand() > 0.5:
            mip = np.flip(mip, axis=2).copy()

        if np.random.rand() > 0.5:
            std = np.random.uniform(0.005, 0.015)
            mip = np.clip(
                mip + np.random.normal(0, std, mip.shape).astype(np.float32),
                0,
                1,
            )

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            mip = np.clip(mip * scale, 0, 1)

        if np.random.rand() > 0.7:
            shift = np.random.uniform(-0.05, 0.05)
            mip = np.clip(mip + shift, 0, 1)

        return mip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mips = compute_mips(row["UID"], row["Institution"], self.data_root)

        if self.augment:
            mips = self.augment_mip(mips)

        tensor = torch.tensor(mips, dtype=torch.float32)
        label = torch.tensor(row["Label"], dtype=torch.long)
        return tensor, label


class SubmissionMIPDataset(Dataset):
    def __init__(self, uids, data_root: Path, center: str):
        self.uids = uids
        self.data_root = data_root
        self.center = center

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        mips = compute_mips(uid, self.center, self.data_root)
        return torch.tensor(mips, dtype=torch.float32), uid


# =============================================================================
# MODEL
# =============================================================================

class MIPClassifier(nn.Module):
    """
    2D ResNet50 classifier for 4-channel MIP breast MRI input.
    Input:  (B, 4, 256, 256)
    Output: (B, 3)
    """
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()

        self.backbone = tv_models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out")

    def forward(self, x):
        return self.backbone(x)


def make_model(device: torch.device):
    model = MIPClassifier(in_channels=4, num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def make_optimizer(model, lr_start, lr_max, warmup_epochs, total_epochs, weight_decay):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_start,
        weight_decay=weight_decay,
    )

    def lr_scale(epoch):
        if epoch < warmup_epochs:
            return 1.0 + (lr_max / lr_start - 1.0) * (epoch / warmup_epochs)

        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return (lr_max / lr_start) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    return optimizer, scheduler


# =============================================================================
# TRAINING HELPERS
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mips, labels in tqdm(loader, desc="Training"):
        mips = mips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(mips)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, return_probs=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for mips, labels in tqdm(loader, desc="Evaluating"):
            mips = mips.to(device)
            labels = labels.to(device)

            outputs = model(mips)
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

    if return_probs:
        return total_loss / len(loader), correct / total, auroc, all_probs, all_labels

    return total_loss / len(loader), correct / total, auroc


def predict_tta(model, loader, device, n_tta=5):
    all_runs = []
    final_uids = []

    for tta_idx in range(n_tta):
        run_uids = []
        run_probs = []

        model.eval()
        with torch.no_grad():
            for mips, uids in tqdm(loader, desc=f"TTA {tta_idx + 1}/{n_tta}"):
                if tta_idx == 1:
                    mips = torch.flip(mips, dims=[2])
                elif tta_idx == 2:
                    mips = torch.flip(mips, dims=[3])
                elif tta_idx == 3:
                    mips = torch.flip(mips, dims=[2, 3])
                elif tta_idx == 4:
                    mips = torch.clamp(
                        mips + torch.randn_like(mips) * 0.005,
                        0,
                        1,
                    )

                outputs = model(mips.to(device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                run_uids.extend(uids)
                run_probs.append(probs)

        all_runs.append(np.concatenate(run_probs, axis=0))
        if tta_idx == 0:
            final_uids = run_uids

    return final_uids, np.mean(all_runs, axis=0)


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
    print("Approach: MIP + 2D ResNet50 ensemble")

    df = load_metadata(data_root, args.centers)

    # Direct mapping: 0=normal, 1=benign, 2=malignant
    df["Label"] = df["Lesion"].astype(int)

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['Label'].value_counts()}")

    class_weights = compute_class_weights(train_df, device)
    print(f"Class weights: {class_weights}")

    train_dataset = MIPDataset(train_df, data_root, augment=True)
    val_dataset = MIPDataset(val_df, data_root, augment=False)
    test_dataset = MIPDataset(test_df, data_root, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion_w = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    criterion_uw = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )

    # -------------------------------------------------------------------------
    # Model A: weighted
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Model A: WITH class weights")
    print("=" * 60)

    model_a = make_model(device)
    opt_a, sch_a = make_optimizer(
        model_a,
        lr_start=args.lr_start,
        lr_max=args.lr_max,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    es_a = EarlyStopping(patience=args.patience)

    best_auc_a = 0.0
    history_a = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auroc": [],
    }

    model_a_path = output_dir / "best_model_a.pth"

    for epoch in range(args.epochs):
        lr = opt_a.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} | LR: {lr:.7f} ---")

        train_loss, train_acc = train_epoch(
            model_a, train_loader, opt_a, criterion_w, device
        )
        val_loss, val_acc, val_auroc = evaluate(
            model_a, val_loader, criterion_w, device
        )
        sch_a.step()

        history_a["train_loss"].append(train_loss)
        history_a["val_loss"].append(val_loss)
        history_a["train_acc"].append(train_acc)
        history_a["val_acc"].append(val_acc)
        history_a["val_auroc"].append(val_auroc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")

        if val_auroc > best_auc_a:
            best_auc_a = val_auroc
            torch.save(model_a.state_dict(), model_a_path)
            print(f"Model A best AUROC: {best_auc_a:.4f}")

        if es_a.step(val_auroc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # -------------------------------------------------------------------------
    # Model B: unweighted
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training Model B: WITHOUT class weights")
    print("=" * 60)

    model_b = make_model(device)
    opt_b, sch_b = make_optimizer(
        model_b,
        lr_start=args.lr_start,
        lr_max=args.lr_max,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        weight_decay=args.weight_decay,
    )
    es_b = EarlyStopping(patience=args.patience)

    best_auc_b = 0.0
    history_b = {
        "val_auroc": [],
    }

    model_b_path = output_dir / "best_model_b.pth"

    for epoch in range(args.epochs):
        lr = opt_b.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} | LR: {lr:.7f} ---")

        train_loss, train_acc = train_epoch(
            model_b, train_loader, opt_b, criterion_uw, device
        )
        val_loss, val_acc, val_auroc = evaluate(
            model_b, val_loader, criterion_uw, device
        )
        sch_b.step()

        history_b["val_auroc"].append(val_auroc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUROC: {val_auroc:.4f}")

        if val_auroc > best_auc_b:
            best_auc_b = val_auroc
            torch.save(model_b.state_dict(), model_b_path)
            print(f"Model B best AUROC: {best_auc_b:.4f}")

        if es_b.step(val_auroc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # -------------------------------------------------------------------------
    # Ensemble evaluation on test set
    # -------------------------------------------------------------------------
    print("\n--- Ensemble Test Evaluation ---")

    model_a.load_state_dict(torch.load(model_a_path, map_location=device, weights_only=True))
    model_b.load_state_dict(torch.load(model_b_path, map_location=device, weights_only=True))

    _, _, auroc_a, probs_a, labels_gt = evaluate(
        model_a, test_loader, criterion_w, device, return_probs=True
    )
    _, _, auroc_b, probs_b, _ = evaluate(
        model_b, test_loader, criterion_uw, device, return_probs=True
    )

    ensemble_probs = (probs_a + probs_b) / 2.0
    binary_labels = (labels_gt == 2).astype(int)
    auroc_ensemble = roc_auc_score(binary_labels, ensemble_probs[:, 2])

    print(f"\nModel A (weighted)   AUROC: {auroc_a:.4f}")
    print(f"Model B (unweighted) AUROC: {auroc_b:.4f}")
    print(f"Ensemble             AUROC: {auroc_ensemble:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history_a["train_loss"], label="Train")
    axes[0].plot(history_a["val_loss"], label="Val")
    axes[0].set_title("Loss (Model A)")
    axes[0].legend()

    axes[1].plot(history_a["train_acc"], label="Train")
    axes[1].plot(history_a["val_acc"], label="Val")
    axes[1].set_title("Accuracy (Model A)")
    axes[1].legend()

    axes[2].plot(history_a["val_auroc"], label="Model A (weighted)")
    axes[2].plot(history_b["val_auroc"], label="Model B (unweighted)")
    axes[2].set_title("Validation AUROC")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_mip.png", dpi=150)
    plt.close()
    print("Saved training curves")

    # -------------------------------------------------------------------------
    # Submission with TTA
    # -------------------------------------------------------------------------
    print(f"\n--- Generating submission for center: {args.test_center} ---")
    submission_center_path = data_root / args.test_center / "data_unilateral"

    if not submission_center_path.exists():
        raise FileNotFoundError(f"Submission center path not found: {submission_center_path}")

    submission_uids = sorted([f.name for f in submission_center_path.iterdir() if f.is_dir()])
    print(f"Found {len(submission_uids)} samples")

    submission_dataset = SubmissionMIPDataset(
        submission_uids,
        data_root=data_root,
        center=args.test_center,
    )
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    print("TTA with Model A (weighted)...")
    uids_a, submission_probs_a = predict_tta(
        model_a,
        submission_loader,
        device,
        n_tta=args.tta_runs,
    )

    print("TTA with Model B (unweighted)...")
    uids_b, submission_probs_b = predict_tta(
        model_b,
        submission_loader,
        device,
        n_tta=args.tta_runs,
    )

    if uids_a != uids_b:
        raise ValueError("Submission UID ordering mismatch between ensemble members.")

    submission_ensemble = (submission_probs_a + submission_probs_b) / 2.0

    submission = pd.DataFrame(
        {
            "ID": uids_a,
            "normal": submission_ensemble[:, 0],
            "benign": submission_ensemble[:, 1],
            "malignant": submission_ensemble[:, 2],
        }
    )

    row_sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    for col in ["normal", "benign", "malignant"]:
        submission[col] = (submission[col] / row_sums).round(6)

    sums = submission[["normal", "benign", "malignant"]].sum(axis=1)
    pred_class = submission[["normal", "benign", "malignant"]].idxmax(axis=1)

    print("\nSanity Checks:")
    print(f"All rows sum to 1: {np.allclose(sums, 1.0, atol=1e-4)}")
    print(f"No NaN:            {not submission.isnull().any().any()}")
    print(f"Prediction distribution:\n{pred_class.value_counts()}")
    print(f"\nPreview:\n{submission.head(5).to_string()}")
    print(f"Total rows: {len(submission)}")

    submission_path = output_dir / "predictions_mip.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission: {submission_path}")

    results = {
        "model": "ResNet50_MIP_Ensemble",
        "approach": "MIP + 2D ResNet50 + weighted/unweighted ensemble",
        "mip_channels": [
            "Post_1",
            "Sub_1 = Post_1 - Pre",
            "Sub_2 = Post_2 - Pre",
            "Post_2",
        ],
        "mip_size": list(MIP_SIZE),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_start": args.lr_start,
        "lr_max": args.lr_max,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "model_a_auroc": auroc_a,
        "model_b_auroc": auroc_b,
        "ensemble_auroc": auroc_ensemble,
        "tta_runs": args.tta_runs,
        "centers": args.centers,
        "test_center": args.test_center,
    }

    with open(output_dir / "results_mip.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "history_model_a.json", "w") as f:
        json.dump(history_a, f, indent=2)

    with open(output_dir / "history_model_b.json", "w") as f:
        json.dump(history_b, f, indent=2)

    print("\nMIP ensemble training complete.")


if __name__ == "__main__":
    main()
