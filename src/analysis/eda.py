import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data")
CENTERS   = ["CAM", "MHA", "RUMC", "UKA"]  # RSH is hidden test set
OUTPUT    = Path("/cluster/home/pariaz/tdt4265_project/eda_output")
OUTPUT.mkdir(exist_ok=True)

# ── Load all metadata ────────────────────────────────────────────────────────
dfs = []
for center in CENTERS:
    meta  = DATA_ROOT / center / "metadata_unilateral" / "annotation.csv"
    split = DATA_ROOT / center / "metadata_unilateral" / "split.csv"
    if meta.exists() and split.exists():
        df = pd.read_csv(meta).merge(pd.read_csv(split), on="UID")
        df["Institution"] = center
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df["Lesion"].value_counts())
print(df["Split"].value_counts())

# ── 1. Class distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="Lesion", palette=["steelblue","tomato"], ax=axes[0])
axes[0].set_title("Overall: Non-Malignant vs Malignant")
axes[0].set_xticklabels(["Non-Malignant (0)", "Malignant (1)"])

df_c = df.groupby(["Institution","Lesion"]).size().reset_index(name="Count")
sns.barplot(data=df_c, x="Institution", y="Count",
            hue="Lesion", palette=["steelblue","tomato"], ax=axes[1])
axes[1].set_title("Lesion Distribution per Center")
plt.tight_layout()
plt.savefig(OUTPUT / "class_distribution.png", dpi=150)
print("Saved: class_distribution.png")

# ── 2. Split distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
df_s = df.groupby(["Institution","Split"]).size().reset_index(name="Count")
sns.barplot(data=df_s, x="Institution", y="Count", hue="Split", ax=ax)
ax.set_title("Train/Val/Test Split per Center")
plt.tight_layout()
plt.savefig(OUTPUT / "split_distribution.png", dpi=150)
print("Saved: split_distribution.png")

# ── 3. Age distribution ──────────────────────────────────────────────────────
if "Age" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue="Lesion",
                 bins=30, kde=True,
                 palette=["steelblue","tomato"], ax=ax)
    ax.set_title("Age Distribution by Lesion Label")
    plt.tight_layout()
    plt.savefig(OUTPUT / "age_distribution.png", dpi=150)
    print("Saved: age_distribution.png")

# ── 4. MRI shape & spacing analysis ─────────────────────────────────────────
print("\n--- MRI Shape Analysis ---")
shapes, spacings, modalities_found = [], [], []
MODALITIES = ["Pre", "Sub_1", "Post_1", "Post_2", "T2"]
sample_rows = df[df["Split"]=="train"].head(20)

for _, row in sample_rows.iterrows():
    uid    = row["UID"]
    center = row["Institution"]
    folder = DATA_ROOT / center / "data_unilateral" / uid
    for mod in MODALITIES:
        f = folder / f"{mod}.nii.gz"
        if f.exists():
            img = nib.load(str(f))
            shapes.append({"UID": uid, "Modality": mod,
                           "Shape": str(img.shape),
                           "H": img.shape[0], "W": img.shape[1],
                           "D": img.shape[2] if len(img.shape)>2 else 1})
            spacings.append(img.header.get_zooms()[:3])
            modalities_found.append(mod)

df_shapes = pd.DataFrame(shapes)
print(df_shapes.groupby("Modality")[["H","W","D"]].describe())

# Plot shape distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, dim in enumerate(["H","W","D"]):
    sns.boxplot(data=df_shapes, x="Modality", y=dim, ax=axes[i])
    axes[i].set_title(f"{dim} dimension across modalities")
plt.tight_layout()
plt.savefig(OUTPUT / "mri_shapes.png", dpi=150)
print("Saved: mri_shapes.png")

# ── 5. Visualize sample slices ───────────────────────────────────────────────
print("\n--- Saving sample MRI slices ---")
sample = df[df["Split"]=="train"].iloc[0]
folder = DATA_ROOT / sample["Institution"] / "data_unilateral" / sample["UID"]
label  = sample["Lesion"]

fig, axes = plt.subplots(1, len(MODALITIES), figsize=(20, 4))
fig.suptitle(f"Sample: {sample['UID']}  |  Label: {'Malignant' if label==1 else 'Non-Malignant'}")
for i, mod in enumerate(MODALITIES):
    f = folder / f"{mod}.nii.gz"
    if f.exists():
        vol = nib.load(str(f)).get_fdata()
        mid = vol.shape[2] // 2 if vol.ndim == 3 else 0
        slc = vol[:, :, mid] if vol.ndim == 3 else vol
        axes[i].imshow(slc.T, cmap="gray", origin="lower")
        axes[i].set_title(mod)
        axes[i].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT / "sample_slices.png", dpi=150)
print("Saved: sample_slices.png")
