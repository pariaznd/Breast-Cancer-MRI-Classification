import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_384 import OdeliaDataset384 as OdeliaDataset
from densenet169_384 import get_model
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time

INSTITUTIONS_TRAIN = ["CAM"]
INSTITUTIONS_VAL = ["CAM", "MHA", "RUMC", "UKA"]
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = Path("results/densenet169_384_best.pth")
SAVE_PATH.parent.mkdir(exist_ok=True)

train_ds = OdeliaDataset(INSTITUTIONS_TRAIN, split="train")
val_ds = OdeliaDataset(INSTITUTIONS_VAL, split="val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)

labels = [s[0]["label"] for s in train_ds.samples]
counts = np.bincount(labels)
weights = 1.0 / counts
weights = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(DEVICE)

model = get_model(num_classes=3).to(DEVICE)
if SAVE_PATH.exists():
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    print(f"Resumed from {SAVE_PATH}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=weights)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_auc = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            out = torch.softmax(model(x), dim=1)
            all_probs.append(out.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    binary_labels = (all_labels == 2).astype(int)
    auc = roc_auc_score(binary_labels, all_probs[:, 2])

    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_dl):.4f} | AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Saved best model (AUC: {best_auc:.4f})")

elapsed = (time.time() - start_time) / 3600
print(f"\nDone! Best AUC: {best_auc:.4f} | Time: {elapsed:.2f}h")
