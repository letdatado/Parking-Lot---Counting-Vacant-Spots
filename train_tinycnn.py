from pathlib import Path
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import numpy as np

# Config
DATA_DIR = Path("data")
IMG_SIZE = 96
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("best_tinycnn.pt")

IS_WINDOWS = platform.system() == "Windows"


NUM_WORKERS = 0 if IS_WINDOWS else 2

# pin_memory only helps when moving host->GPU
# disable on pure CPU to silence warning.
PIN_MEMORY = torch.cuda.is_available()

# ----------------------------
# Model
# ----------------------------


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.backbone = nn.Sequential(
            block(3, 16),   # 96 -> 48
            block(16, 32),  # 48 -> 24
            block(32, 64),  # 24 -> 12
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 1)          # single logit (binary)
        )

    def forward(self, x):
        return self.head(self.backbone(x))  # logits


def main():
    # ----------------------------
    # Transforms
    # ----------------------------
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # ----------------------------
    # Datasets and Loaders
    # ----------------------------
    
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tfms)

    print("[INFO] class_to_idx:", train_ds.class_to_idx)
    POS_IDX = train_ds.class_to_idx["occupied"]  # treat OCCUPIED as positive class
    NEG_IDX = train_ds.class_to_idx["vacant"]

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # ----------------------------
    # Loss pos_weight (handles imbalance deterministically)
    # pos_weight = #negatives / #positives for the POSITIVE class
    # ----------------------------
    try:
        labels = train_ds.targets
    except AttributeError:
        labels = [lbl for _, lbl in train_ds.samples]
    n_pos = sum(1 for t in labels if t == POS_IDX)
    n_neg = sum(1 for t in labels if t == NEG_IDX)
    pos_weight_value = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], device=DEVICE)
    print(f"[INFO] TRAIN counts -> vacant={n_neg}, occupied={n_pos}, pos_weight={pos_weight.item():.3f}")

    # ----------------------------
    # Init model/optim/criterion
    # ----------------------------
    model = TinyCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----------------------------
    # Train/Val loops
    # ----------------------------
    def run_epoch(loader, train_mode=True):
        model.train(train_mode)
        total_loss, n, correct = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            # Convert class indices to {0,1}, where 1 == occupied (POS_IDX)
            y = (lbls == POS_IDX).float().unsqueeze(1).to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            correct += (preds.cpu().squeeze(1) == y.long().cpu().squeeze(1)).sum().item()
        return total_loss / n, correct / n

    best_val = 0.0
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(val_loader,   False)
        print(f"Epoch {ep:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | val_loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  ↳ saved best model")

    # ----------------------------
    # Test evaluation
    # ----------------------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            y_true = (lbls == POS_IDX).long().cpu().numpy()
            probs = torch.sigmoid(model(imgs)).cpu().squeeze(1).numpy()
            y_pred = (probs > 0.5).astype(np.int64)
            all_true.extend(y_true.tolist())
            all_pred .extend(y_pred.tolist())

    print("\n=== Test Classification Report ===")
    print(classification_report(all_true, all_pred, target_names=["vacant",
                                                                  "occupied"]))
    print(f"[INFO] Best validation accuracy was: {best_val:.3f}")


if __name__ == "__main__":
    # On Windows, this guard is REQUIRED when using DataLoader workers (>0).
    # We also default NUM_WORKERS=0 above, which avoids multiprocessing
    # entirely.
    # If you later bump NUM_WORKERS, keep this guard.
    # You can also optionally do:
    # import torch.multiprocessing as mp
    # mp.freeze_support()
    main()
