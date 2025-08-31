# What this script does?
# Compute per-spot decision thresholds using the validation split.
# Requires you used the "spot_{id}__..." filename prefix in make_dataset.py

import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import cv2

RE_SPOT = re.compile(r"spot_(\d+)__")


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.net = nn.Sequential(
            block(3,16), block(16,32), block(32,64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,1)
        )
        
    def forward(self,x): 
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/val", help="validation dir with vacant/ and occupied/")
    ap.add_argument("--model", default="best_tinycnn.pt")
    ap.add_argument("--out",   default="thresholds.json")
    ap.add_argument("--min_thr", type=float, default=0.20)
    ap.add_argument("--max_thr", type=float, default=0.80)
    ap.add_argument("--steps",   type=int,   default=61)  # 0.20..0.80 step ~0.01
    args = ap.parse_args()

    tfm = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    ds = datasets.ImageFolder(args.data, transform=tfm)
    class_to_idx = ds.class_to_idx  # {'occupied':0,'vacant':1} or vice-versa
    pos_is = class_to_idx["occupied"]  # treat occupied as positive

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Collect per-spot probabilities and labels
    per_spot = {}  # id -> {'p':[], 'y':[]}
    with torch.no_grad():
        for img, y_idx in ds:
            # parse spot from filename
            path = ds.samples[len(per_spot.get("_seen", [])) if "_seen" in per_spot else 0][0]  # hacky? let's get name directly
        # Better: iterate over raw samples for filenames:
    per_spot = {}
    for (path, y_idx) in ds.samples:
        m = RE_SPOT.search(Path(path).name)
        if not m: 
            # if missing prefix, lump into -1 (global)
            spot = -1
        else:
            spot = int(m.group(1))
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (96,96), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        img = (img - 0.5)/0.5
        t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(t)).cpu().numpy().squeeze().item()  # prob(occupied)
        y = 1 if y_idx == pos_is else 0
        per_spot.setdefault(spot, {"p":[], "y": []})
        per_spot[spot]["p"].append(p)
        per_spot[spot]["y"].append(y)

    # Sweep thresholds and choose the one with best F1 per spot
    thrs = np.linspace(args.min_thr, args.max_thr, args.steps)
    best = {}
    for spot, d in per_spot.items():
        ps = np.array(d["p"]); ys = np.array(d["y"])
        if ys.sum()==0 or ys.sum()==len(ys):
            # single-class in val -> keep 0.5
            best[spot] = 0.5
            continue
        f1s = []
        for t in thrs:
            preds = (ps >= t).astype(np.int32)
            f1s.append(f1_score(ys, preds))
        t_star = thrs[int(np.argmax(f1s))]
        best[spot] = float(t_star)

    # Save with spot ids as strings to keep it JSON-friendly
    out = {str(k): v for k,v in best.items() if k>=0}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out} with {len(out)} per-spot thresholds.")


if __name__ == "__main__":
    main()
