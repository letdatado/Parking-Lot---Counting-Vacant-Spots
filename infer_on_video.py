# What this script does?
# Use best_tinycnn.pt to classify each parking stall per frame.
# - Loads ROIs from rois.json (your polygons)
# - Warps each ROI to 96x96 (minAreaRect if not 4 points)
# - Batches all spots -> one forward pass per frame
# - Applies temporal smoothing (majority vote) and optional hysteresis
# - Draws green (vacant) / red (occupied) polygons and prints counts

import argparse
import json 
import collections
import cv2
import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Small helpers 
# ----------------------------


def order_quad(pts4):
    """
    Given 4 points, return them ordered as TL, TR, BR, BL.
    Stable ordering makes perspective warp consistent.
    """
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)         # x+y
    d = np.diff(pts, axis=1)    # x-y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def dedup_close(poly):
    """
    Many polygon annotators repeat the first point as the last point.
    Remove the duplicate closing point if present.
    """
    if len(poly) >= 2 and poly[0][0] == poly[-1][0] and poly[0][1] == poly[-1][1]:
        return poly[:-1]
    return poly


def poly_to_quad(poly):
    """
    Convert any polygon (>=3 pts) to a 4-pt quadrilateral using minAreaRect.
    This is robust for skewed stalls and guarantees 4 points for warping.
    """
    cnt = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_quad(box)


def warp_to_square(frame, quad, out=96):
    """
    Perspective-warp a quadrilateral region into a square crop (out x out).
    """
    dst = np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, H, (out, out), flags=cv2.INTER_LINEAR)


# -----------------------------------
# Model (must match training EXACTLY)
# -----------------------------------


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
        # same layer names as in training: backbone + head
        self.backbone = nn.Sequential(
            block(3, 16),   # 96 -> 48
            block(16, 32),  # 48 -> 24
            block(32, 64),  # 24 -> 12
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 1)          # single logit
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x  # logits (no sigmoid)


def preprocess_batch(crops_bgr, img_size=96):
    """
    Convert a list of BGR crops into a normalized Tensor batch:
    - BGR -> RGB
    - resize to img_size
    - scale to [0,1], then normalize to mean=0.5, std=0.5 => [-1,1]
    - shape: (N, 3, H, W)
    """
    arr = []
    for img in crops_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size), 
                         interpolation=cv2.INTER_LINEAR)
        t = rgb.astype(np.float32) / 255.0
        t = (t - 0.5) / 0.5
        t = t.transpose(2, 0, 1)  # CHW
        arr.append(t)
    batch = torch.from_numpy(np.stack(arr, axis=0))
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--rois",  required=True, help="rois.json from annotation")
    ap.add_argument("--model", default="best_tinycnn.pt", help="trained weights")
    ap.add_argument("--rect_size", type=int, default=96)
    ap.add_argument("--smooth", type=int, default=7, help="majority window per spot")
    ap.add_argument("--hysteresis_hi", type=float, default=None, help="e.g., 0.8 to switch to occupied")
    ap.add_argument("--hysteresis_lo", type=float, default=None, help="e.g., 0.2 to switch back to vacant")
    ap.add_argument("--thresholds_json", type=str, default=None, help="optional per-spot thresholds JSON")
    ap.add_argument("--write", type=str, default=None, help="optional output video path")
    args = ap.parse_args()

    # Load ROIs and precompute quads to speed up
    with open(args.rois, "r") as f:
        meta = json.load(f)
    polys = [dedup_close([(int(x), int(y)) for x, y in poly]) for poly in meta["polygons"]]
    quads = []
    for poly in polys:
        if len(poly) == 4:
            quads.append(order_quad(poly))
        else:
            quads.append(poly_to_quad(poly))
    n_spots = len(quads)

    # Load per-spot thresholds (fallback to 0.5)
    spot_thresh = [0.5] * n_spots
    if args.thresholds_json:
        with open(args.thresholds_json, "r") as f:
            td = json.load(f)
        for i in range(n_spots):
            if str(i) in td:
                spot_thresh[i] = float(td[str(i)])

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Temporal smoothing buffers (per-spot)
    bufs = [collections.deque(maxlen=args.smooth) for _ in range(n_spots)]
    # Hysteresis states (0=vacant, 1=occupied)
    states = [0] * n_spots

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video")

    writer = None
    if args.write:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.write, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Warp all stalls to square crops
        crops = [warp_to_square(frame, q.astype(np.float32), out=args.rect_size) for q in quads]

        # Batch inference (probability of OCCUPIED)
        batch = preprocess_batch(crops, img_size=args.rect_size).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).cpu().numpy().squeeze(1)  # shape: (n_spots,)

        # Decide label per spot (threshold OR hysteresis) + majority vote
        labels = []
        for i, p in enumerate(probs):
            thr = spot_thresh[i]
            if args.hysteresis_hi is not None and args.hysteresis_lo is not None:
                # Hysteresis: switch to occupied if p >= hi; to vacant if p <= lo; else keep previous state
                if p >= args.hysteresis_hi:
                    states[i] = 1
                elif p <= args.hysteresis_lo:
                    states[i] = 0
                pred = states[i]
            else:
                pred = 1 if p >= thr else 0

            # Majority vote smoothing over last K predictions
            bufs[i].append(pred)
            vote = 1 if sum(bufs[i]) > (len(bufs[i]) // 2) else 0
            labels.append(vote)

        # Draw overlays
        occ = sum(labels)
        vac = len(labels) - occ
        for i, (poly, lbl) in enumerate(zip(polys, labels)):
            color = (0, 0, 255) if lbl == 1 else (0, 255, 0)  # red occupied / green vacant
            cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, color, 2)
            # tiny id tag near the centroid
            xs = [x for x, y in poly]; ys = [y for x, y in poly]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Vacant: {vac}  Occupied: {occ}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Vacant: {vac}  Occupied: {occ}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("parking", frame)
        if writer is not None:
            writer.write(frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
