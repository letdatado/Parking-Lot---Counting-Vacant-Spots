# What this script does?
# Carve Parking Spots from a Fixed-Camera Video
# - Mode 1: "annotate" -> click polygons for each parking spot, save to JSON
# - Mode 2: "crop"     -> load polygons, sample frames, carve crops per spot
#
# Why this approach?
# - Fixed camera => draw each ROI once
# - JSON for ROIs => repeatable + shareable
# - If ROI has 4 points => also do perspective warp to 96x96 (great for CNN)
# - If ROI has >4 points => still crop with a polygon mask + tight bbox
# - FPS sampling avoids near-duplicate frames, reduces storage + training bias

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


# Utility: ensure dir exists


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Utility: order 4 polygon points (TL,TR,BR,BL)


def order_quad(pts4):
    # pts4: shape (4,2)
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)           # x+y
    diff = np.diff(pts, axis=1)   # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# Draw polygon interactively:
# - Left click to add points
# - Right click to UNDO last point
# - Press 'n' to finalize current polygon and move to next
# - Press 'q' to quit early (saves what you have)

def annotate_polygons(video_path, save_path, expected_spots=15):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read the first frame from video.")

    clone = frame.copy()
    h, w = frame.shape[:2]
    polys = []         # list of list-of-points
    current = []       # collecting points for the current polygon

    win = "annotate: left=add, right=undo, n=next polygon, q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))          # add point
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()                   # undo last point

    cv2.setMouseCallback(win, mouse_cb)

    while True:
        vis = frame.copy()

        # draw already saved polygons
        for poly in polys:
            if len(poly) >= 2:
                cv2.polylines(vis, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
            for (px, py) in poly:
                cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)

        # draw current polygon-in-progress
        if len(current) >= 1:
            for i, (px, py) in enumerate(current):
                cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(vis, current[i-1], current[i], (0, 0, 255), 1)

        # HUD text
        msg = f"Polygons: {len(polys)}/{expected_spots} | Points in current: {len(current)}"
        cv2.putText(vis, msg, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('n'):  # finalize current polygon
            if len(current) >= 3:
                polys.append(current.copy())
                current = []
            else:
                print("Need at least 3 points for a polygon.")
            if len(polys) >= expected_spots:
                break

        elif key == ord('q'):
            # allow quitting early, saving what we have
            break

    cv2.destroyWindow(win)

    # save polygons to JSON (list of lists of [x,y])
    # we also store image size for sanity checks later
    data = {
        "image_size": {"w": w, "h": h},
        "polygons": [[[int(x), int(y)] for (x, y) in poly] for poly in polys]
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(polys)} polygons to {save_path}")

# Crop helpers
# - mask_crop_polygon: crops a tight bbox around polygon + masked background
# - warp_quad_to_square: perspective warp if polygon has 4 points


def mask_crop_polygon(frame, poly):
    """Return a masked crop (tight bbox) of an arbitrary polygon (>=3 points)."""
    poly_np = np.array(poly, dtype=np.int32)

    # create mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 255)

    # apply mask
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # tight bbox around polygon
    x, y, w, h = cv2.boundingRect(poly_np)
    crop = masked[y:y+h, x:x+w].copy()
    return crop


def warp_quad_to_square(frame, poly, out_size=96):
    """If poly has 4 points, warp to a fixed square (out_size x out_size)."""
    if len(poly) != 4:
        return None
    src = order_quad(poly)  # TL,TR,BR,BL
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
    return warped


def deduplicate_closing_point(poly):
    """
    Many annotators export polygons with the last point == first point.
    This function removes that duplicate if present.
    Input: list of (x, y)
    Output: list of (x, y) without duplicate closing point
    """
    if len(poly) >= 2 and poly[0][0] == poly[-1][0] and poly[0][1] == poly[-1][1]:
        return poly[:-1]
    return poly


def polygon_to_minarea_quad(poly):
    """
    Convert an arbitrary polygon (>=3 pts) to a 4-point quadrilateral using
    cv2.minAreaRect -> cv2.boxPoints. This gives a stable, rotated rectangle
    that tightly covers the polygon area. Great for perspective warping.

    Returns: 4 points as a list of (x, y) float tuples, ordered for warp via order_quad.
    """
    cnt = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)               # ((cx,cy),(w,h),theta)
    box  = cv2.boxPoints(rect)                # 4x2 float32 (unordered)
    box  = box.astype(np.float32)
    # order to TL, TR, BR, BL for a stable warp
    ordered = order_quad(box)
    return [(float(x), float(y)) for x, y in ordered]


# Iterate video with FPS sampling, carve crops per polygon
def carve_from_video(video_path, rois_path, out_dir, fps=2, rect_size=96, start_at=0, end_at=None):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    with open(rois_path, "r") as f:
        meta = json.load(f)
    polys = meta["polygons"]  # list of polygons; each polygon is list of [x,y]

    # Make a folder per spot
    for i in range(len(polys)):
        ensure_dir(out_dir / f"spot_{i:02d}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(vid_fps / float(fps))))  # sample every Nth frame

    # set start/end frame bounds
    start_frame = int(start_at * vid_fps) if isinstance(start_at, (int, float)) else int(start_at)
    end_frame = total_frames - 1 if end_at is None else int(end_at * vid_fps) if isinstance(end_at, (int, float)) else int(end_at)
    start_frame = max(0, min(total_frames-1, start_frame))
    end_frame   = max(0, min(total_frames-1, end_frame))
    if end_frame < start_frame:
        end_frame = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_count = 0

    print(f"Video FPS: {vid_fps:.2f} | Sampling every {frame_interval} frames (~{fps} fps)")
    print(f"Processing frames [{start_frame}, {end_frame}] out of {total_frames}")

    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > end_frame:
            break

        ok, frame = cap.read()
        if not ok:
            break

        # only process sampled frames
        if (pos - start_frame) % frame_interval != 0:
            continue

        # Carve each polygon
        for i, poly in enumerate(polys):
            poly_int = [(int(x), int(y)) for x, y in poly]

            # masked tight crop (works for any polygon)
            mask_crop = mask_crop_polygon(frame, poly_int)
            mask_path = out_dir / f"spot_{i:02d}" / f"frame_{pos:06d}_mask.jpg"
            cv2.imwrite(str(mask_path), mask_crop)

            # if 4 points, also save a rectified (warped) 96x96 crop
            # --- Clean polygon (remove duplicate closing point if present)
            poly_clean = deduplicate_closing_point(poly_int)

            # masked tight crop (works for any polygon)
            mask_crop = mask_crop_polygon(frame, poly_clean)
            mask_path = out_dir / f"spot_{i:02d}" / f"frame_{pos:06d}_mask.jpg"
            cv2.imwrite(str(mask_path), mask_crop)

            # Try to produce a rectified crop even if not exactly 4 clicks:
            # We convert the polygon to a minimum-area quadrilateral (very stable)
            try:
                quad = polygon_to_minarea_quad(poly_clean)  # always 4 points
                rect = warp_quad_to_square(frame, quad, out_size=rect_size)
                rect_path = out_dir / f"spot_{i:02d}" / f"frame_{pos:06d}_rect.jpg"
                cv2.imwrite(str(rect_path), rect)
            except Exception as e:
                # If something goes wrong (shouldn't, but being safe), we still keep the mask crop.
                # print(e) for debugging.
                pass

        saved_count += 1

    cap.release()
    print(f"Done. Processed ~{saved_count} sampled frames. Crops saved under: {out_dir}")


# Main (argparse)
def main():
    parser = argparse.ArgumentParser(description="Carve parking spots from a fixed-camera video.")
    parser.add_argument("--mode", choices=["annotate","crop"], required=True, help="annotate: click ROIs; crop: carve crops")
    parser.add_argument("--video", type=str, required=True, help="path to video")
    parser.add_argument("--out_rois", type=str, help="where to save ROIs JSON (annotate mode)")
    parser.add_argument("--rois", type=str, help="ROIs JSON file (crop mode)")
    parser.add_argument("--out_dir", type=str, help="output folder for crops (crop mode)")
    parser.add_argument("--fps", type=float, default=2.0, help="sampling FPS for cropping")
    parser.add_argument("--rect_size", type=int, default=96, help="output size for rectified crops")
    parser.add_argument("--expected_spots", type=int, default=15, help="how many spots you plan to draw")
    parser.add_argument("--start_at", type=float, default=0, help="start time in seconds (or frame index if int)")
    parser.add_argument("--end_at", type=float, default=None, help="end time in seconds (or frame index if int)")

    args = parser.parse_args()
    video_path = Path(args.video)

    if args.mode == "annotate":
        assert args.out_rois, "--out_rois is required for annotate mode"
        annotate_polygons(video_path, args.out_rois, expected_spots=args.expected_spots)

    elif args.mode == "crop":
        assert args.rois and args.out_dir, "--rois and --out_dir are required for crop mode"
        carve_from_video(
            video_path=video_path,
            rois_path=args.rois,
            out_dir=args.out_dir,
            fps=args.fps,
            rect_size=args.rect_size,
            start_at=args.start_at,
            end_at=args.end_at
        )


if __name__ == "__main__":
    main()
