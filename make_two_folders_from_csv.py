# What this script does?
# Build two review folders from labels.csv:
#   review/
#     occupied/
#     vacant/
#
# Each image is copied (or hardlinked) and renamed:
#   spot_{spot_id:02d}__{original_filename}.jpg
#
# We also write an index:
#   review/_review_index.csv with columns:
#     review_rel_path,orig_path,spot_id,frame

import argparse
import csv
import os
from pathlib import Path
import shutil


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, hardlink: bool):
    """
    Copy by default (simplest and always works).
    If --hardlink is requested, try os.link (fast, saves disk).
    Falls back to copy if hardlink fails (e.g., different drive).
    """
    if hardlink:
        try:
            if dst.exists():
                return
            os.link(src, dst)
            return
        except Exception:
            pass  # fallback to copy
    if not dst.exists():
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Create review/occupied & review/vacant from labels.csv")
    ap.add_argument("--labels", type=str, default="labels.csv", help="Path to labels.csv")
    ap.add_argument("--review-dir", type=str, default="review", help="Where to create review folders")
    ap.add_argument("--hardlink", action="store_true", help="Try to hardlink instead of copying")
    args = ap.parse_args()

    labels_csv = Path(args.labels)
    review_dir = Path(args.review_dir)
    occ_dir = review_dir / "occupied"
    vac_dir = review_dir / "vacant"
    index_csv = review_dir / "_review_index.csv"

    # Prepare dirs
    ensure_dir(occ_dir)
    ensure_dir(vac_dir)

    # Load labels
    if not labels_csv.exists():
        raise SystemExit(f"ERROR: {labels_csv} not found.")

    rows = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # keep only those that are labeled occupied/vacant
            if r["label"] in ("occupied", "vacant"):
                rows.append(r)

    if not rows:
        raise SystemExit("No occupied/vacant rows found in labels.csv")

    # Build review set
    index_rows = []
    copied = 0
    missing = 0

    for r in rows:
        orig_path = Path(r["path"])
        spot_id = int(r["spot_id"])
        frame = int(r["frame"])
        label = r["label"]
        if not orig_path.exists():
            print(f"WARNING: missing file: {orig_path}")
            missing += 1
            continue

        dest_dir = occ_dir if label == "occupied" else vac_dir
        # New name to guarantee uniqueness across spots
        new_name = f"spot_{spot_id:02d}__{orig_path.name}"
        dest_path = dest_dir / new_name

        copy_or_link(orig_path, dest_path, hardlink=args.hardlink)
        copied += 1

        # Store relative path (relative to review dir) so it remains portable
        review_rel = dest_path.relative_to(review_dir).as_posix()
        index_rows.append({
            "review_rel_path": review_rel,
            "orig_path": str(orig_path),
            "spot_id": spot_id,
            "frame": frame,
        })

    # Write index
    with open(index_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["review_rel_path","orig_path","spot_id","frame"])
        wr.writeheader()
        wr.writerows(index_rows)

    print(f"Done. Indexed {len(index_rows)} images (copied/linked: {copied}, missing: {missing}).")
    print(f"Open these folders in Explorer and review:")
    print(f"  {occ_dir}")
    print(f"  {vac_dir}")
    print("When you finish moving mislabels, run sync_labels_with_folders.py to update labels.csv.")


if __name__ == "__main__":
    main()
