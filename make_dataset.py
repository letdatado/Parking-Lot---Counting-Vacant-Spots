# What this script does?
# Build data/train|val|test/occupied|vacant from labels.csv
# - Drops "unknown"
# - Splits by time per spot (70/15/15) ONLY for spots that have BOTH classes
# - Single-class spots (all occupied or all vacant) go entirely to TRAIN
# - Renames copies to: spot_{spot_id}__{original_filename}.jpg
#   so we can analyze per-spot later.


import csv
import shutil
from pathlib import Path

LABELS_CSV = Path("labels.csv")
OUT = Path("data")
SPLIT = (0.70, 0.15, 0.15)  # train, val, test


def ensure(p: Path): 
    p.mkdir(parents=True, exist_ok=True)


def main():
    # load labeled rows (occupied/vacant only)
    rows = []
    with open(LABELS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] in ("occupied","vacant"):
                r["frame"] = int(r["frame"])
                r["spot_id"] = int(r["spot_id"])
                rows.append(r)

    # group by spot
    by_spot = {}
    for r in rows:
        by_spot.setdefault(r["spot_id"], []).append(r)

    # find spots that contain both classes (for val/test)
    spots_with_both = set()
    for spot_id, items in by_spot.items():
        labs = {it["label"] for it in items}
        if "occupied" in labs and "vacant" in labs:
            spots_with_both.add(spot_id)

    # prepare dirs
    for split in ["train","val","test"]:
        for cls in ["occupied","vacant"]:
            ensure(OUT/split/cls)

    # split + copy (renaming with spot prefix)
    moved = 0
    for spot_id, items in by_spot.items():
        items.sort(key=lambda r: r["frame"])  # chronological
        n = len(items)
        n_train = int(n*SPLIT[0])
        n_val   = int(n*SPLIT[1])

        if spot_id not in spots_with_both:
            # Single-class spot: all to TRAIN, keeps metrics honest
            parts = [("train", items)]
        else:
            parts = [
                ("train", items[:n_train]),
                ("val",   items[n_train:n_train+n_val]),
                ("test",  items[n_train+n_val:]),
            ]

        for split, lst in parts:
            for r in lst:
                src = Path(r["path"])
                # Prefix spot id so we can recover per-spot metrics later
                dst = OUT/split/r["label"]/f"spot_{int(r['spot_id']):02d}__{src.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                moved += 1

    print(f"Done. Copied {moved} images into {OUT}/train|val|test/occupied|vacant")
    print(f"Spots used in val/test (have both classes): {sorted(spots_with_both)}")
    single_class = sorted(set(by_spot.keys()) - spots_with_both)
    if single_class:
        print(f"Single-class spots (train-only): {single_class}")


if __name__ == "__main__":
    main()
