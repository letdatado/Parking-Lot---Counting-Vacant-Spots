# What this script does?
# Diagnostic role, Compare two CSVs (old vs new) and list rows whose label changed.
# Writes label_flips.csv with: path, spot_id, frame, old_label, new_label

import csv
import argparse


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = r["path"]                    # use path as stable ID
            rows[key] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="labels_backup.csv")
    ap.add_argument("--new", default="labels.csv")
    ap.add_argument("--out", default="label_flips.csv")
    args = ap.parse_args()

    old = load(args.old)
    new = load(args.new)

    flips = []
    for key, rnew in new.items():
        rold = old.get(key)
        if not rold: 
            continue
        if rold["label"] != rnew["label"]:
            flips.append({
                "path": key,
                "spot_id": rnew["spot_id"],
                "frame": rnew["frame"],
                "old_label": rold["label"],
                "new_label": rnew["label"],
            })

    print(f"Found {len(flips)} label change(s). Writing {args.out}")
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["path","spot_id","frame","old_label","new_label"])
        wr.writeheader()
        wr.writerows(flips)


if __name__ == "__main__":
    main()
