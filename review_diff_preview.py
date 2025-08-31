# What this script does?
# Shows which images in review/occupied|vacant would change
# their label in labels.csv (i.e., "mismatches").

import csv
import re
from pathlib import Path

LABELS = Path("labels.csv")
REVIEW = Path("review")
OCC = REVIEW / "occupied"
VAC = REVIEW / "vacant"

RE_SPOT = re.compile(r"spot_(\d+)")
RE_FRAME = re.compile(r"frame_(\d+)")


def parse_spot_frame(name: str):
    sm = RE_SPOT.search(name)
    fm = RE_FRAME.search(name)
    if not sm or not fm: return None
    return int(sm.group(1)), int(fm.group(1))


# load labels into lookups


by_path = {}
by_spot_frame = {}
with open(LABELS, newline="") as f:
    for r in csv.DictReader(f):
        by_path[r["path"]] = r
        try:
            k = (int(r["spot_id"]), int(r["frame"]))
            by_spot_frame.setdefault(k, []).append(r)
        except:
            pass


def expected_changes(dirpath: Path, target_label: str):
    mismatches, missing = [], []
    for p in dirpath.rglob("*"):
        if not p.is_file(): continue
        # filenames should look like: spot_XX__frame_YYYYYY_rect.jpg
        parsed = parse_spot_frame(p.name)
        if not parsed:
            missing.append((p, "cannot-parse-spot/frame"))
            continue
        spot, frame = parsed
        candidates = by_spot_frame.get((spot, frame), [])
        if not candidates:
            missing.append((p, f"no labels row for (spot={spot}, frame={frame})"))
            continue
        # choose the row whose basename matches after 'spot_XX__'
        base = p.name.split("__", 1)[-1]
        row = None
        for c in candidates:
            if Path(c["path"]).name == base:
                row = c; break
        if row is None:
            row = candidates[0]
        if row["label"] != target_label:
            mismatches.append((p, row["label"], target_label, row["path"]))
    return mismatches, missing


occ_mis, occ_missing = expected_changes(OCC, "occupied")
vac_mis, vac_missing = expected_changes(VAC, "vacant")


print("=== Review Diff Preview ===")
print(f"Occupied folder: {len(occ_mis)} file(s) would change to 'occupied'")
for p, old, new, orig in occ_mis[:20]:
    print(f"  OLD={old:8s} -> NEW={new:8s} | {p.name} | orig={orig}")

print(f"\nVacant folder:   {len(vac_mis)} file(s) would change to 'vacant'")
for p, old, new, orig in vac_mis[:20]:
    print(f"  OLD={old:8s} -> NEW={new:8s} | {p.name} | orig={orig}")

if occ_missing or vac_missing:
    print("\nNotes (unmatched files):")
    for p, why in (occ_missing + vac_missing)[:20]:
        print(f"  {p} -> {why}")

print("\nTip: If you *expect* changes but see 0 here, be sure you moved files in 'review/occupied' and 'review/vacant',")
print("and that filenames look like 'spot_XX__frame_YYYYYY_...'.")
