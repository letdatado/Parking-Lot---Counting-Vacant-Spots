# What this script does?
# Print overall counts, plus per-spot counts of occupied/vacant/unknown.
# Writes per_spot_counts.csv for easy viewing.

import csv
from collections import Counter, defaultdict
from pathlib import Path

LABELS = Path("labels.csv")

overall = Counter()
per_spot = defaultdict(Counter)

with open(LABELS, newline="") as f:
    for r in csv.DictReader(f):
        lbl = r["label"]
        spot = int(r["spot_id"])
        overall[lbl] += 1
        per_spot[spot][lbl] += 1

print("\n=== Overall counts ===")
for k, v in overall.items():
    print(f"{k:8s}: {v}")

print("\n=== Per-spot counts ===")
print("spot, occupied, vacant, unknown, total")
rows = []
for spot in sorted(per_spot.keys()):
    occ = per_spot[spot]["occupied"]
    vac = per_spot[spot]["vacant"]
    unk = per_spot[spot]["unknown"]
    tot = occ + vac + unk
    print(f"{spot:4d}, {occ:8d}, {vac:6d}, {unk:7d}, {tot:5d}")
    rows.append({"spot": spot, "occupied": occ, "vacant": vac, "unknown": unk, "total": tot})

with open("per_spot_counts.csv", "w", newline="") as f:
    import csv
    wr = csv.DictWriter(f, fieldnames=["spot", "occupied", "vacant", "unknown", "total"])
    wr.writeheader()
    wr.writerows(rows)

print("\nSaved per_spot_counts.csv")
