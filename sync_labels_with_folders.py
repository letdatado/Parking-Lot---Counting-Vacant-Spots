# What this script does?
# 1) Scans review/occupied and review/vacant.
# 2) Maps each review file back to its original dataset row in labels.csv
#    - Prefers review/_review_index.csv if present.
#    - Otherwise "adopts" files by parsing spot_XX and frame_YYYYYY from the
#      filename (e.g., spot_03__frame_001965_rect.jpg).
# 3) Sets each row's label to match the folder it lives in now.

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

RE_SPOT = re.compile(r"spot_(\d+)")
RE_FRAME = re.compile(r"frame_(\d+)")


def nowstamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_labels(labels_csv: Path):
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # Lookups
    by_orig = {r["path"]: r for r in rows}
    by_spot_frame = {}
    for r in rows:
        try:
            s = int(r["spot_id"])
            fr = int(r["frame"])
            by_spot_frame.setdefault((s, fr), []).append(r)
        except Exception:
            pass
    return rows, by_orig, by_spot_frame


def save_labels(labels_csv: Path, rows):
    with open(labels_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["path","spot_id","frame","label"])
        wr.writeheader()
        wr.writerows(rows)


def load_index(index_csv: Path):
    if not index_csv.exists():
        return []
    with open(index_csv, newline="") as f:
        return list(csv.DictReader(f))


def save_index(index_csv: Path, rows):
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(index_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["review_rel_path","orig_path","spot_id","frame"])
        wr.writeheader()
        wr.writerows(rows)


def parse_spot_frame(name: str):
    sm = RE_SPOT.search(name)
    fm = RE_FRAME.search(name)
    if not sm or not fm:
        return None
    return int(sm.group(1)), int(fm.group(1))


def main():
    ap = argparse.ArgumentParser(description="Sync labels.csv from review folders (with real backup + dry-run + flips).")
    ap.add_argument("--labels", default="labels.csv", help="Path to labels.csv")
    ap.add_argument("--review-dir", default="review", help="Path to review directory")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes; do not write labels.csv")
    ap.add_argument("--backup", action="store_true", help="Save a timestamped backup of labels.csv before writing")
    args = ap.parse_args()

    labels_csv = Path(args.labels)
    review_dir = Path(args.review_dir)
    index_csv = review_dir / "_review_index.csv"
    occ_dir = review_dir / "occupied"
    vac_dir = review_dir / "vacant"

    if not labels_csv.exists():
        raise SystemExit(f"ERROR: {labels_csv} not found.")
    if not occ_dir.exists() or not vac_dir.exists():
        raise SystemExit("ERROR: review/occupied or review/vacant not found.")

    # Load current labels & optional index
    rows, by_orig, by_spot_frame = load_labels(labels_csv)
    rows_before = [dict(r) for r in rows]  # deep-ish copy for backup & flips
    index_rows = load_index(index_csv)
    review_to_orig = {r["review_rel_path"]: r["orig_path"] for r in index_rows}

    def adopt(rel_path: str, abs_path: Path):
        """Map review file -> original dataset row (path, spot, frame)."""
        # Use index if known
        orig = review_to_orig.get(rel_path)
        if orig and orig in by_orig:
            r = by_orig[orig]
            return orig, int(r["spot_id"]), int(r["frame"])
        # Parse spot/frame from filename and find row
        parsed = parse_spot_frame(abs_path.name)
        if not parsed:
            print(f"NOTE: Could not parse spot/frame: {rel_path}")
            return None
        spot, frame = parsed
        candidates = by_spot_frame.get((spot, frame), [])
        if not candidates:
            print(f"NOTE: No labels.csv row with spot={spot}, frame={frame} for {rel_path}")
            return None
        # Prefer exact basename match after the 'spot_XX__' prefix
        base = abs_path.name.split("__", 1)[-1]
        chosen = None
        for r in candidates:
            if Path(r["path"]).name == base:
                chosen = r
                break
        if chosen is None:
            chosen = candidates[0]
        return chosen["path"], spot, frame

    # Build pending updates by scanning both folders
    pending = []   # (orig_path, new_label, rel_review_path, spot, frame)
    for subdir, new_label in [(occ_dir, "occupied"), (vac_dir, "vacant")]:
        for p in subdir.rglob("*"):
            if not p.is_file(): continue
            rel = p.relative_to(review_dir).as_posix()
            adopted = adopt(rel, p)
            if adopted:
                orig_path, spot, frame = adopted
                pending.append((orig_path, new_label, rel, spot, frame))

    # Apply updates (in-memory)
    changed = 0
    touched = set()
    new_index_rows = []
    for orig_path, new_label, rel, spot, frame in pending:
        r = by_orig.get(orig_path)
        if not r:
            print(f"WARNING: Original path not found in labels.csv: {orig_path}")
            continue
        if orig_path in touched:
            # only apply once per unique orig_path
            pass
        touched.add(orig_path)
        if r["label"] != new_label:
            r["label"] = new_label
            changed += 1
        new_index_rows.append({
            "review_rel_path": rel,
            "orig_path": orig_path,
            "spot_id": spot,
            "frame": frame,
        })

    # Prepare flips report (compare rows_before vs current rows)
    # Key by (path, spot_id, frame) to be robust even if basenames repeat
    before_map = {(rb["path"], rb["spot_id"], rb["frame"]): rb["label"] for rb in rows_before}
    after_map = {(ra["path"], ra["spot_id"], ra["frame"]): ra["label"] for ra in rows}
    flips = []
    for key, old_label in before_map.items():
        new_label = after_map.get(key, old_label)
        if new_label != old_label:
            path, spot_id, frame = key
            flips.append({
                "path": path, "spot_id": spot_id, "frame": frame,
                "old_label": old_label, "new_label": new_label
            })

    ts = nowstamp()
    flips_csv = labels_csv.with_name(f"label_flips_{ts}.csv")

    if args.dry_run:
        print(f"[DRY-RUN] Would update {changed} row(s). No files written.")
        if flips:
            with open(flips_csv, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["path","spot_id","frame","old_label","new_label"])
                wr.writeheader(); wr.writerows(flips)
            print(f"[DRY-RUN] A flips preview was written to {flips_csv}")
        return

    # Write true pre-change backup if requested
    if args.backup:
        backup_csv = labels_csv.with_name(f"labels_backup_{ts}.csv")
        save_labels(backup_csv, rows_before)
        print(f"Backup saved to {backup_csv}")

    # Save updated labels + refreshed index
    save_labels(labels_csv, rows)
    
    # Keep old index rows for files we didn't see this run
    seen_rels = {r["review_rel_path"] for r in new_index_rows}
    for r in index_rows:
        if r["review_rel_path"] not in seen_rels:
            new_index_rows.append(r)
    save_index(index_csv, new_index_rows)

    print(f"Sync complete. Updated {changed} row(s) in {labels_csv}.")
    if flips:
        with open(flips_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["path","spot_id","frame","old_label","new_label"])
            wr.writeheader()
            wr.writerows(flips)
        print(f"Wrote flips report to {flips_csv}")
    else:
        print("No label changes detected.")


if __name__ == "__main__":
    main()
