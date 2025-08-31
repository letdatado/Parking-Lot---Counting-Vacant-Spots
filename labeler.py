# What this script does?
# Simple OpenCV labeler for parking crops.
# Produces labels.csv with columns: path,spot_id,frame, label
# label in {occupied, vacant, unknown}

import csv
import re
from pathlib import Path
import cv2

CROPS_DIR = Path("crops")  # <- change if needed
CSV_OUT = Path("labels.csv")


def iter_images(root):
    # prefer rect.jpg; if missing, use mask.jpg
    for spot_dir in sorted(root.glob("spot_*")):
        for jpg in sorted(spot_dir.glob("*.jpg")):
            # pick _rect first
            if "_rect.jpg" in str(jpg):
                yield jpg
        for jpg in sorted(spot_dir.glob("*.jpg")):
            if "_mask.jpg" in str(jpg):
                # only yield mask if there's no rect for that frame
                frame_id = re.search(r"frame_(\d+)_", jpg.name).group(1)
                rect = jpg.with_name(f"frame_{frame_id}_rect.jpg")
                if not rect.exists():
                    yield jpg


def parse_meta(p: Path):
    # path like crops/spot_03/frame_000123_rect.jpg
    spot = int(re.search(r"spot_(\d+)", str(p)).group(1))
    frame = int(re.search(r"frame_(\d+)", p.name).group(1))
    return spot, frame


def main():
    # resume if labels.csv exists
    done = set()
    if CSV_OUT.exists():
        with open(CSV_OUT, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["path"])

    with open(CSV_OUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path","spot_id","frame","label"])
        if CSV_OUT.stat().st_size == 0:
            writer.writeheader()

        imgs = list(iter_images(CROPS_DIR))
        i = 0
        while 0 <= i < len(imgs):
            p = imgs[i]
            if str(p) in done:
                i += 1
                continue

            img = cv2.imread(str(p))
            if img is None:
                i += 1
                continue

            spot, frame = parse_meta(p)
            vis = img.copy()
            cv2.putText(vis, f"{p.name} | spot {spot} | frame {frame}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, "[o] occupied  [v] vacant  [u] unknown  [b] back  [q] quit",
                        (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("label", vis)
            k = cv2.waitKey(0) & 0xFF

            if k == ord('q'):
                break
            if k == ord('b'):
                i = max(-1, i-2)  # step back one image on next loop
            elif k in (ord('o'), ord('v'), ord('u')):
                label = {ord('o'):"occupied", ord('v'):"vacant", ord('u'):"unknown"}[k]
                writer.writerow({"path": str(p), "spot_id": spot, "frame": frame, "label": label})
                f.flush()
                done.add(str(p))
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
