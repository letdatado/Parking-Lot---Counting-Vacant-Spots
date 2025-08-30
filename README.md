# AI Parking Lot CV — Vacant/Occupied Spot Detection

- End-to-end, fixed-camera pipeline to detect vacant vs occupied parking spots:

- Draw parking stall ROIs once on a reference frame

- Auto-crop stalls across a video to build a dataset

- Label fast (keyboard or drag-and-drop between folders)

- Train a small CNN (fast, simple, well-commented)

- Run live inference either as a script or in a Streamlit app with smooth playback


# QuickStart:

### 0) Set up environment (Windows PowerShell shown)
`python -m venv .venv`
`. .\.venv\Scripts\Activate.ps1`
`pip install -U pip`
`pip install -r requirements.txt`

### 1) Annotate polygons once (draw ROIs)
`python carve_parking_spots.py --mode annotate --video cropped_video.mp4 --out_rois rois.json --expected_spots 15`

### 2) Build crops for dataset (1–2 fps is fine)
`python carve_parking_spots.py --mode crop --video cropped_video.mp4 --rois rois.json --out_dir crops --fps 2`

### 3a) Label images quickly (o=occupied, v=vacant, u=unknown)
`python labeler.py`

### 3b) Optional - Thumbnail relabeling via drag & drop
`python make_two_folders_from_csv.py --labels labels.csv --hardlink`

### 3c) Move mislabels between review\vacant and review\occupied
`python sync_labels_with_folders.py --backup`

### 4) Build train/val/test split (time-based; single-class stalls go to train only)
`python make_dataset.py`

### 5) Train the tiny CNN (saves best_tinycnn.pt)
`python train_tinycnn.py`

### 6a) Scripted video overlay (quick check)
`python infer_on_video.py --video cropped_video.mp4 --rois rois.json --model best_tinycnn.pt --smooth 7`

### 6b) Streamlit app (smooth playback UI)
`streamlit run app.py`




# Sidebar inputs

- Model state_dict (.pt): path to your trained weights (default best_tinycnn.pt)

- Video file: e.g., cropped_video.mp4

- ROIs JSON: the polygons you saved, e.g., rois.json

- Device: cpu or cuda (CUDA optional)

- Process every Nth frame: speed/CPU knob (default 2)

- Playback speed (×): 0.25× to 3.0×

- Click Start to play; Stop to pause.



# What you’ll see

- Overlaid polygons per stall:

- Green = vacant

- Red = occupied

- If confidence < 0.30, we draw border only (uncertain)

- Banner with Vacant / Occupied / Total counts



# How it works (internals)

- Polygon masking: crops are from your exact polygons (keeps geometry as annotated).
  Why this (X) vs perspective warp (Y)?
  X preserves your true ROI shape and is simple in code.
  Y (warping) can produce more uniform views for the CNN; we use it in the script pipeline to train, but the app prioritizes ease/visual QA.

- EMA smoothing: exponentially smoothed probability with alpha = 0.6 to de-noise frame-to-frame jitter.

### - Hysteresis:

    - enter occupied when smoothed p >= 0.85

    - return to vacant when smoothed p < 0.75. This reduces flicker at the decision boundary.

- Compute saving: process every Nth frame (default 2) to reduce CPU load; frames in between still render with the last known state.


Tuning knobs (edit in app.py)
If you want different behavior, change these constants near the top:

- `DECISION_THR_ENTER = 0.85`  # switch to OCCUPIED at or above this
- `DECISION_THR_EXIT = 0.75`  # switch back to VACANT below this
- `COLOR_CONF_THR = 0.30`  # min confidence to fill polygons with color
- `alpha = 0.6`  # EMA smoothing factor (higher = more smoothing)


# Project Structure:

.

├── app.py                          # Streamlit UI with EMA + hysteresis playback

├── carve_parking_spots.py          # annotate ROIs & crop/rectify frames

├── infer_on_video.py               # script overlay (batch warps + smoothing/hysteresis)

├── labeler.py                      # fast keyboard labeler -> labels.csv

├── make_two_folders_from_csv.py    # build review/occupied|vacant from CSV

├── sync_labels_with_folders.py     # adopt moves back to labels.csv (backup/dry-run + flips)

├── review_diff_preview.py          # preview which files would change in a sync

├── label_stats.py                  # class balance overall + per spot

├── make_dataset.py                 # build data/train|val|test (time-split; spot_XX__ names)

├── train_tinycnn.py                # tiny CNN trainer (pos_weight, Windows-safe)

├── calibrate_thresholds.py         # per-spot thresholds from validation set

├── rois.json                       # saved polygons (from annotation)

├── crops/                          # carved crops per spot (from step 2)

└── data/                           # train|val|test folders (from step 5)

