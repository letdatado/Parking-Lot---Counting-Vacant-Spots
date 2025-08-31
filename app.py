# app.py
# Parking lot FE with smooth playback
# - EMA smoothing + hysteresis for stable ROI states
# - Decision threshold fixed at 0.85 (enter OCCUPIED)
# - Exit threshold fixed at 0.75 (hysteresis)
# - Coloring min confidence fixed at 0.30
# - Process every Nth frame (default=2, user adjustable)
# - Playback speed scaling (×)

import json, time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# =============================
# 1) TinyCNN as trained
# =============================
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
        self.backbone = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# ======================================
# 2) Eval transforms (match training)
# ======================================
IMG_SIZE = 96
EVAL_PREPROC = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ===========================
# 3) Load state_dict
# ===========================
@st.cache_resource
def load_model_state_dict(path: str, device_str: str = "cpu"):
    device = torch.device(device_str)
    sd = torch.load(path, map_location=device)
    model = TinyCNN().to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return model, device

# ====================
# 4) ROI helpers
# ====================
def load_rois(rois_json_path):
    with open(rois_json_path, "r") as f:
        data = json.load(f)
    ref_w = int(data["image_size"]["w"])
    ref_h = int(data["image_size"]["h"])
    polys = [np.array(poly, dtype=np.float32) for poly in data["polygons"]]
    return ref_w, ref_h, polys

def scale_polys_to_frame(polys, ref_w, ref_h, W, H):
    sx, sy = W / float(ref_w), H / float(ref_h)
    return [np.column_stack([P[:,0]*sx, P[:,1]*sy]) for P in polys]

def crop_polygon(frame_bgr, poly):
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    roi = frame_bgr[y:y+h, x:x+w].copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = (poly - np.array([x, y], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [shifted], 255)
    return cv2.bitwise_and(roi, roi, mask=mask)

def draw_polygon_overlay(frame_bgr, poly, color_fill, alpha=0.35, border_color=(0,0,0)):
    overlay = frame_bgr.copy()
    pts = poly.reshape((-1,1,2)).astype(np.int32)
    cv2.fillPoly(overlay, [pts], color_fill)
    out = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    cv2.polylines(out, [pts], True, border_color, 2)
    return out

# ====================
# 5) Inference helper
# ====================
def predict_p_occupied(model, device, slot_bgr):
    rgb = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2RGB)
    x = EVAL_PREPROC(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(True):
                logit = model(x)
        else:
            logit = model(x)
        return float(torch.sigmoid(logit).item())

# ====================
# 6) Streamlit UI
# ====================
st.set_page_config(page_title="Parking Slots — Vacant/Occupied", layout="centered")
st.title("🚗 Parking Slots: Vacant vs Occupied")
st.caption("Smooth playback with EMA + hysteresis. N=2 compute saver by default.")

# Sidebar essentials
st.sidebar.header("Inputs")
model_path = st.sidebar.text_input("Model state_dict (.pt)", "best_tinycnn.pt")
video_path = st.sidebar.text_input("Video file", "cropped_video.mp4")
rois_path  = st.sidebar.text_input("ROIs JSON", "rois.json")
device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
frame_skip_n  = st.sidebar.number_input("Process every Nth frame", min_value=1, value=2, step=1)
playback_speed = st.sidebar.slider("Playback speed (×)", 0.25, 3.0, 1.0, 0.25)

# Fixed thresholds
DECISION_THR_ENTER = 0.85
DECISION_THR_EXIT  = 0.75
COLOR_CONF_THR     = 0.30

# Start/Stop
if "run" not in st.session_state:
    st.session_state.run = False
c1, c2 = st.columns(2)
if c1.button("▶️ Start"): st.session_state.run = True
if c2.button("⏹️ Stop"):  st.session_state.run = False

# Load model + ROIs
with st.status("Loading model...", expanded=False) as status:
    try:
        model, device = load_model_state_dict(model_path, device_choice)
        status.update(label="Model loaded.", state="complete")
    except Exception as e:
        status.update(label="Failed to load model", state="error")
        st.error(str(e))
        st.stop()

try:
    ref_w, ref_h, polys_ref = load_rois(rois_path)
    st.caption(f"ROI base size: {ref_w}×{ref_h} • slots: {len(polys_ref)}")
except Exception as e:
    st.error(f"Failed to load ROIs: {e}")
    st.stop()

video_area  = st.empty()
totals_area = st.empty()

COLOR_VACANT   = (0, 200, 0)
COLOR_OCCUPIED = (0, 0, 200)
COLOR_BORDER   = (0, 0, 0)

# Main loop
if st.session_state.run:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Could not open video: {video_path}")
        st.session_state.run = False
    else:
        ok, first = cap.read()
        if not ok:
            st.error("Could not read from video.")
            st.session_state.run = False
        else:
            H, W = first.shape[:2]
            polys = scale_polys_to_frame(polys_ref, ref_w, ref_h, W, H)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # state
            num_slots = len(polys)
            ema_p = [None]*num_slots
            state_occ = [False]*num_slots
            alpha = 0.6

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
            base_delay = (1.0/src_fps) / max(0.1, playback_speed)
            i = 0
            last_totals = (0, 0, num_slots)

            while st.session_state.run and cap.isOpened():
                ok, frame = cap.read()
                if not ok: break
                i += 1

                do_infer = (i % frame_skip_n == 0)
                if do_infer:
                    for idx, P in enumerate(polys):
                        slot = crop_polygon(frame, P)
                        p = predict_p_occupied(model, device, slot)
                        if ema_p[idx] is None: ema_p[idx] = p
                        else: ema_p[idx] = alpha*p + (1-alpha)*ema_p[idx]
                        # hysteresis
                        if state_occ[idx]:
                            if ema_p[idx] < DECISION_THR_EXIT: state_occ[idx] = False
                        else:
                            if ema_p[idx] >= DECISION_THR_ENTER: state_occ[idx] = True
                    occ = sum(state_occ)
                    vac = num_slots - occ
                    last_totals = (vac, occ, num_slots)

                # draw overlays every frame
                vac, occ, total = last_totals
                for idx, P in enumerate(polys):
                    p_hat = ema_p[idx] if ema_p[idx] is not None else 0.0
                    confident = (p_hat >= COLOR_CONF_THR) or ((1-p_hat) >= COLOR_CONF_THR)
                    if confident:
                        frame = draw_polygon_overlay(frame, P,
                                                     COLOR_OCCUPIED if state_occ[idx] else COLOR_VACANT,
                                                     alpha=0.35, border_color=COLOR_BORDER)
                    else:
                        pts = P.reshape((-1,1,2)).astype(np.int32)
                        cv2.polylines(frame, [pts], True, COLOR_BORDER, 2)

                # totals banner
                text = f"Vacant: {vac}  |  Occupied: {occ}  |  Total: {total}"
                cv2.rectangle(frame, (10,10), (370,45), (50,50,50), -1)
                cv2.putText(frame, text, (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # show
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_area.image(frame_rgb, channels="RGB", width="stretch")
                totals_area.markdown(f"**Vacant:** `{vac}` &nbsp;&nbsp; **Occupied:** `{occ}` &nbsp;&nbsp; **Total:** `{total}`")

                time.sleep(max(0.0, base_delay))

            cap.release()
            st.info("Playback finished.")
else:
    st.info("Press **Start** to begin.")
