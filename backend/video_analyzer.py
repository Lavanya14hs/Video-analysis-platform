import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, deque

from ultralytics import YOLO
#  MODELS

OBJECT_MODEL_PATH   = "yolov8n.pt"
FIGHT_MODEL_PATH    = "models/fight_detection.pt"
ACCIDENT_MODEL_PATH = "models/accident_detection.pt"

# Load object detection model (download if not present)
try:
    object_model = YOLO(OBJECT_MODEL_PATH)
    print(f"✅ Loaded object detection model: {OBJECT_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Object model not found, downloading YOLOv8n: {e}")
    object_model = YOLO('yolov8n.pt')  # This will download the model
    print("✅ Downloaded and loaded YOLOv8n model")

# Load fight detection model (optional)
try:
    fight_model = YOLO(FIGHT_MODEL_PATH)
    # ── Detect whether the fight model is a CLASSIFIER or a DETECTOR ──
    # Classification models expose `.names` as a flat list/dict of class names
    # and their results have `.probs` instead of `.boxes`.
    # We do a dry-run on a blank frame to figure out which branch to use.
    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    _test  = fight_model(_dummy, verbose=False)[0]
    FIGHT_MODEL_IS_CLASSIFIER = _test.probs is not None
    print(f"✅ Loaded fight detection model: {FIGHT_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Fight model not found, fight detection disabled: {e}")
    fight_model = None
    FIGHT_MODEL_IS_CLASSIFIER = False

# Load accident detection model (optional)
try:
    accident_model = YOLO(ACCIDENT_MODEL_PATH)
    print(f"✅ Loaded accident detection model: {ACCIDENT_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Accident model not found, accident detection disabled: {e}")
    accident_model = None

#  CONFIG
# ── Fight detection ──────────────────────────────────────────────
FIGHT_CONF_MIN        = 0.45   # raw model confidence gate
FIGHT_CONF_STRONG     = 0.65   # "strong" detection — skips motion check
FIGHT_MIN_PERSONS     = 2      # need at least 2 people visible
FIGHT_PROXIMITY_RATIO = 0.55   # person boxes must be within 55% of frame width
                                # (raised from 0.40 — 320px frames have small boxes)
FIGHT_MOTION_MIN      = 1.5    # optical-flow magnitude threshold
                                # (lowered from 8.0 — 320px frames produce ~1–4 naturally)
FIGHT_TEMPORAL_WINDOW = 10     # rolling window (frames) for temporal voting
FIGHT_TEMPORAL_VOTES  = 3      # need N positive frames in window to confirm alert
                                # (lowered from 4 — easier to accumulate with classifier)
FIGHT_COOLDOWN_SEC    = 3      # seconds before a new fight alert can fire

# ── Crowd detection ──────────────────────────────────────────────
CROWD_LOW      = 6    
CROWD_MEDIUM   = 12   
CROWD_HIGH     = 20   
CROWD_CRITICAL = 30   
CROWD_DENSITY_THRESH = 0.30   

# ── General ──────────────────────────────────────────────────────
CONF_THRESHOLD = 0.35   # object / accident confidence gateTARGET_ANALYSIS_FPS = 3  # analyze at most this many frames per second
#  PATHS
ROOT = Path(__file__).resolve().parents[1]

#  HELPERS

def boxes_are_close(box_a, box_b, frame_w, frame_h, ratio=FIGHT_PROXIMITY_RATIO):
    """Return True if two person bounding-boxes are close enough to be fighting."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Centre points
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2

    dist = np.hypot(acx - bcx, acy - bcy)
    threshold = ratio * frame_w          # proximity threshold scales with frame width
    return dist < threshold


def compute_optical_flow_magnitude(prev_gray, curr_gray):
    """Farneback optical flow → mean magnitude in changed regions."""
    if prev_gray is None or curr_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5, levels=2, winsize=10,
        iterations=2, poly_n=5, poly_sigma=1.1,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def persons_overlap_or_close(person_boxes, frame_w, frame_h):
    """Check if at least one pair of person boxes are proximity-close."""
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            if boxes_are_close(person_boxes[i], person_boxes[j], frame_w, frame_h):
                return True
    return False


def run_fight_model(frame, person_count):
    """
    Run the fight model regardless of whether it is a classifier or detector.

    Returns (model_detected: bool, raw_conf: float)

    ── Classifier branch ─
    YOLOv8 classification models return results.probs (a Probs object).
    .probs.top1       → index of the top-1 class
    .probs.top1conf   → confidence of the top-1 class (tensor scalar)

    We treat the model as "fight detected" when the top-1 class is the
    fight/violence class AND confidence ≥ FIGHT_CONF_MIN.

    Fight class index convention used by most public fight datasets:
      0 = fight / violence / abnormal
      1 = no-fight / normal
    If your model uses the opposite convention, flip the check below.

    ── Detector branch ──
    Standard YOLO detection: iterate boxes, pick the highest-conf box.
    """
    if fight_model is None or person_count < FIGHT_MIN_PERSONS:
        return False, 0.0

    results = fight_model(frame, verbose=False)[0]

    # ── CLASSIFIER ──
    if FIGHT_MODEL_IS_CLASSIFIER:
        if results.probs is None:
            return False, 0.0

        top1_idx  = int(results.probs.top1)
        top1_conf = float(results.probs.top1conf)

        # Most fight classifiers: class 0 = fight, class 1 = no-fight.
        # Guard: also accept if the class name contains "fight"/"violence"/"abnormal".
        class_name = (results.names.get(top1_idx, "") or "").lower()
        is_fight_class = (top1_idx == 0) or any(
            kw in class_name for kw in ("fight", "violen", "abnormal", "aggress")
        )

        if is_fight_class and top1_conf >= FIGHT_CONF_MIN:
            return True, top1_conf
        return False, 0.0

    # ── DETECTOR ──
    if results.boxes is None:
        return False, 0.0

    best_conf = 0.0
    detected  = False
    for b in results.boxes:
        conf = float(b.conf[0])
        if conf > FIGHT_CONF_MIN and conf > best_conf:
            best_conf = conf
            detected  = True

    return detected, best_conf


def get_crowd_level(person_count):
    if person_count >= CROWD_CRITICAL:
        return "Critical Crowd", 4
    if person_count >= CROWD_HIGH:
        return "Very Crowded", 3
    if person_count >= CROWD_MEDIUM:
        return "Crowded", 2
    if person_count >= CROWD_LOW:
        return "Busy", 1
    return None, 0


def crowd_density_zones(person_boxes, frame_w, frame_h, grid=4):
    """
    Divide the frame into grid×grid cells, count persons per cell.
    Returns list of dense zone rects (x1,y1,x2,y2) and overall density score.
    """
    cell_w = frame_w / grid
    cell_h = frame_h / grid
    cells  = np.zeros((grid, grid), dtype=int)

    for (px1, py1, px2, py2) in person_boxes:
        cx = int(((px1 + px2) / 2) / cell_w)
        cy = int(((py1 + py2) / 2) / cell_h)
        cx = min(cx, grid - 1)
        cy = min(cy, grid - 1)
        cells[cy, cx] += 1

    max_persons_per_cell = max(1, len(person_boxes))
    dense_zones = []
    for gy in range(grid):
        for gx in range(grid):
            density = cells[gy, gx] / max_persons_per_cell
            if density >= CROWD_DENSITY_THRESH and cells[gy, gx] >= 2:
                dense_zones.append({
                    "x1": int(gx * cell_w),
                    "y1": int(gy * cell_h),
                    "x2": int((gx + 1) * cell_w),
                    "y2": int((gy + 1) * cell_h),
                    "count": int(cells[gy, gx])
                })

    density_score = float(np.max(cells) / max_persons_per_cell)
    return dense_zones, density_score

#  REPORT

def generate_report(alerts):
    fight_count    = sum(1 for a in alerts if a["fight"])
    accident_count = sum(1 for a in alerts if a["accident"])
    crowd_count    = sum(1 for a in alerts if a.get("crowd"))

    summary = f"""
🚨 INCIDENT REPORT

Total Events   : {len(alerts)}
🔥 Fights      : {fight_count}
🚗 Accidents   : {accident_count}
👥 Crowd Alerts: {crowd_count}

"""
    for a in alerts:
        t = a["time_str"]
        if a["fight"]:
            conf_pct = int(a.get("fight_conf", 0) * 100)
            summary += (
                f"\n🔥 {t} — Fight detected "
                f"({a['counts'].get('person', 0)} persons, conf {conf_pct}%)"
            )
        if a["accident"]:
            summary += f"\n🚗 {t} — Accident ({a['accident_type']})"
        if a.get("crowd"):
            summary += (
                f"\n👥 {t} — {a['crowd_level']} "
                f"({a['counts'].get('person', 0)} persons)"
            )

    return summary

#  MAIN

def analyze_video(video_path, progress_callback=None):

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    frame_index      = 0
    processed_frames = 0
    frame_skip       = max(1, int(round(fps / TARGET_ANALYSIS_FPS)))
    second_data      = {}          # best event per second

    # ── Temporal fight voting ──────────────────────────────────────
    fight_vote_window = deque(maxlen=FIGHT_TEMPORAL_WINDOW)
    fight_conf_window = deque(maxlen=FIGHT_TEMPORAL_WINDOW)
    last_fight_second = -999   # cooldown tracker

    # ── Optical flow state ─────────────────────────────────────────
    prev_gray = None

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if progress_callback:
            progress_callback(frame_index / total_frames)

        if frame_index % frame_skip != 0:
            continue

        processed_frames += 1
        frame = cv2.resize(frame, (320, 320))
        frame_h, frame_w = frame.shape[:2]

        current_second = int(frame_index / fps)

        # ── Grayscale for optical flow ──────────────────────────────
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Object detection ────────────────────────────────────────
        obj_results = object_model(frame, verbose=False)[0]

        if obj_results.boxes is None:
            prev_gray = curr_gray
            continue

        raw_boxes   = obj_results.boxes.xyxy.cpu().numpy()
        conf_scores = obj_results.boxes.conf.cpu().numpy()
        class_ids   = obj_results.boxes.cls.cpu().numpy()

        labels          = []
        person_count    = 0
        person_boxes    = []
        vehicle_present = False

        for box, cls, conf in zip(raw_boxes, class_ids, conf_scores):
            if conf < CONF_THRESHOLD:
                continue
            label = object_model.names[int(cls)]
            labels.append(label)

            if label == "person":
                person_count += 1
                person_boxes.append(box.tolist())

            if label in ["car", "motorcycle", "truck", "bus"]:
                vehicle_present = True

        if not labels:
            prev_gray = curr_gray
            continue

        counts = dict(Counter(labels))

        #  FIGHT DETECTION  (multi-layer false-positive suppression)
       
        # --- Layer 1: Model inference (classifier OR detector) --------
        model_detected, raw_fight_conf = run_fight_model(frame, person_count)

        # --- Layer 2: Proximity check – persons must be close ---------
        # NOTE: proximity is only a hard gate when we have ≥2 tracked
        # person boxes. If the object model missed persons but the fight
        # model still fired (e.g. partial occlusion), we allow it through
        # so we don't silently suppress real detections.
        if len(person_boxes) >= 2:
            proximity_ok = persons_overlap_or_close(person_boxes, frame_w, frame_h)
        else:
            # Fewer than 2 person boxes detected — be lenient; the object
            # model may have missed someone. Rely on model confidence instead.
            proximity_ok = raw_fight_conf >= FIGHT_CONF_STRONG

        # --- Layer 3: Optical-flow motion check -----------------------
        flow_mag  = compute_optical_flow_magnitude(prev_gray, curr_gray)
        motion_ok = flow_mag >= FIGHT_MOTION_MIN

        # --- Layer 4: Combine into per-frame vote ---------------------
        # Strong confidence → proximity alone is enough (motion implied)
        # Weak confidence   → needs proximity AND motion
        if model_detected:
            if raw_fight_conf >= FIGHT_CONF_STRONG:
                frame_positive = proximity_ok
            else:
                frame_positive = proximity_ok and motion_ok
        else:
            frame_positive = False

        fight_vote_window.append(1 if frame_positive else 0)
        # Always record the raw confidence for the current frame so the
        # deque stays aligned with the vote window (fixes the stale-window bug).
        fight_conf_window.append(raw_fight_conf if frame_positive else 0.0)

        # --- Layer 5: Temporal voting – need N/M positive frames ------
        votes_in_window = sum(fight_vote_window)
        fight_confirmed = votes_in_window >= FIGHT_TEMPORAL_VOTES

        # --- Layer 6: Cooldown – avoid repeat alerts for same event ---
        in_cooldown = (current_second - last_fight_second) < FIGHT_COOLDOWN_SEC

        fight_alert = fight_confirmed and not in_cooldown

        # Compute mean confidence from the positive frames in the window
        pos_confs = [c for c in fight_conf_window if c > 0.0]
        fight_conf = float(np.mean(pos_confs)) if (fight_alert and pos_confs) else 0.0

        if fight_alert:
            last_fight_second = current_second

        #  ACCIDENT DETECTION
        
        accident_alert = False
        accident_conf  = 0.0
        accident_type  = ""

        if accident_model is not None and vehicle_present:
            results = accident_model(frame, verbose=False)[0]

            if results.boxes is not None:
                for b in results.boxes:
                    conf = float(b.conf[0])
                    if conf > CONF_THRESHOLD and conf > accident_conf:
                        accident_alert = True
                        accident_conf  = conf

                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        involved = []
                        for box, cls in zip(raw_boxes, class_ids):
                            label = object_model.names[int(cls)]
                            bx1, by1, bx2, by2 = map(int, box)
                            if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                                if label in ["car", "motorcycle", "truck", "bus", "person"]:
                                    involved.append(label)
                        involved      = list(set(involved))
                        accident_type = " vs ".join(involved) if len(involved) >= 2 else "Unknown"

        #  CROWD DETECTION
        crowd_alert   = False
        crowd_level   = ""
        crowd_score   = 0
        dense_zones   = []
        density_score = 0.0

        crowd_label, crowd_score = get_crowd_level(person_count)
        if crowd_label:
            crowd_alert   = True
            crowd_level   = crowd_label
            dense_zones, density_score = crowd_density_zones(
                person_boxes, frame_w, frame_h
            )

        #  RECORD EVENT
        if not (fight_alert or accident_alert or crowd_alert):
            prev_gray = curr_gray
            continue

        best_score = max(fight_conf, accident_conf, density_score)

        if current_second not in second_data or second_data[current_second]["score"] < best_score:

            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            video_time = round(frame_index / fps, 2)
            time_str   = f"{int(video_time // 60)}:{int(video_time % 60):02d}"

            snapshot_uri = None
            success, encoded = cv2.imencode(".jpg", frame)
            if success:
                snapshot_uri = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

            second_data[current_second] = {
                "timestamp"    : timestamp,
                "video_time"   : video_time,
                "time_str"     : time_str,
                "counts"       : counts,
                # fight
                "fight"        : fight_alert,
                "fight_conf"   : round(fight_conf, 3),
                # accident
                "accident"     : accident_alert,
                "accident_type": accident_type,
                # crowd
                "crowd"        : crowd_alert,
                "crowd_level"  : crowd_level,
                "crowd_score"  : crowd_score,
                "dense_zones"  : dense_zones,
                "density_score": round(density_score, 3),
                # meta
                "snapshot"     : snapshot_uri,
                "score"        : best_score,
                "flow_mag"     : round(flow_mag, 2),
            }

        prev_gray = curr_gray

    cap.release()

    alerts = list(second_data.values())
    report = generate_report(alerts)

    return {"alerts": alerts, "report": report}