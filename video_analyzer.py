# backend/video_analyzer.py
import cv2
import time
import os
import csv
from ultralytics import YOLO
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

# CONFIGURATION
MODEL_PATH = "yolov8n.pt"
DETECT_INTERVAL_SECONDS = 2.0

model = YOLO(MODEL_PATH)

# Paths
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SNAP_DIR = LOG_DIR / "snapshots"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = LOG_DIR / "logs.csv"

# CSV Init
if not LOG_CSV.exists():
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "camera", "labels", "counts_json", "snapshot_path"])


def append_csv(timestamp, camera, labels, counts, snapshot):
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            timestamp,
            camera,
            ";".join(labels),
            str(counts),
            snapshot
        ])


# MAIN ANALYZER
def analyze_video(video_path, camera_name="Video", stop_flag=None, progress_callback=None):
    """
    Detect objects but emit alerts only once every DETECT_INTERVAL_SECONDS.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_index = 0

    alerts = []
    last_alert_time = 0.0

    while cap.isOpened():
        if stop_flag and getattr(stop_flag, "stop", False):
            print("🛑 Detection stopped by user.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        now = time.time()

        if now - last_alert_time < DETECT_INTERVAL_SECONDS:
            if progress_callback:
                progress_callback(frame_index / total_frames)
            continue

        # YOLO inference
        results = model(frame)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            if progress_callback:
                progress_callback(frame_index / total_frames)
            continue

        labels = []
        for b in boxes:
            try:
                cls_id = int(b.cls)
                labels.append(model.names.get(cls_id, "unknown"))
            except Exception:
                labels.append("unknown")

        if not labels:
            continue

        # ALERT GENERATED
        last_alert_time = now
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counts = dict(Counter(labels))

        annotated = results[0].plot()

        safe_cam = "".join(c if c.isalnum() or c in "._-" else "_" for c in camera_name)
        snap_name = f"{safe_cam}_{frame_index:06}.jpg"
        snap_path = str(SNAP_DIR / snap_name)

        cv2.imwrite(snap_path, annotated)

        append_csv(timestamp, camera_name, labels, counts, snap_path)

        alert = {
            "timestamp": timestamp,
            "camera": camera_name,
            "labels": sorted(set(labels)),
            "counts": counts,
            "snapshot": snap_path
        }
        alerts.append(alert)

        print(f"[{timestamp}] ALERT (interval={DETECT_INTERVAL_SECONDS}s): {counts} -> {snap_path}")

        if progress_callback:
            progress_callback(frame_index / total_frames)

        time.sleep(0.01)

    cap.release()
    return alerts