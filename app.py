import sys
import os
import cv2
import time
import json
import csv
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# Backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend import video_analyzer as analyzer
from backend import notify_utils as notify

# Page setup
st.set_page_config(page_title="Video Analysis Dashboard", layout="wide")

st.markdown("""
<h1 style="margin-bottom:0;">🚨 Video Analysis Platform</h1>
<p style="color:#6b7280; font-size:16px;">
Upload video → Analyse → View CSV → Get summary & email
</p>
<hr>
""", unsafe_allow_html=True)

# Paths
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
SNAP_DIR = LOG_DIR / "snapshots"
LOG_CSV = LOG_DIR / "logs.csv"

LOG_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)

# Sidebar - Email
st.sidebar.header("📧 Email Notification")
enable_email = st.sidebar.checkbox("Enable email", value=False)
smtp_user = st.sidebar.text_input("SMTP user")
smtp_pass = st.sidebar.text_input("SMTP app password", type="password")
email_to = st.sidebar.text_input("Send to")

# Upload
st.markdown("## 📹 Upload Videos")

uploaded_videos = st.file_uploader(
    "Upload mp4 / avi / mov files",
    type=["mp4", "avi", "mov"],
    accept_multiple_files=True
)

start = st.button("▶️ Start Analysis")

def format_counts(d: dict) -> str:
    return ", ".join(f"{v} {k}" for k, v in d.items())

def write_csv(video_name, alerts):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{Path(video_name).stem}_report_{ts}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "camera", "counts_json", "snapshot_path"])
        for a in alerts:
            w.writerow([
                a["timestamp"],
                a["camera"],
                json.dumps(a["counts"]),
                a["snapshot"]
            ])
    return path

def make_file_link(path: str) -> str:
    """Convert local path to clickable file:// link"""
    if not path:
        return ""
    return f"file:///{path.replace(os.sep, '/')}"

# Estimated time BEFORE analysis
video_paths = {}

if uploaded_videos:
    st.markdown("## ⏱ Estimated Time")
    for v in uploaded_videos:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(v.read())
            video_paths[v.name] = tmp.name

        size_mb = os.path.getsize(video_paths[v.name]) / (1024 * 1024)

        cap = cv2.VideoCapture(video_paths[v.name])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration = frames / fps
        eta = int(duration * 0.6)

        st.write(
            f"📹 **{v.name}**  \n"
            f"📦 Size: {size_mb:.2f} MB  \n"
            f"⏱ Estimated analysis time: ~{eta} seconds"
        )

# Analysis
if start and uploaded_videos:
    for v in uploaded_videos:
        st.markdown("---")
        st.subheader(f"📹 {v.name}")

        progress = st.progress(0)
        status = st.empty()
        elapsed_ui = st.empty()

        status.info("🔄 Analysing…")

        start_time = time.time()
        timer = {"last": start_time}

        def progress_cb(p):
            progress.progress(int(p * 100))
            now = time.time()
            if now - timer["last"] >= 1:
                elapsed_ui.write(f"⏱ Elapsed time: {int(now - start_time)} seconds")
                timer["last"] = now

        alerts = analyzer.analyze_video(
            video_paths[v.name],
            camera_name=v.name,
            progress_callback=progress_cb
        )

        total_time = int(time.time() - start_time)

        progress.progress(100)
        status.success("✅ Analysis completed")
        elapsed_ui.write(f"⏱ Completed in {total_time} seconds")

        # VIDEO SUMMARY
        summary_counts = {}
        for a in alerts:
            for k, v2 in a["counts"].items():
                summary_counts[k] = max(summary_counts.get(k, 0), v2)

        st.markdown("## 📊 Video Summary")
        st.markdown(f"**⏱ Analysis Time:** {total_time} seconds")
        st.markdown(f"**🧠 Objects Detected:** {format_counts(summary_counts)}")

        # EMAIL
        if enable_email and smtp_user and smtp_pass and email_to and alerts:
            subject = f"🚨 Video Summary — {v.name}"
            body = (
                f"Video: {v.name}\n"
                f"Analysis time: {total_time} seconds\n\n"
                f"Objects detected:\n{format_counts(summary_counts)}"
            )

            ok, msg = notify.send_email_smtp(
                smtp_host="smtp.gmail.com",
                smtp_port=587,
                username=smtp_user,
                password=smtp_pass,
                to_email=email_to,
                subject=subject,
                body=body,
                image_path=alerts[-1]["snapshot"]
            )

            if ok:
                st.success(f"📧 Email sent successfully to **{email_to}**")
            else:
                st.error(f"❌ Email failed: {msg}")

        # CSV VIEW
        csv_path = write_csv(v.name, alerts)
        df_csv = pd.read_csv(csv_path)

        df_csv["snapshot_link"] = df_csv["snapshot_path"].apply(make_file_link)

        st.markdown("### 📄 Detection CSV ")
        st.dataframe(
            df_csv[["timestamp", "camera", "counts_json", "snapshot_link"]],
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download CSV",
            data=df_csv.to_csv(index=False),
            file_name=csv_path.name,
            mime="text/csv"
        )