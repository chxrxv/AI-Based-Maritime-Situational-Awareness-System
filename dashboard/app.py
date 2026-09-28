"""Streamlit dashboard: video panel + detection list on the left,
weather panel on the right. Run with:
    venv\\Scripts\\streamlit.exe run dashboard\\app.py
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    MODEL_PATH,
    WEATHER_REFRESH_SECONDS,
)
from inference.video_pipeline import TrackHistory, VesselDetector, draw_tracked_vessels  # noqa: E402
from weather.client import fetch_weather  # noqa: E402

st.set_page_config(page_title="Maritime Situational Awareness", layout="wide")

if "weather" not in st.session_state:
    st.session_state.weather = None
    st.session_state.weather_fetched_at = 0.0
    st.session_state.weather_error = None


def get_weather():
    now = time.time()
    stale = now - st.session_state.weather_fetched_at > WEATHER_REFRESH_SECONDS
    if st.session_state.weather is None or stale:
        try:
            st.session_state.weather = fetch_weather(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
            st.session_state.weather_fetched_at = now
            st.session_state.weather_error = None
        except Exception as exc:
            st.session_state.weather_error = str(exc)
    return st.session_state.weather


def render_weather(placeholder, weather):
    with placeholder.container(border=True):
        if weather is None:
            st.warning(f"Weather unavailable: {st.session_state.weather_error}")
            return
        current = weather["current"]
        daily = weather["daily"]
        hourly = weather["hourly"]

        age_seconds = int(time.time() - st.session_state.weather_fetched_at)
        st.caption(f"Current conditions — {current['condition']}  ·  updated {age_seconds}s ago")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{current['temperature_c']}°C",
                   f"feels {current['apparent_temperature_c']}°C")
        c2.metric("Humidity", f"{current['humidity_pct']}%")
        c3.metric("Wind", f"{current['wind_speed_kmh']} km/h", f"{current['wind_direction_deg']}°")
        c4.metric("Visibility", f"{current['visibility_m']/1000:.1f} km")

        st.divider()
        st.caption(f"Today's forecast — {daily['condition']}")
        d1, d2, d3 = st.columns(3)
        d1.metric("High / Low", f"{daily['temp_max_c']}° / {daily['temp_min_c']}°C")
        d2.metric("Precipitation", f"{daily['precipitation_mm']} mm")
        d3.metric("Max wind", f"{daily['wind_speed_max_kmh']} km/h")

        st.divider()
        st.caption("Next 8 hours")
        next_hours = hourly[:8]
        hourly_table = [
            {
                "Time": h["time"].split("T")[1],
                "Temp (°C)": h["temperature_c"],
                "Condition": h["condition"],
                "Visibility (km)": round(h["visibility_m"] / 1000, 1),
            }
            for h in next_hours
        ]
        st.dataframe(hourly_table, hide_index=True, width='stretch')


def render_vessel_counts(placeholder, items):
    """items: Detection list (image mode) or TrackedVessel list (video mode) --
    both expose .class_name, which is all this summary needs."""
    with placeholder.container(border=True):
        st.metric("Total vessels in frame", len(items))
        if items:
            counts = {}
            for item in items:
                counts[item.class_name] = counts.get(item.class_name, 0) + 1
            table = [{"Vessel type": name, "Count": count} for name, count in sorted(
                counts.items(), key=lambda kv: -kv[1]
            )]
            st.dataframe(table, hide_index=True, width='stretch')
        else:
            st.info("No vessels detected in the current frame.")


def render_vessel_details(placeholder, tracked_vessels):
    """Per-vessel table with speed/direction -- video mode only (needs motion
    across frames, which a single image doesn't have)."""
    with placeholder.container():
        if not tracked_vessels:
            return
        st.caption(
            "Speed and direction are pixel-based and frame-relative -- not real-world "
            "knots or true compass heading, which would require camera calibration "
            "(a known distance-to-pixel scale and the camera's compass orientation)."
        )
        rows = [
            {
                "Track ID": tv.detection.track_id,
                "Type": tv.class_name,
                "Confidence": round(tv.detection.confidence, 2),
                "Speed (px/s)": round(tv.speed_px_s) if tv.speed_px_s is not None else "-",
                "Direction": tv.direction_label or "-",
            }
            for tv in tracked_vessels
        ]
        st.dataframe(rows, hide_index=True, width='stretch')


# --- Header ---
st.title("AI-Based Maritime Situational Awareness System")
st.caption("YOLO-based vessel detection & classification, combined with live weather conditions.")

model_ready = MODEL_PATH.exists()
status_col1, status_col2 = st.columns([1, 4])
status_col1.metric("Model status", "Ready" if model_ready else "Not trained yet")
if not model_ready:
    status_col2.warning(
        f"No trained model found at `{MODEL_PATH}`. Run `training/train.py`, then reload this page."
    )

st.divider()

# --- Sidebar controls ---
st.sidebar.header("Input Source")
source_choice = st.sidebar.radio("Choose input", ["Upload video file", "Webcam", "Upload image"])

video_source = None
image_source = None
if source_choice == "Upload video file":
    uploaded_file = st.sidebar.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # unique filename per upload -- a fixed shared filename could still be
        # locked by a previous run's cv2.VideoCapture on Windows, silently
        # failing the write (this was the intermittent "upload not taking" bug)
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        upload_dir = PROJECT_ROOT / "data" / "raw_downloads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(suffix=suffix, dir=str(upload_dir))
        with open(fd, "wb") as f:
            f.write(uploaded_file.read())
        video_source = temp_name
elif source_choice == "Webcam":
    video_source = 0
else:
    uploaded_image = st.sidebar.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
        image_source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

st.sidebar.divider()
confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, CONFIDENCE_THRESHOLD, 0.05)
frame_skip = st.sidebar.slider(
    "Frame skip (video only)", 0, 5, 0,
    help="Process every Nth frame instead of every frame. Higher = faster, less smooth."
)
start_button = st.sidebar.button("Start detection", type="primary", disabled=not model_ready)

# --- Main layout ---
col_video, col_info = st.columns([2, 1])

with col_video:
    header_col, fps_col = st.columns([3, 1])
    header_col.subheader("Live Feed")
    fps_placeholder = fps_col.empty()
    video_placeholder = st.empty()
    if not start_button:
        video_placeholder.info("Choose a source in the sidebar, then click **Start detection**.")
    details_placeholder = st.empty()

with col_info:
    st.subheader("Detected Vessels")
    vessel_placeholder = st.empty()
    render_vessel_counts(vessel_placeholder, [])

    st.subheader("Weather")
    weather_placeholder = st.empty()
    render_weather(weather_placeholder, get_weather())

# --- Detection run ---
if model_ready and start_button:
    if source_choice == "Upload image":
        if image_source is None:
            st.warning("Upload an image file first.")
        else:
            detector = VesselDetector(str(MODEL_PATH), confidence_threshold=confidence)
            # single still image -- no persistent identity across frames, so tracking is off
            annotated_frame, detections = detector.process_frame(image_source, use_tracking=False)
            video_placeholder.image(annotated_frame, channels="BGR")
            render_vessel_counts(vessel_placeholder, detections)
    elif video_source is None:
        st.warning("Upload a video file first.")
    else:
        detector = VesselDetector(str(MODEL_PATH), confidence_threshold=confidence)
        track_history = TrackHistory()
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Could not open video source: {video_source}")
        else:
            frame_index = 0
            last_tick = time.time()
            try:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_index += 1
                    if frame_skip and frame_index % (frame_skip + 1) != 0:
                        continue

                    _, detections = detector.process_frame(frame)
                    now = time.time()
                    # smooths the displayed class per track (fixes label flicker on
                    # momentary low-confidence frames) and derives speed/direction
                    # from the track's recent on-screen motion
                    tracked = track_history.update(detections, now)
                    annotated_frame = draw_tracked_vessels(frame, tracked)

                    video_placeholder.image(annotated_frame, channels="BGR")
                    render_vessel_counts(vessel_placeholder, tracked)
                    render_vessel_details(details_placeholder, tracked)
                    render_weather(weather_placeholder, get_weather())

                    fps_placeholder.metric("FPS", f"{1.0 / max(now - last_tick, 1e-6):.1f}")
                    last_tick = now
            finally:
                cap.release()  # always release, even on error -- an unreleased
                                # capture can lock the file for the next upload
