"""Central config. Plain constants -- no need for a config framework at this scale."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Vessel detection ---
MODEL_PATH = PROJECT_ROOT / "models" / "vessel_yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
CLASS_NAMES = [
    "militaryship",
    "passengership",
    "cruiseship",
    "ferry",
    "sailboat",
    "motorboat",
]

# --- Weather (Open-Meteo, no API key needed) ---
# Placeholder location -- replace with the camera's actual installation site.
DEFAULT_LATITUDE = 18.9220
DEFAULT_LONGITUDE = 72.8347
WEATHER_REFRESH_SECONDS = 600  # 10 minutes; weather doesn't need per-frame calls
