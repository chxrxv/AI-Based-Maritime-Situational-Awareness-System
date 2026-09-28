"""CLI entry point: run vessel detection on a recorded video and save the
annotated result. For the interactive dashboard (video + weather combined),
use `streamlit run dashboard/app.py` instead.

Usage:
    venv\\Scripts\\python.exe main.py path\\to\\video.mp4 --output out.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONFIDENCE_THRESHOLD, MODEL_PATH  # noqa: E402
from inference.video_pipeline import VesselDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run vessel detection on a video file and save the annotated output."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="output.mp4", help="Path to save annotated video")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No trained model found at {MODEL_PATH}. Run training/train.py first.")

    detector = VesselDetector(str(MODEL_PATH), confidence_threshold=args.confidence)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated_frame, detections = detector.process_frame(frame)
            writer.write(annotated_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames -- {len(detections)} vessels in last frame")
    finally:
        cap.release()
        writer.release()

    print(f"Saved annotated video to {args.output} ({frame_count} frames processed)")


if __name__ == "__main__":
    main()
