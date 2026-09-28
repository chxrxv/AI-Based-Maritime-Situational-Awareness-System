"""Run a trained YOLO model over a video (file or webcam), frame by frame.

A video has no special handling in YOLO -- each frame is treated as an
independent still image (see project notes). Tracking (ByteTrack, built into
Ultralytics) is layered on top purely to give each vessel a persistent ID
across frames; detection itself stays per-frame.
"""

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import cv2
import torch
from ultralytics import YOLO

_DIRECTION_LABELS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]


@dataclass
class Detection:
    class_name: str
    confidence: float
    box_xyxy: tuple  # (x1, y1, x2, y2) in pixel coordinates
    track_id: int | None = None


@dataclass
class TrackedVessel:
    """A Detection enriched with across-frame motion info for one tracked vessel.

    speed_px_s / direction_label are pixel-space, frame-relative measurements --
    NOT real-world speed (knots) or true compass heading. Converting to real-world
    units requires camera calibration (a known distance-to-pixel mapping and the
    camera's compass orientation), which this system does not have. Do not present
    these as real-world values without that calibration.
    """

    detection: Detection
    class_name: str  # majority-vote smoothed over recent frames, more stable than a single frame's guess
    speed_px_s: float | None
    direction_label: str | None
    coasting: bool = False  # True = re-shown from memory, not a fresh detection this frame


class VesselDetector:
    def __init__(self, model_path, confidence_threshold=0.4, iou_threshold=0.45, device=None):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        # fp16 (quantize=16) roughly halves GPU inference time; not supported on CPU
        self.quantize = 16 if self.device != "cpu" else None

    def process_frame(self, frame, use_tracking=True):
        """Run detection (+ optional tracking) on a single BGR frame.

        Returns (annotated_frame, list[Detection]).
        """
        if use_tracking:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                quantize=self.quantize,
                persist=True,
                verbose=False,
                # ByteTrack (not the default BoT-SORT) -- BoT-SORT's camera-motion
                # compensation step matches visual features between frames to
                # estimate camera movement, which fails ("not enough matching
                # points") on chaotic wave/spray backgrounds with no stable
                # texture, silently dropping track IDs. ByteTrack skips that step.
                tracker="bytetrack.yaml",
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                quantize=self.quantize,
                verbose=False,
            )

        result = results[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(Detection(class_name, confidence, (x1, y1, x2, y2), track_id))

        annotated_frame = result.plot()
        return annotated_frame, detections


class TrackHistory:
    """Turns per-frame tracked Detections into smoothed class labels + motion info.

    Keyed by the tracker's persistent track_id (from VesselDetector.process_frame's
    use_tracking=True path) -- so this only works in video mode, not single images.
    """

    def __init__(self, position_window=10, class_window=15, coast_seconds=2.0, min_live_frames=3):
        self._positions = defaultdict(lambda: deque(maxlen=position_window))  # (timestamp, cx, cy)
        self._classes = defaultdict(lambda: deque(maxlen=class_window))
        self._last_enriched = {}  # track_id -> (timestamp, TrackedVessel) for coasting through misses
        self._live_counts = defaultdict(int)
        self.coast_seconds = coast_seconds
        # a track needs a few live hits before we trust it enough to keep it on
        # screen during a coast -- otherwise a one-off spurious detection (e.g.
        # the model briefly fragmenting one ship into a separate "mast" box)
        # gets kept alive for the full coast window just like a real vessel
        self.min_live_frames = min_live_frames

    def update(self, detections, timestamp):
        enriched = []
        seen_ids = set()
        for det in detections:
            if det.track_id is None:
                enriched.append(TrackedVessel(det, det.class_name, None, None))
                continue

            seen_ids.add(det.track_id)
            self._live_counts[det.track_id] += 1
            self._classes[det.track_id].append(det.class_name)
            smoothed_class = Counter(self._classes[det.track_id]).most_common(1)[0][0]

            x1, y1, x2, y2 = det.box_xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            positions = self._positions[det.track_id]
            speed_px_s = None
            direction_label = None
            if positions:
                t0, x0, y0 = positions[0]
                dt = timestamp - t0
                if dt > 0.05:  # guard against near-zero elapsed time inflating speed
                    dx, dy = cx - x0, cy - y0
                    speed_px_s = math.hypot(dx, dy) / dt
                    direction_label = self._direction_label(dx, dy)
            positions.append((timestamp, cx, cy))

            tv = TrackedVessel(det, smoothed_class, speed_px_s, direction_label)
            if self._live_counts[det.track_id] >= self.min_live_frames:
                self._last_enriched[det.track_id] = (timestamp, tv)
            enriched.append(tv)

        # coast vessels the detector missed this frame but saw recently -- avoids
        # the box vanishing on every momentary confidence dip below threshold
        for track_id, (last_seen, last_tv) in list(self._last_enriched.items()):
            if track_id in seen_ids:
                continue
            if timestamp - last_seen > self.coast_seconds:
                del self._last_enriched[track_id]
                continue
            enriched.append(TrackedVessel(
                last_tv.detection, last_tv.class_name, last_tv.speed_px_s,
                last_tv.direction_label, coasting=True,
            ))
        return self._deduplicate(enriched)

    @staticmethod
    def _direction_label(dx, dy):
        if math.hypot(dx, dy) < 2:  # negligible on-screen movement
            return "stationary"
        # image y grows downward; flip dy so "up" reads as intuitively up
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        return _DIRECTION_LABELS[round(angle / 45) % 8]

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _deduplicate(self, enriched):
        """Two different track IDs sometimes end up on the same physical vessel
        (e.g. a brief ghost track from a missed association) -- collapse heavily
        overlapping boxes to one, preferring a live detection over a coasted one
        and higher confidence as the tiebreaker."""
        ordered = sorted(enriched, key=lambda tv: (tv.coasting, -tv.detection.confidence))
        kept = []
        for tv in ordered:
            if any(self._iou(tv.detection.box_xyxy, k.detection.box_xyxy) > 0.4 for k in kept):
                continue
            kept.append(tv)
        return kept


_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_text_with_background(img, text, origin_x, baseline_y, scale, thickness, text_color, bg_color):
    """Solid background box behind the text -- plain colored text on a busy
    water/sky/glare background is often unreadable, especially at a distance."""
    (text_w, text_h), gap = cv2.getTextSize(text, _LABEL_FONT, scale, thickness)
    top_left = (origin_x, baseline_y - text_h - gap)
    bottom_right = (origin_x + text_w + 6, baseline_y + gap)
    cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
    cv2.putText(img, text, (origin_x + 3, baseline_y), _LABEL_FONT, scale, text_color,
                thickness, cv2.LINE_AA)


def draw_tracked_vessels(frame, tracked_vessels):
    """Draw boxes + smoothed class/speed/direction, using TrackHistory's output
    instead of the model's raw per-frame result.plot() -- lets the display show
    the stabilized class rather than a single frame's possibly-flickering guess.

    Text/box sizes scale with the frame's actual resolution (baseline: 1280px
    wide) -- a fixed pixel size looks fine on a small frame but shrinks to
    near-illegible once Streamlit downscales a 1920x1080+ video to fit the
    display column, since the drawing happens before that display resize.
    """
    annotated = frame.copy()
    box_color = (0, 255, 0)
    motion_color = (0, 170, 255)

    scale_factor = max(frame.shape[1] / 1280, 0.6)
    label_scale = 1.1 * scale_factor
    motion_scale = 0.9 * scale_factor
    text_thickness = max(2, round(2 * scale_factor))
    box_thickness = max(2, round(3 * scale_factor))

    for tv in tracked_vessels:
        x1, y1, x2, y2 = [int(v) for v in tv.detection.box_xyxy]
        this_box_color = (0, 200, 200) if tv.coasting else box_color  # amber-ish = not a fresh hit this frame
        cv2.rectangle(annotated, (x1, y1), (x2, y2), this_box_color, box_thickness, cv2.LINE_AA)

        label = f"{tv.class_name} {tv.detection.confidence:.2f}"
        if tv.detection.track_id is not None:
            label += f" ID:{tv.detection.track_id}"
        if tv.coasting:
            label += " (tracking)"
        _draw_text_with_background(
            annotated, label, x1, max(y1 - 14, 36),
            scale=label_scale, thickness=text_thickness, text_color=(0, 0, 0), bg_color=this_box_color,
        )

        if tv.speed_px_s is not None:
            motion_label = f"{tv.speed_px_s:.0f} px/s, {tv.direction_label}"
            _draw_text_with_background(
                annotated, motion_label, x1, y2 + 38,
                scale=motion_scale, thickness=text_thickness, text_color=(0, 0, 0), bg_color=motion_color,
            )
    return annotated


def run_on_video(model_path, source, confidence_threshold=0.4, frame_skip=0, display=True):
    """source: a video file path, or an int (0, 1, ...) for a webcam."""
    detector = VesselDetector(model_path, confidence_threshold=confidence_threshold)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_skip and frame_index % (frame_skip + 1) != 0:
                frame_index += 1
                continue

            annotated_frame, detections = detector.process_frame(frame)

            if display:
                cv2.imshow("Vessel Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield annotated_frame, detections
            frame_index += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
