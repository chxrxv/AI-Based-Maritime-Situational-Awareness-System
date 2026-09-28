"""Fine-tune a COCO-pretrained YOLOv8s on the vessel-type subset.

Run from the project root:
    venv\\Scripts\\python.exe training\\train.py
"""

import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import MODEL_PATH  # noqa: E402

DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"


def main():
    model = YOLO("yolov8s.pt")  # COCO-pretrained weights, downloaded on first use

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=-1,  # auto-sized to fit available VRAM (6GB RTX 4050)
        patience=20,  # stop early if val metrics plateau -- don't burn time past convergence
        project=str(PROJECT_ROOT / "models" / "runs"),
        name="vessel_yolov8s",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(best_weights.read_bytes())
    print(f"Copied best weights to {MODEL_PATH}")


if __name__ == "__main__":
    main()
