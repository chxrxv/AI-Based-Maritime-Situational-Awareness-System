"""Build a YOLO-format subset of ABOships-PLUS for the 6 target vessel classes.

Fetches only the specific images needed (via HTTP range requests into the
remote zip) instead of the full 8GB archive -- keeps this tractable on a
slow/unstable connection. Per-split image counts are capped per class so the
subset stays small but every class is represented.

NOTE: the dataset's images[].width/height fields in the COCO json are wrong
(some report e.g. 8x20 px, far smaller than real camera frames). Real
dimensions are read from the downloaded image file instead -- do not trust
the json's width/height for bbox normalization.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

ZIP_URL = "https://zenodo.org/records/8383205/files/ABOships-PLUS.zip?download=1"
ZIP_ROOT = "ABOships-PLUS"

TARGET_CLASSES = {
    "militaryship": 0,
    "passengership": 1,
    "cruiseship": 2,
    "ferry": 3,
    "sailboat": 4,
    "motorboat": 5,
}

PER_CLASS_IMAGE_CAP = {"train": 150, "val": 40, "test": 40}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def select_images(data, cap):
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        name = cat_id_to_name[ann["category_id"]]
        if name in TARGET_CLASSES:
            anns_by_image[ann["image_id"]].append(ann)

    per_class_count = {c: 0 for c in TARGET_CLASSES}
    selected = []
    for image_id in sorted(anns_by_image):
        anns = anns_by_image[image_id]
        classes_here = {cat_id_to_name[a["category_id"]] for a in anns}
        if any(per_class_count[c] < cap for c in classes_here):
            selected.append(image_id)
            for c in classes_here:
                per_class_count[c] += 1

    return selected, anns_by_image, cat_id_to_name, per_class_count


def convert_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    # clamp: a few annotations run slightly past the image edge
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w_norm = min(max(w_norm, 0.0), 1.0)
    h_norm = min(max(h_norm, 0.0), 1.0)
    return x_center, y_center, w_norm, h_norm


def build_split(z, split):
    with z.open(f"{ZIP_ROOT}/annotations/{split}.json") as f:
        data = json.load(f)

    id_to_filename = {im["id"]: im["file_name"] for im in data["images"]}
    selected_ids, anns_by_image, cat_id_to_name, per_class_count = select_images(
        data, PER_CLASS_IMAGE_CAP[split]
    )

    images_out = DATA_DIR / "images" / split
    labels_out = DATA_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failed = []
    for image_id in selected_ids:
        filename = id_to_filename[image_id]
        dest_image = images_out / filename
        label_path = labels_out / (Path(filename).stem + ".txt")

        if dest_image.exists() and label_path.exists():
            skipped += 1
            continue

        zip_path = f"{ZIP_ROOT}/images/{split}/{filename}"

        for attempt in range(1, 4):
            try:
                with z.open(zip_path) as src, open(dest_image, "wb") as dst:
                    dst.write(src.read())
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"  FAILED after 3 attempts: {filename} ({exc})")
                    failed.append(filename)
                    dest_image.unlink(missing_ok=True)
                else:
                    print(f"  retry {attempt}/3 for {filename} after error: {exc}")
                    time.sleep(3)
        else:
            continue  # exhausted retries -- image already logged and cleaned up above

        with Image.open(dest_image) as img:
            img_w, img_h = img.size

        lines = []
        for ann in anns_by_image[image_id]:
            name = cat_id_to_name[ann["category_id"]]
            class_id = TARGET_CLASSES[name]
            x, y, w, h = convert_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        written += 1

    print(f"[{split}] images written: {written}, skipped (already had): {skipped}, failed: {len(failed)}")
    print(f"[{split}] per-class image counts: {per_class_count}")
    return written


def write_data_yaml():
    names = {v: k for k, v in TARGET_CLASSES.items()}
    lines = [
        f"path: {DATA_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    (DATA_DIR / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'data.yaml'}")


def main():
    with RemoteZip(ZIP_URL) as z:
        for split in ["train", "val", "test"]:
            build_split(z, split)
    write_data_yaml()


if __name__ == "__main__":
    main()
