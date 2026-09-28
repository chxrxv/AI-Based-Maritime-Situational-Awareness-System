"""One-time local re-split of the already-fetched images/train pool into
train/val/test (70/15/15). Used because the network fetch of ABOships-PLUS's
own pre-defined val/test splits was too slow to finish in a reasonable time --
see project notes. Deterministic (fixed seed) so it's reproducible.

NOTE: this means val/test are drawn from the same time period/source as
train, not an independently-collected split -- a known limitation, worth
naming explicitly in the internship report rather than presenting the
resulting metrics as if from a fully independent held-out set.
"""

import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def main():
    train_images = sorted((DATA_DIR / "images" / "train").glob("*.png"))
    random.Random(SEED).shuffle(train_images)

    n = len(train_images)
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)

    val_set = train_images[:n_val]
    test_set = train_images[n_val:n_val + n_test]
    # remaining images simply stay in train -- no action needed for those

    for split_name, image_paths in [("val", val_set), ("test", test_set)]:
        image_dir = DATA_DIR / "images" / split_name
        label_dir = DATA_DIR / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            label_path = DATA_DIR / "labels" / "train" / (image_path.stem + ".txt")
            image_path.rename(image_dir / image_path.name)
            label_path.rename(label_dir / label_path.name)

    remaining_train = len(list((DATA_DIR / "images" / "train").glob("*.png")))
    print(f"train: {remaining_train}, val: {len(val_set)}, test: {len(test_set)}")


if __name__ == "__main__":
    main()
