import os
from pathlib import Path

DATASET_PATH = "dataset"

removed = 0

for split in ["train", "valid", "test"]:
    label_dir = Path(DATASET_PATH) / split / "labels"
    img_dir = Path(DATASET_PATH) / split / "images"

    for label_file in label_dir.glob("*.txt"):

        if os.path.getsize(label_file) == 0:
            img_file = img_dir / (label_file.stem + ".jpg")

            label_file.unlink()
            if img_file.exists():
                img_file.unlink()

            removed += 1

print(f"✅ Removed {removed} empty samples")