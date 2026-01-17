import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "datasets")

CLASSES = ["Benign", "Malignant", "Normal"]
  # MUST match folder names exactly

TRAIN_DIR = "datasets/train"
VAL_DIR = "datasets/val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)

    if not os.path.exists(cls_path):
        raise FileNotFoundError(f"Missing folder: {cls_path}")

    images = [
        img for img in os.listdir(cls_path)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    train_imgs, val_imgs = train_test_split(
        images, test_size=0.2, random_state=42
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(VAL_DIR, cls, img)
        )

print("✅ Dataset successfully split into train and validation sets.")
