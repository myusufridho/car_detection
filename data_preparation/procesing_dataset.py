# processing_dataset.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)

def is_valid(path):
    img = cv2.imread(path)
    if img is None:
        return False
    if img.mean() < 10:  # terlalu gelap
        return False
    return True

def preprocess(path, size=IMG_SIZE):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

print("🔍 Loading dataset...")

images, labels = [], []
for fname in os.listdir(DATASET_DIR):
    fpath = os.path.join(DATASET_DIR, fname)
    if not os.path.isfile(fpath):
        continue
    if not is_valid(fpath):
        continue
    try:
        img = preprocess(fpath)
        label = "_".join(fname.split("_")[:2])  # contoh: front_left_open
        images.append(img)
        labels.append(label)
    except Exception as e:
        print("skip", fname, e)

images = np.array(images, dtype="float32")
labels = np.array(labels)

print(f"✅ Loaded {len(images)} valid images")

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_encoded, test_size=0.3, stratify=labels_encoded, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42
)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Saved processed dataset: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")
print("📊 Classes:", encoder.classes_)
