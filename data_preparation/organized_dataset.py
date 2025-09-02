# organize_dataset.py

import os
import shutil

DATASET_DIR = "dataset"

print("🔍 Organizing dataset...")

for fname in os.listdir(DATASET_DIR):
    fpath = os.path.join(DATASET_DIR, fname)
    if not os.path.isfile(fpath):
        continue
    
    try:
        parts = fname.split("_")
        if parts[0] == "hood":
            label = "_".join(parts[:2])   # hood_open / hood_closed
        else:
            label = "_".join(parts[:3])   # rear_left_closed, front_right_open, dll

        target_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(target_dir, exist_ok=True)
        
        shutil.move(fpath, os.path.join(target_dir, fname))
        print(f"✅ moved {fname} -> {target_dir}")
    except Exception as e:
        print("skip", fname, e)

print("🎉 Done! Dataset sudah dipisah ke folder per kelas.")
