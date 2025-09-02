import cv2
import os
import numpy as np
import random

def augment_image(image):
    aug_type = random.choice(['flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270', 'brightness', 'contrast'])
    
    if aug_type == 'flip_h':
        return cv2.flip(image, 1)
    elif aug_type == 'flip_v':
        return cv2.flip(image, 0)
    elif aug_type == 'rotate_90':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif aug_type == 'rotate_180':
        return cv2.rotate(image, cv2.ROTATE_180)
    elif aug_type == 'rotate_270':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_type == 'brightness':
        factor = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif aug_type == 'contrast':
        factor = random.uniform(0.5, 1.5)
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

def process_subfolder(subfolder_path, target_count=5000):
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(image_files)
    
    if current_count >= target_count:
        print(f"Subfolder {subfolder_path} already has {current_count} images, skipping.")
        return
    
    needed = target_count - current_count
    print(f"Subfolder {subfolder_path}: {current_count} images, need {needed} more.")
    
    for i in range(needed):
        orig_file = random.choice(image_files)
        orig_path = os.path.join(subfolder_path, orig_file)
        image = cv2.imread(orig_path)
        
        if image is None:
            continue
        
        augmented = augment_image(image)
        aug_filename = f"aug_{i:06d}_{orig_file}"
        aug_path = os.path.join(subfolder_path, aug_filename)
        cv2.imwrite(aug_path, augmented)

def augment_dataset(dataset_root):
    for subfolder in os.listdir(dataset_root):
        subfolder_path = os.path.join(dataset_root, subfolder)
        if os.path.isdir(subfolder_path):
            process_subfolder(subfolder_path)

dataset_root = 'data-final'
augment_dataset(dataset_root)