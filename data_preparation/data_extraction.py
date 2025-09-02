import cv2
import os

def extract_frames_from_video(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting frames from {video_path}: FPS={fps}, Total Frames={total_frames}")
    success, image = vidcap.read()
    count = 0
    while success:
        frame_filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"Extracted {count} frames to {output_folder}")

def process_dataset_folder(dataset_root):
    for subfolder in os.listdir(dataset_root):
        subfolder_path = os.path.join(dataset_root, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.lower().endswith('.mp4'):
                    video_path = os.path.join(subfolder_path, file)
                    output_folder = os.path.join(subfolder_path, 'frames')
                    extract_frames_from_video(video_path, output_folder)

dataset_root = 'dataset'
process_dataset_folder(dataset_root)