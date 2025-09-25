"""
data_preparation.py
Data Preparation for MedVidQA VideoRAG Pipeline.

1. Load MedVidQA dataset.
2. Download YouTube videos.
3. Extract video frames and textual features (placeholders for InternVideo2 and BLIP-2).
"""

import os
import json
from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2

DATASET_DIR = "MedVidQA"
VIDEO_DIR = "videos_train"
CLEANED_DIR = "MedVidQA_cleaned"

def load_medvidqa_dataset(json_path):
    """Load MedVidQA dataset from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def download_video(video_url, save_dir, video_id):
    """Download YouTube video using pytube. Returns True if successful."""
    try:
        yt = YouTube(video_url, use_po_token=True, on_progress_callback=on_progress, use_oauth=True, allow_oauth_cache=True)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            stream.download(output_path=save_dir, filename=f"{video_id}.mp4")
            print(f"Downloaded {video_id} successfully.")
            return True
        else:
            return False
    except Exception as e:
        print(f"Failed to download {video_url}: {e}")
        return False

def get_unique_videos(dataset):
    """Return a dict of unique video_id -> video_url from dataset."""
    video_map = {}
    for entry in dataset:
        vid = entry.get("video_id")
        url = entry.get("video_url")
        if vid and url and vid not in video_map:
            video_map[vid] = url
    return video_map

def download_unique_videos(video_map, save_dir):
    """Download each unique video only once and track failures. Log unique video count and iteration. Skip if already downloaded."""
    failed_ids = []
    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(video_map)} unique video ids. Starting download in {save_dir} ...")
    for idx, (video_id, video_url) in enumerate(video_map.items(), 1):
        video_path = os.path.join(save_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            print(f"[{idx}] {video_id} already downloaded. Skipping.")
            continue
        print(f"[{idx} / {len(video_map)}] Downloading video: {video_id}")
        if not download_video(video_url, save_dir, video_id):
            failed_ids.append(video_id)
    return failed_ids

def clean_dataset(dataset, failed_ids):
    """Remove entries with failed video downloads."""
    return [entry for entry in dataset if entry.get("video_id") not in failed_ids]

def extract_video_frames(video_path, frame_dir, interval=1):
    """Extract frames from video at given interval (seconds)."""
    os.makedirs(frame_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    count = 0
    success, image = vidcap.read()
    while success:
        if int(count % (fps * interval)) == 0:
            frame_path = os.path.join(frame_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()

def extract_textual_features(text):
    """Placeholder for BLIP-2 or domain model feature extraction."""
    # TODO: Replace with actual model inference
    return [0.0] * 768  # Dummy embedding

def extract_visual_features(frame_path):
    """Placeholder for InternVideo2 or domain model feature extraction."""
    # TODO: Replace with actual model inference
    return [0.0] * 1024  # Dummy embedding

if __name__ == "__main__":
    train_json = os.path.join(DATASET_DIR, "train.json")
    val_json = os.path.join(DATASET_DIR, "val.json")
    test_json = os.path.join(DATASET_DIR, "test.json")
    train_data = load_medvidqa_dataset(train_json)
    val_data = load_medvidqa_dataset(val_json)
    test_data = load_medvidqa_dataset(test_json)

    train_video_map = get_unique_videos(train_data)
    val_video_map = get_unique_videos(val_data)
    test_video_map = get_unique_videos(test_data)

    failed_train = download_unique_videos(train_video_map, VIDEO_DIR)
    failed_val = download_unique_videos(val_video_map, "videos_val")
    failed_test = download_unique_videos(test_video_map, "videos_test")

    cleaned_train = clean_dataset(train_data, failed_train)
    cleaned_val = clean_dataset(val_data, failed_val)
    cleaned_test = clean_dataset(test_data, failed_test)

    os.makedirs(CLEANED_DIR, exist_ok=True)
    with open(os.path.join(CLEANED_DIR, "train_cleaned.json"), "w") as f:
        json.dump(cleaned_train, f, indent=2)
    with open(os.path.join(CLEANED_DIR, "val_cleaned.json"), "w") as f:
        json.dump(cleaned_val, f, indent=2)
    with open(os.path.join(CLEANED_DIR, "test_cleaned.json"), "w") as f:
        json.dump(cleaned_test, f, indent=2)

    # Extract frames and features (example for one video)
    # for entry in cleaned_train[:1]:
    #     video_id = entry["video_id"]
    #     video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    #     frame_dir = os.path.join(VIDEO_DIR, f"{video_id}_frames")
    #     extract_video_frames(video_path, frame_dir)
    #     # Extract features (placeholder)
    #     for frame_file in os.listdir(frame_dir):
    #         frame_path = os.path.join(frame_dir, frame_file)
    #         visual_feat = extract_visual_features(frame_path)
    #     text_feat = extract_textual_features(entry.get("transcript", ""))
