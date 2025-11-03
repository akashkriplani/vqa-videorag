"""
data_preparation.py
Data Preparation for MedVidQA VideoRAG Pipeline.

1. Load MedVidQA dataset.
2. Download YouTube videos.
3. Clean dataset by removing entries with failed downloads.
"""

import os
import json
import re
from pytubefix import YouTube
from pytubefix.cli import on_progress

CLEANED_DIR = "MedVidQA_cleaned"
DATASET_DIR = "MedVidQA"
VIDEO_TRAIN_DIR = "videos_train"
VIDEO_TEST_DIR = "videos_test"
VIDEO_VAL_DIR = "videos_val"

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

def filter_by_embeddings(dataset, split, model_name=None):
    """Keep only entries for which both textual and visual embeddings exist.

    If `model_name` is provided, prefer directories named
    `feature_extraction_{sanitized_model}` (for example
    `feature_extraction_openai_whisper_small`). Falls back to
    the default `feature_extraction` directory when the model-specific
    directory does not exist.
    """
    # sanitize model name to a filesystem-friendly suffix (replace
    # non-alphanumeric characters with underscore)
    model_suffix = None
    if model_name:
        model_suffix = re.sub(r"[^A-Za-z0-9]+", "_", model_name)

    # prefer model-specific feature extraction dir when present
    candidate_base_dirs = []
    if model_suffix:
        candidate_base_dirs.append(f"feature_extraction_{model_suffix}")
    candidate_base_dirs.append("feature_extraction")

    base_dir = None
    for cand in candidate_base_dirs:
        if os.path.exists(cand):
            base_dir = cand
            break
    # if none exists, keep using the default name (will result in empty sets)
    if base_dir is None:
        base_dir = "feature_extraction"

    textual_dir = os.path.join(base_dir, "textual", split)
    visual_dir = os.path.join(base_dir, "visual", split)
    # Get set of video_ids for which both textual and visual embedding files exist
    textual_ids = set()
    visual_ids = set()
    if os.path.exists(textual_dir):
        for fname in os.listdir(textual_dir):
            if fname.endswith(".json"):
                vid = fname.split(".")[0]
                textual_ids.add(vid)
    if os.path.exists(visual_dir):
        for fname in os.listdir(visual_dir):
            if fname.endswith(".json"):
                vid = fname.split(".")[0]
                visual_ids.add(vid)
    valid_ids = textual_ids & visual_ids
    # Only keep entries whose video_id is in valid_ids
    return [entry for entry in dataset if entry.get("video_id") in valid_ids]

def _sanitize_for_filename(s: str) -> str:
    """Create a filesystem-safe string from a model name for filenames.

    Example: 'openai/whisper-small' -> 'openai_whisper_small'
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def filter_json_by_embeddings(model_name: str):
    """Filter dataset JSON files to keep only entries with both textual and visual embeddings.

    Writes new cleaned files in `CLEANED_DIR` named like
    `{split}_{sanitized_model}_cleaned.json`.
    """
    sanitized = _sanitize_for_filename(model_name)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        cleaned_path = os.path.join(CLEANED_DIR, f"{split}_cleaned.json")
        if not os.path.exists(cleaned_path):
            print(f"Warning: base cleaned file not found: {cleaned_path}. Skipping {split}.")
            continue
        with open(cleaned_path, "r") as f:
            data = json.load(f)
        filtered = filter_by_embeddings(data, split, model_name=model_name)
        out_path = os.path.join(CLEANED_DIR, f"{split}_{sanitized}_cleaned.json")
        with open(out_path, "w") as out_f:
            json.dump(filtered, out_f, indent=2)
        print(f"Wrote filtered file: {out_path} ({len(filtered)} entries)")

def main():
    splits = ["train", "val", "test"]
    video_dirs = {"train": VIDEO_TRAIN_DIR, "val": VIDEO_VAL_DIR, "test": VIDEO_TEST_DIR}
    cleaned_data = {}

    for split in splits:
        json_path = os.path.join(DATASET_DIR, f"{split}.json")
        data = load_medvidqa_dataset(json_path)
        video_map = get_unique_videos(data)
        failed_ids = download_unique_videos(video_map, video_dirs[split])
        cleaned_data[split] = clean_dataset(data, failed_ids)

    os.makedirs(CLEANED_DIR, exist_ok=True)
    for split in splits:
        out_path = os.path.join(CLEANED_DIR, f"{split}_cleaned.json")
        with open(out_path, "w") as f:
            json.dump(cleaned_data[split], f, indent=2)

if __name__ == "__main__":
    main()
