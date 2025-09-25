"""
feature_extraction.py
Feature extraction for MedVidQA VideoRAG pipeline using BLIP-2 (text) and BiomedCLIP (visual).
Stores embeddings in FAISS via vector_db.py.
If a gated model is used, requests HuggingFace token interactively or from env variable.
"""
import os
import json
import torch
from transformers import Blip2Processor, Blip2Model, AutoFeatureExtractor, AutoModel
from huggingface_hub import login as hf_login
import getpass
from vector_db import VectorDB
from PIL import Image
import cv2
from dotenv import load_dotenv
import open_clip

# --- HuggingFace Token Handling ---
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    try:
        HF_TOKEN = getpass.getpass("Enter your HuggingFace token (for gated models): ")
        hf_login(HF_TOKEN)
    except Exception as e:
        print("Warning: Could not login to HuggingFace. If you get a 401 error, set HF_TOKEN env variable or login manually.")
else:
    hf_login(HF_TOKEN)

# Paths
CLEANED_DIR = "MedVidQA_cleaned"
VIDEO_DIRS = {"train": "videos_train", "val": "videos_val", "test": "videos_test"}
DB_DIR = "faiss_db"
TEXT_EMB_DIR = os.path.join(DB_DIR, "text_extraction")
VISUAL_EMB_DIR = os.path.join(DB_DIR, "visual_extraction")
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(TEXT_EMB_DIR, exist_ok=True)
os.makedirs(VISUAL_EMB_DIR, exist_ok=True)

# --- BLIP-2 (best for text, but may be gated) ---
try:
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model.eval()
    print("Loaded BLIP-2 (Salesforce/blip2-opt-2.7b)")
except Exception as e:
    print("BLIP-2 (opt-2.7b) not available or gated. Falling back to open BLIP-2 checkpoint.")
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip2_model.eval()
    print("Loaded BLIP-2 (Salesforce/blip2-flan-t5-xl)")

# --- BiomedCLIP (best open medical vision model, using open_clip) ---
biomedclip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
# If you need the tokenizer for text, you can also get it:
biomedclip_tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def extract_text_embedding(text):
    inputs = blip2_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        outputs = blip2_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_visual_embedding(frame_path):
    image = Image.open(frame_path).convert('RGB')
    image_tensor = preprocess_val(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        image_features = biomedclip_model.encode_image(image_tensor)
    return image_features.squeeze().cpu().numpy()

def process_dataset(split):
    json_path = os.path.join(CLEANED_DIR, f"{split}_cleaned.json")
    video_dir = VIDEO_DIRS[split]
    db_path_text = os.path.join(TEXT_EMB_DIR, f"{split}_text.index")
    db_path_visual = os.path.join(VISUAL_EMB_DIR, f"{split}_visual.index")
    with open(json_path, "r") as f:
        data = json.load(f)
    text_vectors, text_meta = [], []
    visual_vectors, visual_meta = [], []
    for entry in data:
        # Textual embedding
        if 'transcript' in entry and entry['transcript']:
            text_emb = extract_text_embedding(entry['transcript'])
            text_vectors.append(text_emb)
            text_meta.append({"video_id": entry["video_id"], "type": "text"})
        # Visual embedding (sample 1 frame per video for demo)
        video_id = entry.get("video_id")
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        frame_dir = os.path.join(video_dir, f"{video_id}_frames")
        if os.path.exists(frame_dir):
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
            if frame_files:
                frame_path = os.path.join(frame_dir, frame_files[0])
                visual_emb = extract_visual_embedding(frame_path)
                visual_vectors.append(visual_emb)
                visual_meta.append({"video_id": video_id, "frame": frame_files[0], "type": "visual"})
    # Store in FAISS
    if text_vectors:
        dim = text_vectors[0].shape[0]
        db_text = VectorDB(dim, db_path_text)
        db_text.add(text_vectors, text_meta)
        db_text.save()
    if visual_vectors:
        dim = visual_vectors[0].shape[0]
        db_visual = VectorDB(dim, db_path_visual)
        db_visual.add(visual_vectors, visual_meta)
        db_visual.save()

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")
        process_dataset(split)
        print(f"Done with {split} split.")
