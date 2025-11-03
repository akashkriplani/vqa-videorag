"""
multimodal_pipeline.py
Modular pipeline for multimodal medical video processing:
- ASR (United-MedASR)
- Entity recognition & text embeddings (BioBERT/ClinicalBERT, SciSpacy/HF NER)
- Frame extraction & visual embeddings (BiomedCLIP)
- FAISS DB integration
- Demo pipeline
"""
import os
import gc
import subprocess
import tempfile
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import open_clip
import getpass
import faiss
import spacy
from scispacy.umls_linking import UmlsEntityLinker
from PIL import Image
import cv2
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from tqdm import tqdm  # <-- Add tqdm for progress bars
# from google.colab import userdata
from whisper_patch import get_whisper_pipeline_with_timestamps_simple
from embedding_storage import save_video_features, save_faiss_indices_from_lists
from data_preparation import filter_json_by_embeddings

# Top-level function for multiprocessing

# Parallel processing flag
ENABLE_PARALLEL = True  # Set to False to disable parallel processing

# NER toggle flag
ENABLE_NER = False  # Set to True to enable NER/entity extraction


# Flag to use Colab secrets for HF_TOKEN (default: True)
USE_COLAB_SECRETS = True

# Load HuggingFace token from .env, Colab secrets, or prompt
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# if HF_TOKEN is None and USE_COLAB_SECRETS:
#     try:
#         HF_TOKEN = userdata.get('HF_TOKEN')
#     except Exception:
#         HF_TOKEN = None

if HF_TOKEN is None:
    try:
        HF_TOKEN = getpass.getpass("Enter your HuggingFace token (for gated models): ")
        hf_login(HF_TOKEN)
    except Exception as e:
        print("Warning: Could not login to HuggingFace. If you get a 401 error, set HF_TOKEN env variable or login manually.")
else:
    hf_login(HF_TOKEN)

# 1. ASR with United-MedASR

def extract_audio_ffmpeg(video_path, audio_path):
    cmd = [
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
    ]
    subprocess.run(cmd, check=True)

def transcribe_with_asr(video_path, asr_model_id="openai/whisper-tiny"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        extract_audio_ffmpeg(video_path, tmp_audio.name)
        pipe = get_whisper_pipeline_with_timestamps_simple(asr_model_id, device=None)
    print(f"Transcribing audio with ASR: {video_path} using model: {asr_model_id}")
    result = pipe(tmp_audio.name)
    os.remove(tmp_audio.name)
    # result['chunks'] contains word-level timestamps
    chunks = result['chunks']
    return chunks

# 2. Entity recognition & text embeddings

def load_ner_and_embed_models():
    if ENABLE_NER:
        nlp = spacy.load("en_core_sci_lg")
        linker = UmlsEntityLinker(resolve_abbreviations=True)
        nlp.add_pipe(linker)
    else:
        nlp = None
    bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return nlp, bert_tokenizer, bert_model

def extract_entities_and_embed(transcript_chunks, nlp, bert_tokenizer, bert_model):
    results = []
    for chunk in transcript_chunks:
        text = chunk['text']
        ts = chunk['timestamp']
        if ENABLE_NER and nlp is not None:
            doc = nlp(text)
            entities = [(ent.text, ent._.umls_ents) for ent in doc.ents]
        else:
            entities = []
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            emb = bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        results.append({"timestamp": ts, "text": text, "entities": entities, "embedding": emb})
    return results

# 3. Frame extraction & visual embeddings

def extract_frames_and_embed(video_path, entities, window=1.0, model_id='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'):
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []

    print(f"Extracting frames and visual embeddings: {video_path}")
    for ent in tqdm(entities, desc="Frame extraction", unit="frame"):
        ts = ent['timestamp']
        frame_idx = int(float(ts[0]) * fps) # Assuming timestamp is a tuple (start, end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Use a managed temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp_img:
            cv2.imwrite(tmp_img.name, frame)
            image = Image.open(tmp_img.name).convert('RGB')
            image_tensor = preprocess_val(image).unsqueeze(0)

            with torch.no_grad():
                emb = model.encode_image(image_tensor).squeeze().cpu().numpy()

            # Note: The temp file is deleted upon exiting the 'with' block
            results.append({"timestamp": ts, "frame_path": "in-memory", "embedding": emb})

    cap.release()
    return results

def process_video_batch(batch_fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id):
    batch_results = {}
    for fname in batch_fnames:
        if not fname.endswith('.mp4'):
            batch_results[fname] = (None, None, 'Not an mp4 file')
            continue
        video_id = os.path.splitext(fname)[0]
        video_path = os.path.join(video_dir, fname)
        text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
        visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")
        # If JSON feature files already exist, load embeddings + metadata and return them
        try:
            text_embs, text_meta = None, None
            visual_embs, visual_meta = None, None
            if os.path.exists(text_json_path):
                with open(text_json_path, 'r') as f:
                    text_json = json.load(f)
                # Reconstruct embeddings and metadata
                text_embs = [np.array(r['embedding']) if isinstance(r.get('embedding'), list) else np.array(r.get('embedding')) for r in text_json]
                text_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in text_json]
            if os.path.exists(visual_json_path):
                with open(visual_json_path, 'r') as f:
                    visual_json = json.load(f)
                visual_embs = [np.array(r['embedding']) if isinstance(r.get('embedding'), list) else np.array(r.get('embedding')) for r in visual_json]
                visual_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}} for r in visual_json]

            if text_embs is not None or visual_embs is not None:
                # Normalize output shape to match the usual (text, visual, error) tuple structure
                text_tuple = (text_embs, text_meta) if text_embs is not None else (None, None)
                visual_tuple = (visual_embs, visual_meta) if visual_embs is not None else (None, None)
                batch_results[fname] = (text_tuple, visual_tuple, None)
                continue
        except Exception as e:
            # If existing JSONs are corrupt/unreadable, return an error for this file and continue
            batch_results[fname] = (None, None, f"Failed to load existing JSONs: {e}")
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            transcript_chunks = transcribe_with_asr(video_path, asr_model_id)
            nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()
            text_results = extract_entities_and_embed(transcript_chunks, nlp, bert_tokenizer, bert_model)
            visual_results = extract_frames_and_embed(video_path, text_results)
            # Delegate saving to embedding_storage
            try:
                save_video_features(video_id, text_results, visual_results, text_feat_dir, visual_feat_dir)
            except Exception:
                pass
            text_embs = [r['embedding'] for r in text_results]
            text_meta = [{"video_id": video_id, **r} for r in text_results]
            visual_embs = [r['embedding'] for r in visual_results]
            visual_meta = [{"video_id": video_id, **r} for r in visual_results]
            batch_results[fname] = ((text_embs, text_meta), (visual_embs, visual_meta), None)
        except Exception as e:
            batch_results[fname] = (None, None, str(e))
    return batch_results

def parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id="openai/whisper-tiny", batch_size=1, max_workers=2):
    """
    Full pipeline parallel processing with batching: ASR, NER, embedding, JSON saving.
    """
    # Split fnames into batches
    batches = [fnames[i:i+batch_size] for i in range(0, len(fnames), batch_size)]
    results = {}
    total = len(fnames)
    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video_batch, batch, video_dir, text_feat_dir, visual_feat_dir, asr_model_id): tuple(batch) for batch in batches}
        for fut in as_completed(futures):
            batch_result = fut.result()
            for fname, (text, visual, error) in batch_result.items():
                processed += 1
                if error == 'JSONs already exist':
                    print(f"[SKIP] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                elif error:
                    print(f"[ERROR] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                else:
                    print(f"[DONE] {fname} | Progress: {processed}/{total} videos processed.")
                results[fname] = (text, visual)
    return results

# 4. FAISS database integration
class FaissDB:
    def __init__(self, dim, index_path):
        self.index = faiss.IndexFlatL2(dim)
        self.index_path = index_path
        self.metadata = []
    def add(self, embeddings, metadata):
        embeddings = np.stack([e/np.linalg.norm(e) for e in embeddings])
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".meta.json", "w") as f:
            json.dump(self.metadata, f)
    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)
    def query(self, query_emb, top_k=5):
        query_emb = query_emb / np.linalg.norm(query_emb)
        D, I = self.index.search(query_emb[None, :], top_k)
        return [self.metadata[i] for i in I[0]]

# 5. Demo pipeline

def demo_pipeline(video_path, faiss_text_path, faiss_visual_path):
    # ASR
    transcript_chunks = transcribe_with_asr(video_path)
    # NER + text embedding
    nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()
    text_results = extract_entities_and_embed(transcript_chunks, nlp, bert_tokenizer, bert_model)
    # Visual embedding
    visual_results = extract_frames_and_embed(video_path, text_results)
    # Save to FAISS
    text_db = FaissDB(dim=text_results[0]['embedding'].shape[0], index_path=faiss_text_path)
    visual_db = FaissDB(dim=visual_results[0]['embedding'].shape[0], index_path=faiss_visual_path)
    text_db.add([r['embedding'] for r in text_results], text_results)
    visual_db.add([r['embedding'] for r in visual_results], visual_results)
    text_db.save()
    visual_db.save()
    print("Databases saved.")
    # Query demo
    query = "laparoscopic surgery"
    inputs = bert_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_emb = bert_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    print("Text DB results:", text_db.query(query_emb, top_k=3))

def process_video(fname, video_dir, text_feat_dir, visual_feat_dir):
    if not fname.endswith('.mp4'):
        return None, None
    video_id = os.path.splitext(fname)[0]
    video_path = os.path.join(video_dir, fname)
    text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
    visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")
    # Skip if both text and visual json files exist
    if os.path.exists(text_json_path) and os.path.exists(visual_json_path):
        print(f"[SKIP] {fname}: Both text and visual JSONs exist.")
        return None, None
    # Each thread loads its own models
    nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()
    try:
        transcript_chunks = transcribe_with_asr(video_path)
    except Exception as e:
        print(f"ASR failed for {video_path}: {e}")
        return None, None
    text_results = extract_entities_and_embed(transcript_chunks, nlp, bert_tokenizer, bert_model)
    # Convert embeddings to lists for JSON serialization
    text_results_serializable = []
    for r in text_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        text_results_serializable.append(r_copy)
    with open(text_json_path, 'w') as f:
        json.dump(text_results_serializable, f)
    text_embs = [r['embedding'] for r in text_results]
    text_meta = [{"video_id": video_id, **r} for r in text_results]
    visual_results = extract_frames_and_embed(video_path, text_results)
    # Convert embeddings to lists for JSON serialization
    visual_results_serializable = []
    for r in visual_results:
        r_copy = r.copy()
        if isinstance(r_copy.get('embedding'), np.ndarray):
            r_copy['embedding'] = r_copy['embedding'].tolist()
        visual_results_serializable.append(r_copy)
    with open(visual_json_path, 'w') as f:
        json.dump(visual_results_serializable, f)
    visual_embs = [r['embedding'] for r in visual_results]
    visual_meta = [{"video_id": video_id, **r} for r in visual_results]
    return (text_embs, text_meta), (visual_embs, visual_meta)


def process_split(split, video_dir, text_feat_dir, visual_feat_dir, faiss_text_path, faiss_visual_path):
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    all_text_embs, all_text_meta = [], []
    all_visual_embs, all_visual_meta = [], []
    fnames = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    batch_size = 4  # Tune for your hardware
    max_workers = 2 if torch.cuda.is_available() or torch.backends.mps.is_available() else os.cpu_count()
    if ENABLE_PARALLEL:
        print(f"Parallel processing enabled: batch_size={batch_size}, max_workers={max_workers}")
        results = parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, batch_size=batch_size, max_workers=max_workers)
        for fname in fnames:
            text, visual = results.get(fname, (None, None))
            if text and all(item is not None for item in text):
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual and all(item is not None for item in visual):
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    else:
        total = len(fnames)
        done = 0
        for idx, fname in enumerate(fnames, 1):
            text, visual = process_video(fname, video_dir, text_feat_dir, visual_feat_dir)
            if text:
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual:
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            done += 1
            print(f"Progress: {done}/{total} videos processed.")
    # Delegate FAISS index building / saving to embedding_storage
    try:
        save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path)
    except Exception:
        # Fallback: try previous logic if the helper fails
        if all_text_embs:
            print("Saving text embeddings to FAISS (fallback)...")
            text_db = FaissDB(dim=len(all_text_embs[0]), index_path=faiss_text_path)
            text_db.add(all_text_embs, all_text_meta)
            text_db.save()
        if all_visual_embs:
            print("Saving visual embeddings to FAISS (fallback)...")
            visual_db = FaissDB(dim=len(all_visual_embs[0]), index_path=faiss_visual_path)
            visual_db.add(all_visual_embs, all_visual_meta)
            visual_db.save()
    print(f"Done processing split: {split}")

def test_single_video(video_path, text_feat_dir, visual_feat_dir):
    nlp, bert_tokenizer, bert_model = load_ner_and_embed_models()
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    fname = os.path.basename(video_path)
    (text_embs, text_meta), (visual_embs, visual_meta) = process_video(fname, os.path.dirname(video_path), text_feat_dir, visual_feat_dir, nlp, bert_tokenizer, bert_model)
    print(f"Text embeddings: {len(text_embs)}; Visual embeddings: {len(visual_embs)}")
    print(f"Sample text meta: {text_meta[0] if text_meta else None}")
    print(f"Sample visual meta: {visual_meta[0] if visual_meta else None}")

def main():
  # Test mode for a single video
  # test_single_video("videos_train/_6csIJAWj_s.mp4", "feature_extraction/textual/test_single", "feature_extraction/visual/test_single")

  # Uncomment above and set your video path to test single video
  splits = [
    ("train", "videos_train"),
    ("val", "videos_val"),
    ("test", "videos_test")
  ]
  for split, video_dir in splits:
      print(f"Processing split: {split}")
      process_split(
          split=split,
          video_dir=video_dir,
          text_feat_dir=f"feature_extraction/textual/{split}",
          visual_feat_dir=f"feature_extraction/visual/{split}",
          faiss_text_path=f"faiss_db/textual_{split}.index",
          faiss_visual_path=f"faiss_db/visual_{split}.index"
      )

  # After all splits are processed, filter JSON files based on available embeddings
  print("\nFiltering JSON files to keep only entries with both textual and visual embeddings...")
  filter_json_by_embeddings()

if __name__ == "__main__":
    main()