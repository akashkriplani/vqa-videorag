"""
query_faiss.py

Load FAISS indices and metadata from `faiss_db/` and run retrieval for a user query.

Features:
- Embed queries with Bio_ClinicalBERT (text) to search textual FAISS indexes (created with Bio_ClinicalBERT embeddings).
- Embed queries with BiomedCLIP text encoder (open_clip) to search visual FAISS indexes (created with BiomedCLIP image embeddings) for cross-modal retrieval.
- CLI: choose index paths, top_k, and mode (text / visual / both).

Usage examples:
python query_faiss.py --query "laparoscopic surgery" --text_index faiss_db/textual_train.index --visual_index faiss_db/visual_train.index --top_k 5 --mode both

"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import open_clip
import faiss


class FaissIndex:
    def __init__(self, index_path):
        self.index_path = index_path
        self.index = None
        self.metadata = []
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(index_path + ".meta.json"):
            raise FileNotFoundError(f"Metadata JSON file not found for index: {index_path}. Expected: {index_path}.meta.json")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def search(self, query_vec, top_k=5):
        # query_vec should be 1D numpy array
        q = query_vec.astype(np.float32)
        # normalize as the pipeline used normalized vectors
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Query vector has zero norm.")
        q = q / norm
        D, I = self.index.search(q.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"score": float(dist), "meta": None})
            else:
                results.append({"score": float(dist), "meta": self.metadata[idx]})
        return results


class EmbeddingModels:
    def __init__(self, device=None, biomedclip_model_id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Load Bio Clinical BERT for textual embeddings
        self.bio_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bio_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        self.bio_model.eval()

        # Load BiomedCLIP (open_clip) for cross-modal text->visual matching
        # open_clip.create_model_and_transforms returns (model, _, preprocess)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(biomedclip_model_id, pretrained=True)
        # move model to device
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def embed_text_bio(self, text, max_length=128):
        inputs = self.bio_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.bio_model(**inputs).last_hidden_state.mean(dim=1).squeeze()  # (D,)
        vec = out.cpu().numpy().astype(np.float32)
        return vec

    def embed_text_clip(self, text):
        # open_clip provides a tokenize helper
        tokens = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens).squeeze(0)
        vec = emb.cpu().numpy().astype(np.float32)
        return vec


def format_timestamp(ts):
    """Format timestamp tuple (start, end) to readable string"""
    if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
        return "unknown time"
    start, end = ts
    return f"{start:.2f}s - {end:.2f}s"

def extract_metadata(result):
    """Extract useful metadata fields from a search result"""
    meta = result.get("meta", {}) or {}  # Handle None case

    # Common fields across text and visual results
    video_id = meta.get("video_id", "unknown")
    timestamp = format_timestamp(meta.get("timestamp"))

    # Text-specific fields
    text = meta.get("text", "")  # Transcript text if available
    entities = meta.get("entities", [])  # Medical entities if NER was enabled
    entity_text = ", ".join(ent[0] for ent in entities) if entities else ""

    return {
        "video_id": video_id,
        "timestamp": timestamp,
        "text": text,
        "entities": entity_text
    }

def pretty_print_results(results, query=None):
    """Format search results with rich metadata context"""
    if not results:
        print("No results found.")
        return []

    out = []
    print("\nTop Results:")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        metadata = extract_metadata(r)
        modal = r.get("modal", "unknown")
        sim = float(r.get("sim", 0.0))
        source = os.path.basename(r.get("source_index", "unknown"))

        entry = {
            "rank": i,
            "similarity": f"{sim:.3f}",
            "modality": modal,
            "video_id": metadata["video_id"],
            "timestamp": metadata["timestamp"],
            "source": source
        }

        # Add text snippet for textual results
        if modal == "text" and metadata["text"]:
            entry["transcript"] = metadata["text"]
        if metadata["entities"]:
            entry["entities"] = metadata["entities"]

        out.append(entry)

        # Print human-friendly format
        print(f"\n{i}. Video: {metadata['video_id']} ({modal}, sim={sim:.3f})")
        print(f"   Time: {metadata['timestamp']}")
        print(f"   Source: {source}")
        if modal == "text" and metadata["text"]:
            print(f"   Transcript: {metadata['text'][:200]}...")
        if metadata["entities"]:
            print(f"   Entities: {metadata['entities']}")
        print("-" * 80)

    # Also save structured output to JSON
    print("\nStructured results saved to: search_results.json")
    with open("search_results.json", "w") as f:
        json.dump({"query": query, "results": out}, f, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser(description="Query FAISS textual/visual indices using user query embeddings")
    parser.add_argument("--query", type=str, required=False, help="Text query to search (if omitted you'll be prompted)")
    parser.add_argument("--text_index", type=str, default=None, help="Path to textual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--visual_index", type=str, default=None, help="Path to visual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--final_k", type=int, default=10, help="FINAL number of combined results to return (default: 10)")
    parser.add_argument("--device", type=str, choices=["cpu","cuda","mps"], default=None, help="Device to run models on; if omitted auto-detects")
    args = parser.parse_args()

    # If query not provided, prompt the user
    if not args.query:
        try:
            args.query = input("Enter search query: ")
        except Exception:
            raise ValueError("No query provided")

    # Auto-discover indexes in faiss_db/ if explicit paths are not provided.
    def discover_indexes(prefix):
        import glob
        paths = sorted(glob.glob(os.path.join("faiss_db", f"{prefix}_*.index")))
        return paths

    text_index_paths = [args.text_index] if args.text_index else discover_indexes("textual")
    visual_index_paths = [args.visual_index] if args.visual_index else discover_indexes("visual")

    if not text_index_paths and not visual_index_paths:
        raise ValueError("No FAISS indexes found in faiss_db/. Place textual_*.index or visual_*.index or pass explicit paths.")

    device = None
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")

    print(f"Loading embedding models on device: {device or 'auto'}")
    models = EmbeddingModels(device=device)

    # Search each index for a larger local pool, normalize distances to similarities, and merge both modalities
    def search_index_pool(index_paths, query_vec, local_k=50):
        pool = []
        for p in index_paths:
            try:
                idx = FaissIndex(p)
                idx.load()
                res = idx.search(query_vec, top_k=local_k)
                # collect distances to compute per-index normalization if needed
                for r in res:
                    pool.append({
                        "raw_score": r.get("score"),
                        "meta": r.get("meta"),
                        "source_index": p
                    })
            except Exception as e:
                print(f"Warning: failed to search index {p}: {e}")
        return pool

    # create pools for both modalities
    print("Embedding query with Bio_ClinicalBERT and BiomedCLIP to build combined pool...")
    q_vec_text = models.embed_text_bio(args.query) if text_index_paths else None
    q_vec_clip = models.embed_text_clip(args.query) if visual_index_paths else None

    text_pool = search_index_pool(text_index_paths, q_vec_text, local_k=50) if q_vec_text is not None else []
    visual_pool = search_index_pool(visual_index_paths, q_vec_clip, local_k=50) if q_vec_clip is not None else []

    # Normalize raw distances to similarity in (0,1] using sim = 1/(1+dist), then merge
    combined = []
    for item in text_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
        combined.append({"sim": sim, "meta": item.get("meta"), "source_index": item.get("source_index"), "modal": "text", "raw_score": dist})
    for item in visual_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
        combined.append({"sim": sim, "meta": item.get("meta"), "source_index": item.get("source_index"), "modal": "visual", "raw_score": dist})

    print(combined)
    # Sort by descending similarity and take final_k results
    final_k = args.final_k or 10
    combined_sorted = sorted(combined, key=lambda x: x.get("sim", 0.0), reverse=True)
    final_results = combined_sorted[:final_k]

    # Pretty print combined results with metadata context
    pretty_print_results(final_results, query=args.query)


if __name__ == "__main__":
    main()
