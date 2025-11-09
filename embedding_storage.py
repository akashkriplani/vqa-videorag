"""
embedding_storage.py
Helpers to persist extracted embeddings and metadata.

Provides:
- save_video_features: write per-video JSON files for text and visual features
- save_faiss_indices_from_lists: build and save FAISS indices from in-memory lists
"""
import os
import json
import numpy as np
from typing import List

import faiss


def _to_serializable(results: List[dict]):
    out = []
    for r in results:
        r_copy = r.copy()
        if isinstance(r_copy.get("embedding"), (np.ndarray,)):
            r_copy["embedding"] = r_copy["embedding"].tolist()
        out.append(r_copy)
    return out


def save_video_features(video_id: str, text_results: List[dict], visual_results: List[dict], text_feat_dir: str, visual_feat_dir: str):
    """Write per-video JSON files for the given results.

    Args:
        video_id: base filename without extension
        text_results: list of dicts produced by extract_entities_and_embed
        visual_results: list of dicts produced by extract_frames_and_embed
        text_feat_dir: directory for textual JSONs
        visual_feat_dir: directory for visual JSONs
    """
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
    visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")

    text_serializable = _to_serializable(text_results)
    visual_serializable = _to_serializable(visual_results)

    with open(text_json_path, "w") as f:
        json.dump(text_serializable, f)

    with open(visual_json_path, "w") as f:
        json.dump(visual_serializable, f)


class FaissDBSimple:
    """Minimal helper to build and save a FAISS IndexFlatL2 and metadata file."""
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadata):
        embeddings = np.stack([np.array(e) / np.linalg.norm(e) for e in embeddings])
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def save(self):
        # Ensure parent directory exists before attempting to write index file
        parent_dir = os.path.dirname(self.index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        # Convert all NumPy arrays inside the metadata to lists
        sanitized_metadata = []
        for item in self.metadata:
            sanitized_item = item.copy()
            if 'embedding' in sanitized_item and isinstance(sanitized_item['embedding'], np.ndarray):
                sanitized_item['embedding'] = sanitized_item['embedding'].tolist()
            sanitized_metadata.append(sanitized_item)

        with open(self.index_path + ".meta.json", "w") as f:
            json.dump(sanitized_metadata, f)

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

        return sorted(results, key=lambda x: x["score"], reverse=True)


def save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path):
    """Build and save FAISS indices from provided lists.

    This function is convenient when extraction code collects embeddings in memory
    and then wants to persist them as indices.
    """
    if all_text_embs:
        # all_text_embs is expected to be a list/sequence of array-like embeddings
        first_emb = np.array(all_text_embs[0])
        dim = int(first_emb.shape[0])
        text_db = FaissDBSimple(dim=dim, index_path=faiss_text_path)
        text_db.add(all_text_embs, all_text_meta)
        text_db.save()

    if all_visual_embs:
        first_emb = np.array(all_visual_embs[0])
        dim = int(first_emb.shape[0])
        visual_db = FaissDBSimple(dim=dim, index_path=faiss_visual_path)
        visual_db.add(all_visual_embs, all_visual_meta)
        visual_db.save()

    print("FAISS indices saved.")
