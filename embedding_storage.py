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
    """Convert numpy arrays to lists recursively in a list of dictionaries."""
    def convert_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: convert_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_item(x) for x in item]
        elif isinstance(item, tuple):
            return tuple(convert_item(x) for x in item)
        else:
            return item

    out = []
    for r in results:
        if r is None:
            out.append(None)
        else:
            out.append(convert_item(r))
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


class FaissDB:
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

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def save(self):
        # Ensure parent directory exists before attempting to write index file
        parent_dir = os.path.dirname(self.index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        # Use _to_serializable to handle all numpy arrays recursively
        sanitized_metadata = _to_serializable(self.metadata)

        with open(self.index_path + ".meta.json", "w") as f:
            json.dump(sanitized_metadata, f)

    def search(self, query_vec, top_k=5, save_results=True, results_file="search_results.json"):
        """Search for top-k similar embeddings and return serializable results.

        Args:
            query_vec: 1D numpy array representing the query vector
            top_k: number of top results to return
            save_results: whether to save results to a JSON file
            results_file: name of the file to save results (default: "search_results.json")

        Returns:
            List of dictionaries with 'score' and 'meta' keys, sorted by score (descending)
        """
        # query_vec should be 1D numpy array
        q = query_vec.astype(np.float32)
        # normalize as the pipeline used normalized vectors
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Query vector has zero norm.")
        q = q / norm

        # Perform FAISS search
        D, I = self.index.search(q.reshape(1, -1), top_k)

        # Build results list with metadata
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"score": float(dist), "meta": None})
            else:
                result_meta = self.metadata[idx].copy() if self.metadata[idx] else None
                results.append({"score": float(dist), "meta": result_meta})

        # Sort by score (higher scores first for similarity)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Use _to_serializable to ensure all numpy arrays are converted to lists
        serializable_results = _to_serializable(sorted_results)

        # Save results to JSON file if requested
        if save_results:
            # Determine the directory to save results (same as index directory)
            index_dir = os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else "."
            results_path = os.path.join(index_dir, results_file)

            # Create search results with metadata
            search_output = {
                "query_info": {
                    "top_k": top_k,
                    "num_results": len(serializable_results),
                    "index_path": self.index_path
                },
                "results": serializable_results
            }

            with open(results_path, "w") as f:
                json.dump(search_output, f, indent=2)

            print(f"Search results saved to: {results_path}")

        return serializable_results


def save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path):
    """Build and save FAISS indices from provided lists.

    This function is convenient when extraction code collects embeddings in memory
    and then wants to persist them as indices.
    """
    if all_text_embs:
        # all_text_embs is expected to be a list/sequence of array-like embeddings
        first_emb = np.array(all_text_embs[0])
        dim = int(first_emb.shape[0])
        text_db = FaissDB(dim=dim, index_path=faiss_text_path)
        text_db.add(all_text_embs, all_text_meta)
        text_db.save()

    if all_visual_embs:
        first_emb = np.array(all_visual_embs[0])
        dim = int(first_emb.shape[0])
        visual_db = FaissDB(dim=dim, index_path=faiss_visual_path)
        visual_db.add(all_visual_embs, all_visual_meta)
        visual_db.save()

    print("FAISS indices saved.")
