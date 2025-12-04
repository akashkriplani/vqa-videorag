"""
Dense Search Module

FAISS-based dense vector search for text and visual embeddings.
"""
import os
import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import open_clip


class FaissIndex:
    """
    FAISS index wrapper for loading and searching vector embeddings.
    """

    def __init__(self, index_path):
        """
        Initialize FAISS index loader.

        Args:
            index_path: Path to FAISS index file

        Raises:
            FileNotFoundError: If index or metadata file doesn't exist
        """
        self.index_path = index_path
        self.index = None
        self.metadata = []

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(index_path + ".meta.json"):
            raise FileNotFoundError(
                f"Metadata JSON file not found for index: {index_path}. "
                f"Expected: {index_path}.meta.json"
            )

    def load(self):
        """Load FAISS index and associated metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def search(self, query_vec, top_k=5):
        """
        Search FAISS index with normalized query vector.

        Args:
            query_vec: 1D numpy array (should already be normalized)
            top_k: Number of results to return

        Returns:
            List of dicts with score and metadata
        """
        q = query_vec.astype(np.float32).reshape(1, -1)
        D, I = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"score": float(dist), "meta": None})
            else:
                results.append({"score": float(dist), "meta": self.metadata[idx]})
        return results


class EmbeddingModels:
    """
    Embedding models for query encoding.

    Supports:
    - Bio_ClinicalBERT for text queries (matches text embeddings)
    - BiomedCLIP for cross-modal text->visual queries
    """

    def __init__(
        self,
        device=None,
        biomedclip_model_id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        """
        Initialize embedding models.

        Args:
            device: torch device (auto-detected if None)
            biomedclip_model_id: HuggingFace model ID for BiomedCLIP
        """
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
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            biomedclip_model_id,
            pretrained=True
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def embed_text_bio(self, text, max_length=512):
        """
        Generate text embedding using Bio_ClinicalBERT with CLS token pooling.

        This matches the embedding generation strategy in the video processing pipeline.

        Args:
            text: Input text query
            max_length: Maximum sequence length (default 512 to match training)

        Returns:
            Normalized numpy array of shape (embedding_dim,)
        """
        inputs = self.bio_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bio_model(**inputs)
            # Use CLS token pooling (position 0) to match training embeddings
            emb = outputs.last_hidden_state[:, 0, :].squeeze()

        # Normalize embedding to match FAISS index normalization
        vec = emb.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_text_clip(self, text):
        """
        Generate text embedding using BiomedCLIP for cross-modal retrieval.

        Args:
            text: Input text query

        Returns:
            Normalized numpy array of shape (embedding_dim,)
        """
        tokens = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens).squeeze(0)

        # Normalize embedding to match FAISS index normalization
        vec = emb.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
