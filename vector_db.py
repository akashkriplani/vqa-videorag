"""
vector_db.py
Handles storage and retrieval of embeddings using FAISS for the MedVidQA VideoRAG pipeline.
"""

import faiss
import numpy as np
import os
import pickle

class VectorDB:
    def __init__(self, dim, db_path):
        self.dim = dim
        self.db_path = db_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # To store metadata for each vector
        if os.path.exists(db_path):
            self.load()

    def add(self, vectors, metadata_list):
        vectors = np.array(vectors).astype('float32')
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector, top_k=5):
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        D, I = self.index.search(query_vector, top_k)
        results = [(self.metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]
        return results

    def save(self):
        faiss.write_index(self.index, self.db_path)
        with open(self.db_path + '.meta', 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.db_path)
        with open(self.db_path + '.meta', 'rb') as f:
            self.metadata = pickle.load(f)
