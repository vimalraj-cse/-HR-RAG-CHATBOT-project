# backend/app/retriever.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

class Retriever:
    def __init__(self,
                 index_path="data/faiss.index",
                 emb_path="data/embeddings.npy",
                 chunks_path="data/chunks.json",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = Path(index_path)
        self.emb_path = Path(emb_path)
        self.chunks_path = Path(chunks_path)

        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index not found. Build it first.")
        if not self.chunks_path.exists():
            raise FileNotFoundError("Chunks metadata not found.")

        # load data
        self.index = faiss.read_index(str(self.index_path))
        self.embeddings = np.load(str(self.emb_path)).astype("float32")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)  # list of {"page":..., "text":...}

        # load embedding model for queries
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        v = self.model.encode([text], convert_to_numpy=True)
        v = v.astype("float32")
        # normalize
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return v

    def search(self, query, top_k=50):
        qv = self.embed_query(query)
        D, I = self.index.search(qv, top_k)  # D: inner product scores; I: indices
        # flatten
        scores = D[0].tolist()
        indices = [int(i) for i in I[0]]
        # build result list
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            meta = self.chunks[idx]
            results.append({
                "doc_id": idx,
                "score": float(score),
                "text": meta.get("text", "")[:800],  # truncated preview
                "page": meta.get("page")
            })
        return results
