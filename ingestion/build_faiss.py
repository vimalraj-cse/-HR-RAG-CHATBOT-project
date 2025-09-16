# ingestion/build_faiss.py
import numpy as np
import faiss
from pathlib import Path

def build_faiss(emb_path="data/embeddings.npy", out_index_path="data/faiss.index"):
    emb_path = Path(emb_path)
    out_index_path = Path(out_index_path)

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    # load embeddings
    emb = np.load(str(emb_path))  # shape: (n_docs, dim)
    print("Loaded embeddings:", emb.shape)

    # ensure float32
    emb = emb.astype("float32")

    # normalize for cosine similarity (cosine(a,b) == dot(norm(a), norm(b)))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product index
    print("Index type:", type(index).__name__, "dimension:", dim)

    index.add(emb)  # add vectors
    print("Added vectors to index. n_total:", index.ntotal)

    # save index to disk
    faiss.write_index(index, str(out_index_path))
    print(f"Saved FAISS index to {out_index_path}")

if __name__ == "__main__":
    build_faiss()
