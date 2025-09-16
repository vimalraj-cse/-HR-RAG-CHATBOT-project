import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def load_chunks(path="data/chunks.json"):
    """Load text chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate vector embeddings for all text chunks."""
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, chunks, out_path="data/embeddings"):
    """Save embeddings and chunks to disk."""
    np.save(f"{out_path}.npy", embeddings)
    with open(f"{out_path}_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved embeddings to {out_path}.npy and metadata to {out_path}_meta.json")

if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, chunks)
