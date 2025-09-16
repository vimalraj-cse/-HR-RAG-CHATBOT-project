# backend/app/reranker.py
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale
import re

def simple_tokenize(text):
    # lowercase + basic punctuation removal + split
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return tokens

class ReRanker:
    def __init__(self, chunks_path="data/chunks.json"):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.docs = [c.get("text","") for c in self.chunks]
        self.tokenized = [simple_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def bm25_scores_for_candidates(self, query, candidate_ids):
        q_tok = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tok)  # full array
        # select only candidate ids
        return np.array([scores[i] for i in candidate_ids], dtype=float)

    def rerank(self, query, candidates, cosine_scores, alpha=0.6):
        """
        candidates: list of doc indices (ints)
        cosine_scores: numpy array or list with same length -> raw inner-product scores
        alpha: weight for cosine (0..1). Combined = alpha*cosine_norm + (1-alpha)*bm25_norm
        """
        candidates = list(candidates)
        cosine_scores = np.array(cosine_scores, dtype=float)

        bm25_scores = self.bm25_scores_for_candidates(query, candidates)

        # Normalize to 0..1 for fair combination
        if len(cosine_scores) > 1:
            c_norm = minmax_scale(cosine_scores)
        else:
            c_norm = np.array([1.0])

        if len(bm25_scores) > 1:
            b_norm = minmax_scale(bm25_scores)
        else:
            b_norm = np.array([1.0])

        combined = alpha * c_norm + (1.0 - alpha) * b_norm
        # return list of tuples (doc_id, combined_score, cosine_score, bm25_score)
        order = np.argsort(-combined)
        ranked = []
        for i in order:
            ranked.append({
                "doc_id": candidates[i],
                "combined_score": float(combined[i]),
                "cosine_score": float(cosine_scores[i]),
                "bm25_score": float(bm25_scores[i])
            })
        return ranked
