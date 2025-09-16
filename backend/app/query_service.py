# backend/app/query_service.py
from .retriever import Retriever
from .reranker import ReRanker
from .cache_layer import make_cache_key, get_cached, set_cached

class QueryService:
    def __init__(self, index_path="data/faiss.index", emb_path="data/embeddings.npy", chunks_path="data/chunks.json"):
        self.retriever = Retriever(index_path=index_path, emb_path=emb_path, chunks_path=chunks_path)
        self.reranker = ReRanker(chunks_path=chunks_path)

    def run_query(self, query, top_k_faiss=50, top_k_return=5, alpha=0.6, cache_ttl=3600):
        key = make_cache_key(query, top_k_faiss, alpha)
        cached = get_cached(key)
        if cached:
            cached["cached"] = True
            return cached

        # 1) FAISS initial retrieval
        faiss_results = self.retriever.search(query, top_k=top_k_faiss)
        candidate_ids = [r["doc_id"] for r in faiss_results]
        cosine_scores = [r["score"] for r in faiss_results]

        if len(candidate_ids) == 0:
            return {"answer": None, "sources": [], "cached": False}

        # 2) Rerank
        ranked = self.reranker.rerank(query, candidate_ids, cosine_scores, alpha=alpha)

        # select top_k_return
        top_ranked = ranked[:top_k_return]

        # assemble sources and context
        sources = []
        context_pieces = []
        for r in top_ranked:
            doc_id = r["doc_id"]
            meta = self.retriever.chunks[doc_id]
            sources.append({"doc_id": doc_id, "page": meta.get("page")})
            context_pieces.append(f"SourceID:{doc_id} Page:{meta.get('page')}\n{meta.get('text')}")

        result = {
            "answer": None,  # to be filled by LLM in your /query endpoint
            "sources": sources,
            "context": "\n\n".join(context_pieces),
            "ranked": top_ranked,
            "cached": False
        }

        # cache for TTL seconds
        set_cached(key, result, expire=cache_ttl)
        return result
