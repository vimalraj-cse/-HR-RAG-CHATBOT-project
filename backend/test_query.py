# backend/test_query.py
from app.query_service import QueryService

qs = QueryService(index_path="data/faiss.index", emb_path="data/embeddings.npy", chunks_path="data/chunks.json")
q = "what is the maternity leave policy?"
res = qs.run_query(q, top_k_faiss=50, top_k_return=5, alpha=0.6, cache_ttl=3600)
print("SOURCES:")
for s in res["sources"]:
    print(s)
print("\nCONTEXT SNIPPET:")
print(res["context"][:2000])
print("\nRANKED (top):")
for r in res["ranked"]:
    print(r)
print("cached:", res["cached"])
