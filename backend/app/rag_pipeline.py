# backend/app/rag_pipeline.py
import os
from groq import Groq
from dotenv import load_dotenv
from .query_service import QueryService

load_dotenv()  # Load GROQ_API_KEY from .env

class RAGPipeline:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.qs = QueryService(
            index_path="data/faiss.index",
            emb_path="data/embeddings.npy",
            chunks_path="data/chunks.json"
        )

    def build_prompt(self, query, context):
        """Create a prompt for the LLM using retrieved context."""
        return f"""
You are an AI assistant for HR policies. 
Answer the question using only the CONTEXT below. 
If not found, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

Answer:
"""

    def answer(self, query):
        # 1. Retrieve relevant context
        res = self.qs.run_query(query, top_k_faiss=50, top_k_return=5, alpha=0.6)
        context = res.get("context", "")

        # 2. Build prompt
        prompt = self.build_prompt(query, context)

        # 3. Call Groq LLM
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",  # free Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        res["answer"] = answer
        return res


if __name__ == "__main__":
    rag = RAGPipeline()
    q = "What is the maternity leave policy?"
    result = rag.answer(q)
    print("\nANSWER:", result["answer"])
    print("\nSOURCES:", result["sources"])
