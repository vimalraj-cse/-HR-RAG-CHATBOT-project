# backend/app/api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from .rag_pipeline import RAGPipeline

app = FastAPI(title="HR Chatbot API", version="1.0")

# Initialize RAG pipeline
rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_bot(request: QueryRequest):
    """
    Takes a user question and returns an answer + sources.
    """
    try:
        result = rag.answer(request.question)
        return {
            "question": request.question,
            "answer": result.get("answer"),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "HR Chatbot API is running!"}
