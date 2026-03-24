from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from main_rag import RAGPipeline
from config import HOST, PORT

app = FastAPI(
    title="Modular RAG System API",
    description="Modular Production Ready RAG System with Chroma, Redis, and CrossEncoder Re-ranking",
    version="1.0.0"
)

# Initialize the pipeline globally
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    print(f"Warning: RAG Pipeline initialization failed. Ensure you have the required models and databases running. Error: {e}")
    rag_pipeline = None

class QueryModel(BaseModel):
    session_id: str
    question: str

class IndexModel(BaseModel):
    source: str

@app.post("/index")
def index_document(req: IndexModel):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized.")
    try:
        result = rag_pipeline.index_document(req.source)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
def query_system(req: QueryModel):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized.")
    try:
        response = rag_pipeline.query(req.session_id, req.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized.")
    rag_pipeline.cache.clear_chat_history(session_id)
    return {"status": "History cleared", "session_id": session_id}

@app.delete("/db")
def clear_db():
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized.")
    rag_pipeline.clear_db()
    return {"status": "Database cleared"}

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
