from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

from main_rag import RAGPipeline
from config import HOST, PORT
from core.llm import list_providers, detect_from_env

app = FastAPI(
    title="Modular RAG System API",
    description="Modular Production Ready RAG System with Chroma, Redis, CrossEncoder Re-ranking, and Multi-Provider LLMs (API + Hugging Face / GGUF)",
    version="1.1.0"
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
    model: Optional[str] = None

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
        response = rag_pipeline.query(req.session_id, req.question, model=req.model)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
def query_stream_system(req: QueryModel):
    """Stream RAG response using Server-Sent Events (SSE)."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized.")
    
    def event_generator():
        try:
            for chunk in rag_pipeline.query_stream(req.session_id, req.question, model=req.model):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/providers")
def get_providers():
    """List all supported LLM providers and configuration status."""
    providers = list_providers()
    active = detect_from_env()
    return {
        "active_in_env": active,
        "providers": [
            {
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
                "default_model": p.default_model,
                "env_vars": p.env_vars,
                "is_configured": p.is_configured or (p.name in active),
                "capabilities": {
                    "supports_streaming": p.capabilities.supports_streaming,
                    "supports_tool_use": p.capabilities.supports_tool_use,
                    "supports_thinking": p.capabilities.supports_thinking,
                    "supports_vision": p.capabilities.supports_vision,
                    "max_context_window": p.capabilities.max_context_window,
                    "max_output_tokens": p.capabilities.max_output_tokens,
                }
            }
            for p in providers
        ]
    }

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
