from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import HOST, PORT
from core.job_store import JobStore
from core.llm import detect_from_env, list_providers
from core.middleware import APIKeyMiddleware, RateLimitMiddleware, RequestIDMiddleware
from main_rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Durable Job Store (Redis-backed with in-memory fallback) ─────────────────

job_store = JobStore()

# ─── App lifecycle ────────────────────────────────────────────────────────────

rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        if rag_pipeline.cache.use_redis and rag_pipeline.cache.redis_client:
            job_store.redis_client = rag_pipeline.cache.redis_client
        logger.info("rag_pipeline_ready")
    except Exception as e:
        logger.error("rag_pipeline_init_failed", extra={"error": str(e)})
        raise RuntimeError(f"RAGPipeline failed to initialise: {e}") from e

    yield

    if rag_pipeline and rag_pipeline.feedback_logger:
        rag_pipeline.feedback_logger.close()
    logger.info("rag_pipeline_shutdown")


app = FastAPI(
    title="Modular RAG System API",
    description=(
        "Production RAG with Chroma, Redis, BM25+Dense RRF, CrossEncoder Re-ranking, "
        "Scoped SemanticCache, QueryRouter, QueryRewriter, FaithfulnessVerifier, "
        "FeedbackLogger, and AdaptationEngine."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# ─── Middleware (Restricted CORS + Rate Limiter + Auth + RequestID) ───────────

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RequestIDMiddleware)

# ─── Prometheus metrics (soft import, protected by auth middleware) ───────────

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    pass

# ─── Dependency ───────────────────────────────────────────────────────────────

def get_pipeline() -> RAGPipeline:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized.")
    return rag_pipeline


# ─── Request / Response models ────────────────────────────────────────────────

class QueryModel(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = None

class IndexModel(BaseModel):
    source: str

class CodebaseIndexModel(BaseModel):
    source: str
    max_file_size_kb: int = 1000

class FeedbackModel(BaseModel):
    log_id: int
    feedback: int   # 1 = good, 0 = bad


# ─── Existing endpoints (unversioned — kept for backwards compat) ──────────────

@app.get("/health", tags=["ops"])
def health():
    """Shallow liveness probe."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Deep readiness probe — checks Chroma and Redis."""
    checks: dict = {}
    try:
        pipeline.vdb.vector_store.similarity_search("ping", k=1)
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = f"error: {e}"
    checks["redis"] = "ok" if pipeline.cache.use_redis else "in-memory fallback"
    if any("error" in str(v) for v in checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return checks


@app.post("/index", tags=["ingestion"])
def index_document(req: IndexModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        return pipeline.index_document(req.source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("index_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Indexing failed. See server logs.")


@app.post("/index/codebase", tags=["ingestion"])
def index_codebase(req: CodebaseIndexModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Index an entire codebase (300k+ LoC) using AST and logical structural chunking."""
    try:
        return pipeline.index_codebase(req.source, max_file_size_kb=req.max_file_size_kb)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("index_codebase_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Codebase indexing failed. See server logs.")


@app.post("/query", tags=["query"])
def query_system(req: QueryModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    try:
        return pipeline.query(req.session_id, req.question, model=req.model)
    except Exception as e:
        logger.error("query_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Query failed. See server logs.")


@app.post("/query/stream", tags=["query"])
def query_stream_system(req: QueryModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    def event_generator():
        try:
            for chunk in pipeline.query_stream(req.session_id, req.question, model=req.model):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as err:
            logger.error("stream_error", extra={"error": str(err)})
            yield f"data: {json.dumps({'error': 'Stream failed. See server logs.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback", tags=["adaptation"])
def submit_feedback(req: FeedbackModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    if req.feedback not in (0, 1):
        raise HTTPException(status_code=400, detail="feedback must be 0 or 1")
    pipeline.add_feedback(req.log_id, req.feedback)
    return {"status": "feedback recorded", "log_id": req.log_id, "feedback": req.feedback}


@app.post("/adapt", tags=["adaptation"])
def run_adaptation(pipeline: RAGPipeline = Depends(get_pipeline)):
    return pipeline.adapt()


@app.get("/semantic-cache/stats", tags=["ops"])
def semantic_cache_stats(pipeline: RAGPipeline = Depends(get_pipeline)):
    if not pipeline.semantic_cache:
        return {"enabled": False}
    return {"enabled": True, **pipeline.semantic_cache.stats()}


@app.get("/providers", tags=["ops"])
def get_providers():
    providers = list_providers()
    active = detect_from_env()
    return {
        "active_in_env": active,
        "providers": [
            {
                "name":          p.name,
                "display_name":  p.display_name,
                "description":   p.description,
                "default_model": p.default_model,
                "env_vars":      p.env_vars,
                "is_configured": p.is_configured or (p.name in active),
                "capabilities": {
                    "supports_streaming":    p.capabilities.supports_streaming,
                    "supports_tool_use":     p.capabilities.supports_tool_use,
                    "supports_thinking":     p.capabilities.supports_thinking,
                    "supports_vision":       p.capabilities.supports_vision,
                    "max_context_window":    p.capabilities.max_context_window,
                    "max_output_tokens":     p.capabilities.max_output_tokens,
                },
            }
            for p in providers
        ],
    }


@app.delete("/history/{session_id}", tags=["session"])
def clear_history(session_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    pipeline.cache.clear_chat_history(session_id)
    return {"status": "history cleared", "session_id": session_id}


@app.delete("/db", tags=["admin"])
def clear_db(pipeline: RAGPipeline = Depends(get_pipeline)):
    pipeline.clear_db()
    return {"status": "database cleared"}


# ─── Async background indexing with Durable Redis Job Store ───────────────────

def _run_index_job(job_id: str, source: str, pipeline: RAGPipeline):
    try:
        job_store.update_job(job_id, {"status": "running"})
        result = pipeline.index_document(source)
        job_store.update_job(job_id, {"status": "done", "result": result})
    except Exception as e:
        job_store.update_job(job_id, {"status": "error", "detail": str(e)})


@app.post("/index/async", tags=["ingestion"])
def index_document_async(
    req: IndexModel,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    job_id = str(uuid4())
    job_store.set_job(job_id, {"status": "queued", "source": req.source})
    background_tasks.add_task(_run_index_job, job_id, req.source, pipeline)
    return {"job_id": job_id, "status": "queued"}


@app.get("/index/jobs/{job_id}", tags=["ingestion"])
def get_index_job(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


# ─── Versioned router (v1) ────────────────────────────────────────────────────

v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.get("/health", tags=["ops"])
def v1_health():
    return health()


@v1.get("/ready", tags=["ops"])
def v1_ready(pipeline: RAGPipeline = Depends(get_pipeline)):
    return ready(pipeline)


@v1.post("/index", tags=["ingestion"])
def v1_index_document(req: IndexModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    return index_document(req, pipeline)


@v1.post("/index/codebase", tags=["ingestion"])
def v1_index_codebase(req: CodebaseIndexModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    return index_codebase(req, pipeline)


@v1.post("/index/async", tags=["ingestion"])
def v1_index_document_async(
    req: IndexModel,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    return index_document_async(req, background_tasks, pipeline)


@v1.get("/index/jobs/{job_id}", tags=["ingestion"])
def v1_get_index_job(job_id: str):
    return get_index_job(job_id)


@v1.post("/query", tags=["query"])
def v1_query_system(req: QueryModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    return query_system(req, pipeline)


@v1.post("/query/stream", tags=["query"])
def v1_query_stream_system(req: QueryModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    return query_stream_system(req, pipeline)


@v1.delete("/history/{session_id}", tags=["session"])
def v1_clear_history(session_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    return clear_history(session_id, pipeline)


@v1.delete("/db", tags=["admin"])
def v1_clear_db(pipeline: RAGPipeline = Depends(get_pipeline)):
    return clear_db(pipeline)


@v1.post("/adapt", tags=["adaptation"])
def v1_run_adaptation(pipeline: RAGPipeline = Depends(get_pipeline)):
    return run_adaptation(pipeline)


@v1.post("/feedback", tags=["adaptation"])
def v1_submit_feedback(req: FeedbackModel, pipeline: RAGPipeline = Depends(get_pipeline)):
    return submit_feedback(req, pipeline)


@v1.get("/semantic-cache/stats", tags=["ops"])
def v1_semantic_cache_stats(pipeline: RAGPipeline = Depends(get_pipeline)):
    return semantic_cache_stats(pipeline)


@v1.get("/providers", tags=["ops"])
def v1_get_providers():
    return get_providers()


app.include_router(v1)


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
