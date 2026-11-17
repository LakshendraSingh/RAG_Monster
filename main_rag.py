import os
import logging
try:
    import langchain
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    from langchain_core.prompts import PromptTemplate
except ImportError:
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.template = template
        def format(self, **kwargs):
            return self.template.format(**kwargs)

from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.vdb import VectorDB
from core.retriever import RAGRetriever
from core.cache import CacheManager
from core.semantic_cache import SemanticCache
from core.query_router import route as route_query
from core.query_rewriter import rewrite_if_needed
from core.faithfulness_verifier import verify as verify_faithfulness
from core.feedback_logger import FeedbackLogger
from core.adaptation_engine import AdaptationEngine
from config import LLM_MODEL
from core.llm import get_llm
import concurrent.futures
try:
    from config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES
except ImportError:
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES = 0.0, 2048, 3

logger = logging.getLogger(__name__)

ESCALATION_MESSAGE = (
    "I cannot verify the accuracy of this response against the retrieved source documents. "
    "Please escalate to a human reviewer or consult the original documentation directly."
)


def _invoke_with_retry(llm, prompt: str, max_retries: int = LLM_MAX_RETRIES) -> str:
    """Invoke LLM with simple exponential backoff retry on transient errors."""
    import time
    last_exc = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "llm_invoke_retry",
                extra={"attempt": attempt + 1, "wait_s": wait, "error": str(exc)},
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM invoke failed after {max_retries} attempts: {last_exc}") from last_exc


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline with BM25+Dense RRF retrieval,
    scoped semantic caching, faithfulness enforcement, and feedback logging.
    """

    _PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a highly capable AI assistant powered by a Modular RAG system.
Use the following pieces of relevant context and previous chat history to answer the question.
If the answer cannot be found in the context, explicitly say so.
Always cite the numbered sources as [1], [2], etc. when you use information from them.

--- Chat History ---
{chat_history}

--- Context ---
{context}

--- Question ---
{question}

Answer (cite sources as [1], [2], ...):"""
    )

    def __init__(
        self,
        use_reranker: bool = True,
        model: str | None = None,
        enable_semantic_cache: bool = True,
        enable_query_routing: bool = True,
        enable_query_rewriting: bool = True,
        enable_faithfulness_check: bool = True,
        enable_feedback_logging: bool = True,
        semantic_cache_threshold: float = 0.92,
        faithfulness_min_score: float = 0.6,
        rewrite_threshold: float = 0.5,
    ):
        self.model = model or LLM_MODEL
        self.vdb = VectorDB()
        self.retriever = RAGRetriever(use_reranker=use_reranker)
        self.cache = CacheManager()
        self.llm = get_llm(model=self.model, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
        self.prompt_template = self._PROMPT_TEMPLATE

        # Feature flags
        self._routing_enabled   = enable_query_routing
        self._rewrite_enabled   = enable_query_rewriting
        self._faith_enabled     = enable_faithfulness_check
        self._faith_min_score   = faithfulness_min_score
        self._rewrite_threshold = rewrite_threshold

        # Semantic cache
        redis_client = self.cache.redis_client if self.cache.use_redis else None
        self.semantic_cache = (
            SemanticCache(
                similarity_threshold=semantic_cache_threshold,
                redis_client=redis_client,
            )
            if enable_semantic_cache
            else None
        )

        # Feedback logging & adaptation
        self.feedback_logger = FeedbackLogger() if enable_feedback_logging else None
        self.adaptation_engine = (
            AdaptationEngine(self.feedback_logger) if enable_feedback_logging else None
        )

        self._strategy_map = None

        logger.info(
            "rag_pipeline_initialized",
            extra={
                "model":              self.model,
                "semantic_cache":     enable_semantic_cache,
                "query_routing":      enable_query_routing,
                "query_rewriting":    enable_query_rewriting,
                "faithfulness_check": enable_faithfulness_check,
                "feedback_logging":   enable_feedback_logging,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public: Indexing
    # ─────────────────────────────────────────────────────────────────────────

    def index_document(self, file_path_or_url: str):
        """Loads, chunks, and indexes a document into VectorDB & BM25 index."""
        from config import CHUNKING_STRATEGY
        from core.embedder import Embedder

        logger.info("index_document_start", extra={"source": file_path_or_url})
        docs = DocumentLoader.load_any(file_path_or_url)

        if CHUNKING_STRATEGY == "semantic":
            logger.info("chunking_semantic", extra={"doc_count": len(docs)})
            embeddings = Embedder.get_embeddings()
            chunked_docs = DocumentChunker.chunk_documents_semantic(docs, embeddings)
        else:
            logger.info("chunking_recursive", extra={"doc_count": len(docs)})
            chunked_docs = DocumentChunker.chunk_documents_recursive(docs)

        logger.info("indexing_chunks", extra={"chunk_count": len(chunked_docs)})
        # Add to vector database (upsert with deterministic IDs)
        self.vdb.add_documents(chunked_docs)
        # Add to sparse BM25 index
        self.retriever.add_documents(chunked_docs)

        # Invalidate semantic cache when index changes
        if self.semantic_cache:
            self.semantic_cache.invalidate()

        return {
            "status":         "success",
            "chunks_indexed": len(chunked_docs),
            "source":         file_path_or_url,
        }

    def index_codebase(self, root_path: str, max_file_size_kb: int = 1000):
        """
        Specialized ingestion for large codebases (supporting 300k+ Lines of Code).
        Loads code files concurrently, applies AST/syntax-aware logical chunking,
        and indexes into VectorDB and BM25 with deterministic IDs.
        """
        logger.info("index_codebase_start", extra={"root_path": root_path})
        docs = DocumentLoader.load_codebase(root_path, max_file_size_kb=max_file_size_kb)
        logger.info("codebase_loaded", extra={"file_count": len(docs), "root_path": root_path})

        chunked_docs = DocumentChunker.chunk_documents_recursive(docs)
        logger.info("codebase_chunked", extra={"chunk_count": len(chunked_docs)})

        self.vdb.add_documents(chunked_docs)
        self.retriever.add_documents(chunked_docs)

        if self.semantic_cache:
            self.semantic_cache.invalidate()

        total_lines = sum(d.metadata.get("line_count", 0) for d in docs)
        return {
            "status": "success",
            "files_indexed": len(docs),
            "lines_of_code": total_lines,
            "chunks_indexed": len(chunked_docs),
            "source": root_path,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public: Query (non-streaming)
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, session_id: str, question: str, model: str | None = None):
        """
        Full RAG query with scoped semantic caching, query routing, query rewriting,
        source attribution, and faithfulness enforcement.
        """
        import time
        t0 = time.perf_counter()

        # ── 0. Chat history ───────────────────────────────────────────────
        history = self.cache.get_chat_history(session_id)
        has_history = bool(history)
        history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-5:]]
        )

        # ── 1. Semantic cache check (skip when chat history is present) ───
        query_embedding = self._embed(question)
        if not has_history and self.semantic_cache and query_embedding:
            cached = self.semantic_cache.get(query_embedding, scope=session_id)
            if cached:
                logger.info("query_semantic_cache_hit", extra={"session": session_id})
                self.cache.add_chat_message(session_id, "user", question)
                self.cache.add_chat_message(session_id, "assistant", cached["answer"])
                return {**cached, "cache_hit": True}

        # ── 2. Query routing ──────────────────────────────────────────────
        routing = {"type": "factual", "strategy": "dense", "top_k": 5, "confidence": 0.5}
        if self._routing_enabled:
            try:
                routing = route_query(
                    question,
                    strategy_map=self._strategy_map,
                    llm=self.llm,
                )
            except Exception as exc:
                logger.warning("query_routing_error", extra={"error": str(exc)})

        logger.info(
            "query_routed",
            extra={
                "type":     routing["type"],
                "strategy": routing["strategy"],
                "top_k":    routing["top_k"],
            },
        )

        # ── 3. Retrieval honoring strategy (dense, sparse, hybrid) ─────────
        docs = self.retriever.retrieve(
            question,
            top_k=routing["top_k"],
            strategy=routing["strategy"],
        )

        # ── 4. Query rewriting + retry ────────────────────────────────────
        was_rewritten = False
        rewritten_query = question
        if self._rewrite_enabled and docs is not None:
            rw = rewrite_if_needed(
                question,
                docs,
                threshold=self._rewrite_threshold,
                llm=self.llm,
            )
            if rw["should_retry"]:
                was_rewritten   = True
                rewritten_query = rw["rewritten"]
                logger.info(
                    "query_rewritten",
                    extra={"original": question, "rewritten": rewritten_query},
                )
                retry_docs = self.retriever.retrieve(
                    rewritten_query,
                    top_k=routing["top_k"],
                    strategy=routing["strategy"],
                )
                if len(retry_docs) > len(docs):
                    docs = retry_docs

        # ── 5. Format context with numbered citations ──────────────────────
        context, sources = self._format_context_with_citations(docs)

        # ── 6. Generate ───────────────────────────────────────────────────
        active_model = model or self.model
        llm = (
            get_llm(model=active_model, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
            if model and model != self.model
            else self.llm
        )

        prompt = self.prompt_template.format(
            context=context,
            question=rewritten_query,
            chat_history=history_str,
        )
        answer = _invoke_with_retry(llm, prompt)

        # ── 7. Faithfulness verification & enforcement ────────────────────
        faith_result = {"faithful": True, "score": 1.0, "hallucinated_claims": [], "reason": "skipped", "error": False}
        if self._faith_enabled:
            try:
                faith_result = verify_faithfulness(
                    question, answer, context,
                    llm=llm,
                    min_score=self._faith_min_score,
                )
            except Exception as exc:
                logger.warning("faithfulness_check_error", extra={"error": str(exc)})
                faith_result = {"faithful": False, "score": 0.0, "reason": str(exc), "error": True, "hallucinated_claims": [str(exc)]}

        # Enforce faithfulness: return safe escalation response on failure or error
        is_faithful = (
            bool(faith_result.get("faithful", False))
            and (float(faith_result.get("score", 0.0)) >= self._faith_min_score)
            and not faith_result.get("error", False)
        )

        can_cache = is_faithful
        if not is_faithful:
            logger.warning(
                "faithfulness_enforcement_triggered",
                extra={"score": faith_result.get("score"), "reason": faith_result.get("reason")},
            )
            answer = ESCALATION_MESSAGE

        elapsed = time.perf_counter() - t0

        # ── 8. Update chat history ────────────────────────────────────────
        self.cache.add_chat_message(session_id, "user", question)
        self.cache.add_chat_message(session_id, "assistant", answer)

        # ── 9. Feedback log ───────────────────────────────────────────────
        log_id = None
        if self.feedback_logger:
            scores = [
                float(d.metadata.get("relevance_score", 0.0)) for d in docs
            ]
            log_id = self.feedback_logger.log(
                query                = question,
                query_type           = routing["type"],
                strategy_used        = routing["strategy"],
                top_k                = routing["top_k"],
                num_chunks_retrieved = len(docs),
                num_chunks_passed    = len(docs),
                avg_relevance_score  = sum(scores) / max(len(scores), 1),
                top_relevance_score  = max(scores, default=0.0),
                faithfulness_score   = faith_result.get("score", 0.0),
                was_rewritten        = was_rewritten,
                rewritten_query      = rewritten_query if was_rewritten else None,
                answer               = answer,
                model_used           = active_model,
            )

        result = {
            "answer":          answer,
            "context_used":    context,
            "model_used":      active_model,
            "sources":         sources,
            "query_type":      routing["type"],
            "strategy":        routing["strategy"],
            "faithfulness":    {
                "score":               faith_result.get("score", 1.0),
                "faithful":            is_faithful,
                "hallucinated_claims": faith_result.get("hallucinated_claims", []),
                "escalated":           not is_faithful,
            },
            "was_rewritten":   was_rewritten,
            "rewritten_query": rewritten_query if was_rewritten else None,
            "log_id":          log_id,
            "latency_ms":      round(elapsed * 1000, 1),
            "cache_hit":       False,
        }

        # Store in semantic cache ONLY if answer is faithful AND no chat history was present
        if can_cache and not has_history and self.semantic_cache and query_embedding:
            self.semantic_cache.set(question, query_embedding, result, scope=session_id)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Public: Streaming query
    # ─────────────────────────────────────────────────────────────────────────

    def query_stream(self, session_id: str, question: str, model: str | None = None):
        """Streaming RAG query with real post-stream faithfulness logging."""
        history = self.cache.get_chat_history(session_id)
        has_history = bool(history)
        history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-5:]]
        )

        query_embedding = self._embed(question)
        if not has_history and self.semantic_cache and query_embedding:
            cached = self.semantic_cache.get(query_embedding, scope=session_id)
            if cached:
                yield cached["answer"]
                return

        routing = {"type": "factual", "strategy": "dense", "top_k": 5}
        if self._routing_enabled:
            try:
                routing = route_query(question, strategy_map=self._strategy_map, llm=self.llm)
            except Exception:
                pass

        docs = self.retriever.retrieve(
            question,
            top_k=routing["top_k"],
            strategy=routing["strategy"],
        )

        was_rewritten, rewritten_query = False, question
        if self._rewrite_enabled and docs:
            rw = rewrite_if_needed(question, docs, threshold=self._rewrite_threshold, llm=self.llm)
            if rw["should_retry"]:
                was_rewritten = True
                rewritten_query = rw["rewritten"]
                retry_docs = self.retriever.retrieve(
                    rewritten_query,
                    top_k=routing["top_k"],
                    strategy=routing["strategy"],
                )
                if len(retry_docs) > len(docs):
                    docs = retry_docs

        context, sources = self._format_context_with_citations(docs)

        active_model = model or self.model
        llm = (
            get_llm(model=active_model, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
            if model and model != self.model
            else self.llm
        )

        prompt = self.prompt_template.format(
            context=context, question=rewritten_query, chat_history=history_str,
        )

        full_answer = ""
        for chunk in llm.stream(prompt):
            chunk_str = str(chunk)
            full_answer += chunk_str
            yield chunk_str

        self.cache.add_chat_message(session_id, "user", question)
        self.cache.add_chat_message(session_id, "assistant", full_answer)

        # Real post-stream faithfulness evaluation
        stream_faith = verify_faithfulness(
            question, full_answer, context,
            llm=llm,
            min_score=self._faith_min_score,
        )
        real_score = float(stream_faith.get("score", 0.0))
        is_stream_faithful = bool(stream_faith.get("faithful", False)) and (real_score >= self._faith_min_score)

        if self.feedback_logger:
            scores = [float(d.metadata.get("relevance_score", 0.0)) for d in docs]
            self.feedback_logger.log(
                query=question, query_type=routing["type"],
                strategy_used=routing["strategy"], top_k=routing["top_k"],
                num_chunks_retrieved=len(docs), num_chunks_passed=len(docs),
                avg_relevance_score=sum(scores) / max(len(scores), 1),
                top_relevance_score=max(scores, default=0.0),
                faithfulness_score=real_score,
                was_rewritten=was_rewritten,
                rewritten_query=rewritten_query if was_rewritten else None,
                answer=full_answer, model_used=active_model,
            )

        # Store in cache only if verified faithful and without multi-turn history
        if is_stream_faithful and not has_history and self.semantic_cache and query_embedding:
            self.semantic_cache.set(
                question,
                query_embedding,
                {
                    "answer": full_answer,
                    "context_used": context,
                    "model_used": active_model,
                    "sources": sources,
                },
                scope=session_id,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public: Adaptation
    # ─────────────────────────────────────────────────────────────────────────

    def adapt(self) -> dict:
        """Run one adaptation cycle to recalibrate query routing."""
        if not self.adaptation_engine:
            return {"message": "Adaptation disabled (enable_feedback_logging=False)"}
        result = self.adaptation_engine.run()
        self._strategy_map = result["updated_map"]
        return result

    def add_feedback(self, log_id: int, feedback: int):
        """Record user feedback (1=thumbs-up, 0=thumbs-down)."""
        if self.feedback_logger:
            self.feedback_logger.add_feedback(log_id, feedback)

    # ─────────────────────────────────────────────────────────────────────────
    # Public: DB management
    # ─────────────────────────────────────────────────────────────────────────

    def clear_db(self):
        """Clear vector database, sparse index, and invalidate semantic cache."""
        self.vdb.clear_database()
        self.retriever.clear()
        if self.semantic_cache:
            self.semantic_cache.invalidate()
        return {"status": "success"}

    def deep_reset(self):
        """Physically delete the Chroma directory, clear sparse index & cache."""
        import shutil
        from config import CHROMA_PATH
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        self.vdb = VectorDB()
        self.retriever.clear()
        if self.semantic_cache:
            self.semantic_cache.invalidate()
        return {"status": "success"}

    def delete_collection(self, name: str):
        self.vdb.delete_collection(name)
        self.retriever.clear()
        if self.semantic_cache:
            self.semantic_cache.invalidate()
        return {"status": "success"}

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float] | None:
        try:
            from core.embedder import Embedder
            embeddings_model = Embedder.get_embeddings()
            return embeddings_model.embed_query(text)
        except Exception as exc:
            logger.warning("embed_error", extra={"error": str(exc)})
            return None

    @staticmethod
    def _format_context_with_citations(docs) -> tuple[str, list[str]]:
        import os as _os
        parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source_path = doc.metadata.get("source", "unknown")
            source_name = _os.path.basename(source_path)
            sources.append(source_name)
            parts.append(f"[{i}] {source_name}\n{doc.page_content}")
        return "\n\n".join(parts), sources
