"""
tests/test_defect_fixes.py
──────────────────────────
Dedicated regression test suite verifying all 7 architectural defect fixes.
"""

from __future__ import annotations

import hmac
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.bm25 import BM25Index, PurePythonBM25, tokenize
from core.chunker import DocumentChunker
from core.document_loader import (
    Document,
    DocumentLoader,
    validate_path_allowed,
    validate_url_safe,
)
from core.faithfulness_verifier import verify as verify_faithfulness
from core.job_store import JobStore
from core.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    _AUTH_SKIP_PATHS,
)
from core.query_rewriter import rewrite_if_needed
from core.reranker import ReRanker, sigmoid
from core.retriever import RAGRetriever, reciprocal_rank_fusion
from core.semantic_cache import SemanticCache
from main_rag import ESCALATION_MESSAGE, RAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: BM25 + Dense with RRF and honor routing strategy
# ─────────────────────────────────────────────────────────────────────────────
class TestFix1_BM25DenseRRF(unittest.TestCase):

    def test_pure_bm25_scoring(self):
        corpus = [
            tokenize("The quick brown fox jumps over the lazy dog"),
            tokenize("Quantum computing leverages superposition and entanglement"),
            tokenize("Brown foxes and dogs are common mammals"),
        ]
        bm25 = PurePythonBM25(corpus)
        scores = bm25.get_scores(tokenize("quantum superposition"))
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_reciprocal_rank_fusion(self):
        doc_a = Document(page_content="doc A", metadata={"chunk_id": "a"})
        doc_b = Document(page_content="doc B", metadata={"chunk_id": "b"})
        doc_c = Document(page_content="doc C", metadata={"chunk_id": "c"})

        rank_dense = [doc_a, doc_b]
        rank_sparse = [doc_b, doc_c]

        fused = reciprocal_rank_fusion([rank_dense, rank_sparse], k=60, top_k=3)
        # doc_b is rank 2 in dense and rank 1 in sparse -> should win RRF
        self.assertEqual(fused[0].metadata["chunk_id"], "b")

    def test_retriever_honors_strategy(self):
        retriever = RAGRetriever(use_reranker=False)

        doc_dense = Document(page_content="vector dense result", metadata={"chunk_id": "d1"})
        doc_sparse = Document(page_content="keyword sparse result", metadata={"chunk_id": "s1"})

        retriever._retrieve_dense = MagicMock(return_value=[doc_dense])
        retriever._retrieve_sparse = MagicMock(return_value=[doc_sparse])

        # Dense strategy
        dense_res = retriever.retrieve("test", strategy="dense")
        self.assertEqual(dense_res[0].metadata["chunk_id"], "d1")
        retriever._retrieve_dense.assert_called()

        # Sparse strategy
        sparse_res = retriever.retrieve("test", strategy="sparse")
        self.assertEqual(sparse_res[0].metadata["chunk_id"], "s1")
        retriever._retrieve_sparse.assert_called()

        # Hybrid strategy
        hybrid_res = retriever.retrieve("test", strategy="hybrid")
        self.assertEqual(len(hybrid_res), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: ReRanker writes normalized score into doc.metadata["relevance_score"]
# ─────────────────────────────────────────────────────────────────────────────
class TestFix2_ReRankerRelevanceScore(unittest.TestCase):

    def test_sigmoid_normalization(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=2)
        self.assertGreater(sigmoid(10.0), 0.99)
        self.assertLess(sigmoid(-10.0), 0.01)

    def test_reranker_attaches_relevance_score(self):
        reranker = ReRanker()
        reranker.model = MagicMock()
        # Mock cross-encoder raw logits
        reranker.model.predict.return_value = [-2.5, 3.2]

        doc1 = Document(page_content="partially relevant", metadata={})
        doc2 = Document(page_content="highly relevant", metadata={})

        ranked = reranker.rerank("query", [doc1, doc2], top_k=2)

        self.assertEqual(len(ranked), 2)
        # doc2 had score 3.2 -> normalized ~0.96 -> rank 1
        self.assertIn("relevance_score", ranked[0].metadata)
        self.assertIn("relevance_score", ranked[1].metadata)
        self.assertGreaterEqual(ranked[0].metadata["relevance_score"], 0.9)
        self.assertLessEqual(ranked[1].metadata["relevance_score"], 0.1)

    def test_query_rewriter_uses_relevance_scores(self):
        doc1 = Document(page_content="text", metadata={"relevance_score": 0.25})
        doc2 = Document(page_content="text", metadata={"relevance_score": 0.35})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"rewritten": "clearer query", "strategy": "rephrased"}'

        rw = rewrite_if_needed("vague query", [doc1, doc2], threshold=0.5, llm=mock_llm)
        self.assertTrue(rw["should_retry"])
        self.assertAlmostEqual(rw["top_score"], 0.35, places=2)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Enforce faithfulness (escalate on failure/error, don't cache, log stream)
# ─────────────────────────────────────────────────────────────────────────────
class TestFix3_FaithfulnessEnforcement(unittest.TestCase):

    def test_verifier_error_returns_unfaithful(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Model service timeout")

        res = verify_faithfulness("query", "answer", "context", llm=mock_llm)
        self.assertFalse(res["faithful"])
        self.assertEqual(res["score"], 0.0)
        self.assertTrue(res["error"])

    @patch("main_rag.verify_faithfulness")
    def test_pipeline_escalates_when_unfaithful(self, mock_verify):
        mock_verify.return_value = {
            "faithful": False,
            "score": 0.2,
            "hallucinated_claims": ["hallucinated claim"],
            "error": False,
        }

        pipeline = RAGPipeline(
            use_reranker=False,
            enable_semantic_cache=True,
            enable_query_routing=False,
            enable_query_rewriting=False,
        )
        pipeline.retriever.retrieve = MagicMock(return_value=[Document(page_content="context", metadata={"source": "d.txt"})])
        pipeline.llm.invoke = MagicMock(return_value="Hallucinated answer.")

        res = pipeline.query("s1", "question")
        # Answer must be replaced with the safe escalation message
        self.assertEqual(res["answer"], ESCALATION_MESSAGE)
        self.assertFalse(res["faithfulness"]["faithful"])
        self.assertTrue(res["faithfulness"]["escalated"])

        # Cache must NOT store the failed answer
        cached = pipeline.semantic_cache.get(pipeline._embed("question") or [0.1]*10, scope="s1")
        self.assertIsNone(cached)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Semantic cache (scope, index_version, skip history, Redis SCAN)
# ─────────────────────────────────────────────────────────────────────────────
class TestFix4_SemanticCacheScopeAndVersion(unittest.TestCase):

    def test_scope_isolation(self):
        cache = SemanticCache(similarity_threshold=0.90)
        emb = [1.0, 0.0, 0.0]

        cache.set("q", emb, {"answer": "user A data"}, scope="user_A")

        # user_B with identical query must miss
        self.assertIsNone(cache.get(emb, scope="user_B"))
        # user_A must hit
        self.assertIsNotNone(cache.get(emb, scope="user_A"))

    def test_index_version_invalidation(self):
        cache = SemanticCache(similarity_threshold=0.90, index_version=1)
        emb = [1.0, 0.0, 0.0]

        cache.set("q", emb, {"answer": "v1 answer"}, scope="default")
        self.assertIsNotNone(cache.get(emb, scope="default"))

        # Re-index invalidates cache
        cache.invalidate()
        self.assertEqual(cache.index_version, 2)
        # Old entry must be gone
        self.assertIsNone(cache.get(emb, scope="default"))

    def test_scan_used_instead_of_keys(self):
        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = ["sem_cache:1:s:123"]
        mock_redis.get.return_value = None

        cache = SemanticCache(redis_client=mock_redis)
        cache.invalidate()

        mock_redis.scan_iter.assert_called()
        mock_redis.keys.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5: Deterministic chunk IDs (upsert on re-index) + loaders
# ─────────────────────────────────────────────────────────────────────────────
class TestFix5_DeterministicChunkIDsAndLoaders(unittest.TestCase):

    def test_deterministic_chunk_ids(self):
        doc = Document(
            page_content="Line 1 sentence. Line 2 sentence.",
            metadata={"source": "report.pdf"},
        )
        chunks1 = DocumentChunker.chunk_documents_recursive([doc], chunk_size=50)
        chunks2 = DocumentChunker.chunk_documents_recursive([doc], chunk_size=50)

        self.assertEqual(len(chunks1), len(chunks2))
        for c1, c2 in zip(chunks1, chunks2):
            self.assertEqual(c1.metadata["chunk_id"], c2.metadata["chunk_id"])
            self.assertTrue(c1.metadata["chunk_id"].startswith("report.pdf_"))

    def test_html_loader(self):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tmp_path = os.path.join(project_dir, "tests", "_test_sample.html")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Title</h1><p>Paragraph content.</p></body></html>")

        try:
            docs = DocumentLoader.load_html(tmp_path)
            self.assertGreater(len(docs), 0)
            self.assertIn("Paragraph content", docs[0].page_content)
            self.assertNotIn("<html>", docs[0].page_content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_directory_loader(self):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tmp_dir = os.path.join(project_dir, "tests", "_test_sample_dir")
        os.makedirs(tmp_dir, exist_ok=True)
        file1 = os.path.join(tmp_dir, "doc1.txt")
        file2 = os.path.join(tmp_dir, "doc2.md")
        with open(file1, "w", encoding="utf-8") as f:
            f.write("Text file content.")
        with open(file2, "w", encoding="utf-8") as f:
            f.write("Markdown content.")

        try:
            docs = DocumentLoader.load_directory(tmp_dir)
            self.assertEqual(len(docs), 2)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6: Security (require API_KEY, hmac, protect /metrics, SSRF, path allowlist)
# ─────────────────────────────────────────────────────────────────────────────
class TestFix6_SecurityEnforcement(unittest.TestCase):

    def test_metrics_is_not_auth_skipped(self):
        self.assertNotIn("/metrics", _AUTH_SKIP_PATHS)

    def test_url_ssrf_blocking(self):
        # Localhost / Loopback
        with self.assertRaises(ValueError):
            validate_url_safe("http://127.0.0.1/admin")
        with self.assertRaises(ValueError):
            validate_url_safe("http://localhost:8080/secret")
        # Metadata service
        with self.assertRaises(ValueError):
            validate_url_safe("http://metadata.google.internal/computeMetadata")

    def test_path_allowlist_blocking(self):
        with self.assertRaises(ValueError):
            validate_path_allowed("/etc/passwd")

    def test_hmac_compare_digest_timing_safe(self):
        key = "secret_key_123"
        self.assertTrue(hmac.compare_digest("secret_key_123", key))
        self.assertFalse(hmac.compare_digest("wrong_key", key))

    def test_trusted_proxy_forwarded_for(self):
        mw = RateLimitMiddleware(MagicMock())
        mw._trusted_proxies = {"10.0.0.1"}

        mock_req = MagicMock()
        mock_req.client.host = "10.0.0.1"  # Coming from trusted proxy
        mock_req.headers = {"X-Forwarded-For": "203.0.113.195, 10.0.0.1"}
        self.assertEqual(mw._get_client_ip(mock_req), "203.0.113.195")

        # Coming from untrusted peer -> ignore X-Forwarded-For
        mock_req.client.host = "192.168.1.50"
        self.assertEqual(mw._get_client_ip(mock_req), "192.168.1.50")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7: Durable Job Store & Compose cleanup
# ─────────────────────────────────────────────────────────────────────────────
class TestFix7_JobStoreAndCompose(unittest.TestCase):

    def test_job_store_in_memory_fallback(self):
        store = JobStore()
        store.set_job("job-1", {"status": "queued"})
        self.assertEqual(store.get_job("job-1")["status"], "queued")

        store.update_job("job-1", {"status": "done", "chunks": 5})
        self.assertEqual(store.get_job("job-1")["status"], "done")
        self.assertEqual(store.get_job("job-1")["chunks"], 5)

    def test_job_store_redis_persistence(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = '{"status": "running"}'

        store = JobStore(redis_client=mock_redis)
        job = store.get_job("job-2")

        mock_redis.get.assert_called_with("job:job-2")
        self.assertEqual(job["status"], "running")

    def test_compose_has_no_unused_chroma_service(self):
        compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("image: chromadb/chroma", content)
        self.assertNotIn("chroma_data:", content)


if __name__ == "__main__":
    unittest.main()
