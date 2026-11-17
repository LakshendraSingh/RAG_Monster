import os
import unittest
from unittest.mock import MagicMock
from collections import deque

from core.semantic_cache import SemanticCache
from core.query_router import route, classify_query, DEFAULT_STRATEGY_MAP
from core.query_rewriter import rewrite_if_needed
from core.faithfulness_verifier import verify
from core.adaptation_engine import AdaptationEngine, _composite_score
from core.middleware import (
    RequestIDMiddleware,
    APIKeyMiddleware,
    RateLimitMiddleware,
    _AUTH_SKIP_PATHS,
)
import config


class TestEnterpriseConfig(unittest.TestCase):
    """Verify enterprise configuration defaults."""

    def test_config_defaults(self):
        self.assertEqual(config.API_KEY, "")
        self.assertIsInstance(config.RATE_LIMIT_PER_MINUTE, int)
        self.assertEqual(config.LLM_TEMPERATURE, 0.0)
        self.assertEqual(config.LLM_MAX_TOKENS, 2048)
        self.assertEqual(config.LLM_MAX_RETRIES, 3)


class TestSemanticCache(unittest.TestCase):
    """Test the in-memory semantic cache logic and threshold matching."""

    def setUp(self):
        self.cache = SemanticCache(similarity_threshold=0.90)

    def test_cache_miss_initially(self):
        emb = [1.0, 0.0, 0.0]
        res = self.cache.get(emb)
        self.assertIsNone(res)
        self.assertEqual(self.cache.misses, 1)

    def test_cache_hit_identical(self):
        emb = [1.0, 0.0, 0.0]
        self.cache.set("what is rag?", emb, {"answer": "Retrieval Augmented Generation"})

        hit = self.cache.get([1.0, 0.0, 0.0])
        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "Retrieval Augmented Generation")
        self.assertEqual(self.cache.hits, 1)

    def test_cache_hit_similar(self):
        # Cosine similarity between [1.0, 0.1, 0.0] and [1.0, 0.0, 0.0] is ~0.995 >= 0.90
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.05, 0.0]
        self.cache.set("what is rag?", emb1, {"answer": "RAG definition"})

        hit = self.cache.get(emb2)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "RAG definition")

    def test_cache_miss_dissimilar(self):
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        self.cache.set("what is rag?", emb1, {"answer": "RAG"})

        hit = self.cache.get(emb2)
        self.assertIsNone(hit)

    def test_cache_stats_and_clear(self):
        emb = [1.0, 0.0, 0.0]
        self.cache.set("q", emb, {"ans": "a"})
        self.cache.get(emb)  # hit
        self.cache.get([0.0, 1.0, 0.0])  # miss
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["cache_size"], 1)

        self.cache.clear()
        self.assertEqual(self.cache.stats()["cache_size"], 0)


class TestQueryRouter(unittest.TestCase):
    """Test query classification and strategy routing."""

    def test_route_fallback_without_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Service unavailable")

        decision = route("What is GDP?", llm=mock_llm)
        self.assertEqual(decision["type"], "factual")
        self.assertEqual(decision["strategy"], "dense")
        self.assertEqual(decision["top_k"], 3)

    def test_route_successful_json_parse(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '```json\n{"type": "comparative", "confidence": 0.9, "reasoning": "comparing two things"}\n```'

        decision = route("Compare stocks vs bonds", llm=mock_llm)
        self.assertEqual(decision["type"], "comparative")
        self.assertEqual(decision["strategy"], "hybrid")
        self.assertEqual(decision["top_k"], 6)

    def test_route_reasoning(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"type": "reasoning", "confidence": 0.85, "reasoning": "explaining mechanism"}'

        decision = route("Why does inflation devalue currency?", llm=mock_llm)
        self.assertEqual(decision["type"], "reasoning")
        self.assertEqual(decision["strategy"], "hybrid")
        self.assertEqual(decision["top_k"], 5)


class TestQueryRewriter(unittest.TestCase):
    """Test retrieval score evaluation and rewriting trigger."""

    def test_no_rewrite_when_scores_high(self):
        doc1 = MagicMock()
        doc1.metadata = {"relevance_score": 0.85}
        doc2 = MagicMock()
        doc2.metadata = {"relevance_score": 0.75}

        res = rewrite_if_needed("good query", [doc1, doc2], threshold=0.5)
        self.assertFalse(res["should_retry"])
        self.assertEqual(res["rewritten"], "good query")

    def test_rewrite_triggered_when_scores_low(self):
        doc1 = MagicMock()
        doc1.metadata = {"relevance_score": 0.3}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"rewritten": "better specific query", "strategy": "clarified"}'

        res = rewrite_if_needed("vague", [doc1], threshold=0.5, llm=mock_llm)
        self.assertTrue(res["should_retry"])
        self.assertEqual(res["rewritten"], "better specific query")

    def test_rewrite_triggered_when_empty_docs(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"rewritten": "rephrased query", "strategy": "broadened"}'

        res = rewrite_if_needed("nothing found", [], threshold=0.5, llm=mock_llm)
        self.assertTrue(res["should_retry"])
        self.assertEqual(res["rewritten"], "rephrased query")


class TestFaithfulnessVerifier(unittest.TestCase):
    """Test post-generation faithfulness verification."""

    def test_verify_faithful_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"faithful": true, "score": 0.95, "hallucinated_claims": [], "reason": "all supported"}'

        res = verify("What is X?", "X is 123.", "Context: X is 123.", llm=mock_llm)
        self.assertTrue(res["faithful"])
        self.assertGreaterEqual(res["score"], 0.9)
        self.assertEqual(len(res["hallucinated_claims"]), 0)

    def test_verify_unfaithful_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"faithful": false, "score": 0.2, "hallucinated_claims": ["X is 999"], "reason": "unsupported"}'

        res = verify("What is X?", "X is 999.", "Context: X is 123.", llm=mock_llm, min_score=0.6)
        self.assertFalse(res["faithful"])
        self.assertLess(res["score"], 0.6)
        self.assertIn("X is 999", res["hallucinated_claims"])

    def test_verify_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Model down")

        res = verify("What is X?", "Answer", "Context", llm=mock_llm)
        self.assertFalse(res["faithful"])  # Error must be treated as unfaithful
        self.assertEqual(res["score"], 0.0)


class TestAdaptationEngine(unittest.TestCase):
    """Test strategy adaptation calculations."""

    def test_composite_score(self):
        stats = {
            "avg_relevance": 0.8,
            "avg_faithfulness": 0.9,
            "avg_feedback": 1.0,
        }
        # 0.8*0.4 + 0.9*0.4 + 1.0*0.2 = 0.32 + 0.36 + 0.20 = 0.88
        score = _composite_score(stats)
        self.assertAlmostEqual(score, 0.88, places=2)

    def test_adaptation_promotes_better_strategy(self):
        mock_logger = MagicMock()
        # Pretend 'sparse' scored higher than 'dense' for 'factual'
        mock_logger.get_strategy_stats.return_value = {
            "factual:dense": {
                "count": 10,
                "avg_relevance": 0.5,
                "avg_faithfulness": 0.5,
                "avg_feedback": 0.5,
            },
            "factual:sparse": {
                "count": 10,
                "avg_relevance": 0.95,
                "avg_faithfulness": 0.95,
                "avg_feedback": 1.0,
            },
        }

        engine = AdaptationEngine(mock_logger)
        res = engine.run()

        self.assertEqual(res["updated_map"]["factual"]["strategy"], "sparse")
        self.assertEqual(len(res["changes"]), 1)
        self.assertEqual(res["changes"][0]["query_type"], "factual")
        self.assertEqual(res["changes"][0]["new_strategy"], "sparse")


class TestEnterpriseMiddlewareUnits(unittest.TestCase):
    """Unit test individual middleware logic."""

    def test_rate_limit_sliding_window(self):
        dummy_app = MagicMock()
        mw = RateLimitMiddleware(dummy_app)
        mw._limit = 3  # 3 requests per minute

        ip = "192.168.1.1"
        self.assertTrue(mw._is_allowed(ip))
        self.assertTrue(mw._is_allowed(ip))
        self.assertTrue(mw._is_allowed(ip))
        # 4th request within window must be rejected
        self.assertFalse(mw._is_allowed(ip))

        # Different IP must still be allowed
        self.assertTrue(mw._is_allowed("10.0.0.1"))

    def test_api_key_skip_paths(self):
        self.assertIn("/health", _AUTH_SKIP_PATHS)
        self.assertIn("/ready", _AUTH_SKIP_PATHS)
        self.assertIn("/docs", _AUTH_SKIP_PATHS)
        # /metrics is protected per security policy
        self.assertNotIn("/metrics", _AUTH_SKIP_PATHS)


class TestPipelineHelpers(unittest.TestCase):
    """Unit test pipeline retry and citation formatters."""

    def test_invoke_with_retry_succeeds_first_try(self):
        from main_rag import _invoke_with_retry
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "hello world"

        result = _invoke_with_retry(mock_llm, "test prompt", max_retries=2)
        self.assertEqual(result, "hello world")
        self.assertEqual(mock_llm.invoke.call_count, 1)

    def test_invoke_with_retry_succeeds_after_transient_failure(self):
        from main_rag import _invoke_with_retry
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [Exception("Rate limit"), "recovered answer"]

        result = _invoke_with_retry(mock_llm, "prompt", max_retries=2)
        self.assertEqual(result, "recovered answer")
        self.assertEqual(mock_llm.invoke.call_count, 2)

    def test_invoke_with_retry_exhausted_raises(self):
        from main_rag import _invoke_with_retry
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Permanent failure")

        with self.assertRaises(RuntimeError):
            _invoke_with_retry(mock_llm, "prompt", max_retries=2)

    def test_format_context_with_citations(self):
        from main_rag import RAGPipeline
        doc1 = MagicMock()
        doc1.metadata = {"source": "/path/to/handbook.pdf"}
        doc1.page_content = "Aurora handbook content."
        doc2 = MagicMock()
        doc2.metadata = {"source": "/path/to/runbook.txt"}
        doc2.page_content = "Infra runbook content."

        context, sources = RAGPipeline._format_context_with_citations([doc1, doc2])
        self.assertEqual(sources, ["handbook.pdf", "runbook.txt"])
        self.assertIn("[1] handbook.pdf", context)
        self.assertIn("[2] runbook.txt", context)



if __name__ == "__main__":
    unittest.main()
