"""
Unit tests for RAG_MONSTER Universal LLM Provider framework.
"""

import os
import tempfile
import unittest

from core.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    GroqProvider,
    MistralProvider,
    OllamaProvider,
    OpenAICompatProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
    WeightsLLMProvider,
    WeightsProvider,
    detect_provider,
    get_llm,
    get_provider,
    get_registry,
    list_providers,
)


class TestProviderRegistry(unittest.TestCase):
    """Test registry and dynamic model-to-provider routing."""

    def test_provider_detection_prefixes(self):
        cases = [
            ("openai:gpt-4o", "openai", "gpt-4o"),
            ("gpt-4o-mini", "openai", "gpt-4o-mini"),
            ("anthropic:claude-3-5-sonnet-20241022", "anthropic", "claude-3-5-sonnet-20241022"),
            ("claude-3-5-haiku", "anthropic", "claude-3-5-haiku"),
            ("gemini:gemini-flash-latest", "gemini", "gemini-flash-latest"),
            ("gemini-pro-latest", "gemini", "gemini-pro-latest"),
            ("groq:llama-3.3-70b-versatile", "groq", "llama-3.3-70b-versatile"),
            ("deepseek:deepseek-chat", "deepseek", "deepseek-chat"),
            ("mistral:mistral-large-latest", "mistral", "mistral-large-latest"),
            ("openrouter:anthropic/claude-3.5-sonnet", "openrouter", "anthropic/claude-3.5-sonnet"),
            ("together:meta-llama/Llama-3", "together", "meta-llama/Llama-3"),
            ("fireworks:accounts/fireworks/models/llama", "fireworks", "accounts/fireworks/models/llama"),
            ("ollama:llama3:latest", "ollama", "llama3:latest"),
            ("llamacpp:local-model", "llamacpp", "local-model"),
        ]

        for model_str, expected_prov, expected_clean in cases:
            prov_name, clean_name = detect_provider(model_str)
            self.assertEqual(
                prov_name,
                expected_prov,
                f"Failed for '{model_str}': expected provider '{expected_prov}', got '{prov_name}'",
            )
            self.assertEqual(
                clean_name,
                expected_clean,
                f"Failed clean name for '{model_str}': expected '{expected_clean}', got '{clean_name}'",
            )

    def test_provider_classes(self):
        self.assertIsInstance(get_provider("openai:gpt-4o"), OpenAIProvider)
        self.assertIsInstance(get_provider("anthropic:claude"), AnthropicProvider)
        self.assertIsInstance(get_provider("gemini:gemini-flash"), GeminiProvider)
        self.assertIsInstance(get_provider("groq:llama"), GroqProvider)
        self.assertIsInstance(get_provider("deepseek:deepseek-chat"), DeepSeekProvider)
        self.assertIsInstance(get_provider("mistral:large"), MistralProvider)
        self.assertIsInstance(get_provider("openrouter:model"), OpenRouterProvider)
        self.assertIsInstance(get_provider("together:model"), TogetherProvider)
        self.assertIsInstance(get_provider("ollama:llama3"), OllamaProvider)
        self.assertIsInstance(get_provider("weights:/tmp/test.gguf"), WeightsLLMProvider)

    def test_list_providers_coverage(self):
        providers = list_providers()
        names = {p.name for p in providers}
        expected = {"openai", "anthropic", "gemini", "groq", "deepseek", "mistral", "openrouter", "together", "fireworks", "ollama", "llamacpp", "weights"}
        self.assertTrue(expected.issubset(names))


class TestWeightsProvider(unittest.TestCase):
    """Test local weights detection for Hugging Face and GGUF."""

    def test_gguf_path_detection(self):
        prov_name, _ = detect_provider("./models/llama-3.gguf")
        self.assertEqual(prov_name, "weights")

        prov_name, _ = detect_provider("weights:/path/to/model.gguf")
        self.assertEqual(prov_name, "weights")

    def test_safetensors_path_detection(self):
        prov_name, _ = detect_provider("/models/model.safetensors")
        self.assertEqual(prov_name, "weights")

    def test_hf_directory_marker_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create HF marker file
            config_file = os.path.join(tmp_dir, "config.json")
            with open(config_file, "w") as f:
                f.write('{"model_type": "llama"}')

            prov_name, _ = detect_provider(tmp_dir)
            self.assertEqual(prov_name, "weights")


class TestLangChainAdapter(unittest.TestCase):
    """Test get_llm factory and UnifiedLLM wrapper."""

    def test_get_llm_creation(self):
        llm = get_llm("openai:gpt-4o", temperature=0.5, max_tokens=1024)
        self.assertEqual(llm.model, "openai:gpt-4o")
        self.assertEqual(llm.temperature, 0.5)
        self.assertEqual(llm.max_tokens, 1024)

    def test_get_llm_defaults(self):
        llm = get_llm()
        self.assertTrue(bool(llm.model))


if __name__ == "__main__":
    unittest.main()

