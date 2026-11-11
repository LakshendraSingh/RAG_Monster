"""
Universal LLM Provider Registry for RAG_MONSTER.

Handles provider registration, hot-swapping, model-to-provider routing,
and environment-based provider detection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from core.llm.anthropic_provider import AnthropicProvider
from core.llm.base import LLMProvider, ProviderCapabilities
from core.llm.deepseek_provider import DeepSeekProvider
from core.llm.fireworks_provider import FireworksProvider
from core.llm.gemini_provider import GeminiProvider
from core.llm.groq_provider import GroqProvider
from core.llm.llamacpp_provider import LlamaCppProvider
from core.llm.mistral_provider import MistralProvider
from core.llm.ollama_provider import OllamaProvider
from core.llm.openai_compat_provider import OpenAICompatProvider
from core.llm.openai_provider import OpenAIProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.together_provider import TogetherProvider
from core.llm.weights_provider import WeightsLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderInfo:
    """Information and capability summary for a registered provider."""

    name: str
    display_name: str
    description: str
    default_model: str
    env_vars: list[str]
    is_configured: bool
    capabilities: ProviderCapabilities


class ProviderRegistry:
    """Registry managing available LLM providers and dynamic routing."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._prefix_map: dict[str, str] = {}
        self._default_provider_name: str = "ollama"

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
        display_name: str | None = None,
        description: str | None = None,
        default_model: str | None = None,
        env_vars: list[str] | None = None,
        prefixes: list[str] | None = None,
        is_default: bool = False,
    ) -> None:
        """Register or replace a provider implementation."""
        clean_name = name.lower()
        self._providers[clean_name] = provider
        self._metadata[clean_name] = {
            "display_name": display_name or clean_name.capitalize(),
            "description": description or f"{display_name or clean_name} LLM Provider",
            "default_model": default_model or "",
            "env_vars": env_vars or [],
        }

        self._prefix_map[clean_name] = clean_name
        if prefixes:
            for p in prefixes:
                self._prefix_map[p.lower()] = clean_name

        if is_default:
            self._default_provider_name = clean_name

    def detect_from_env(self) -> list[str]:
        """Detect which providers have valid API keys or active configuration."""
        active: list[str] = []
        for name, meta in self._metadata.items():
            env_vars = meta.get("env_vars", [])
            provider = self._providers.get(name)

            has_env_key = any(bool(os.environ.get(var)) for var in env_vars)
            if has_env_key or (provider and provider.is_configured() and name in ("ollama", "llamacpp")):
                active.append(name)
        return active

    def detect_provider_name(self, model: str) -> str:
        """Detect provider name from model string, file path, or environment."""
        if not model:
            return self._default_provider_name

        # 1. Direct local file or directory check (Hugging Face safetensors, GGUF, weights)
        expanded = os.path.expanduser(model)
        if (
            model.startswith("weights:")
            or model.endswith(".gguf")
            or model.endswith(".safetensors")
            or model.startswith("/")
            or model.startswith("./")
            or (os.path.exists(expanded) and not os.path.isdir(expanded))
        ):
            return "weights"

        if os.path.isdir(expanded):
            hf_markers = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer_config.json"]
            if any(os.path.exists(os.path.join(expanded, m)) for m in hf_markers):
                return "weights"

        # 2. Explicit prefix (e.g. "openai:gpt-4o", "gemini:gemini-2.5-flash", "anthropic:claude-3-5-sonnet")
        if ":" in model and not model.startswith("http"):
            prefix, _ = model.split(":", 1)
            clean_prefix = prefix.lower()
            if clean_prefix in self._prefix_map:
                return self._prefix_map[clean_prefix]

        clean_lower = model.lower()

        # Direct name match in prefix map (e.g. "gemini", "openai", "anthropic", "groq")
        if clean_lower in self._prefix_map:
            return self._prefix_map[clean_lower]

        # 3. Model name heuristic prefixes
        if clean_lower.startswith(("claude-", "claude")):
            return "anthropic"
        if clean_lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return "openai"
        if clean_lower.startswith("gemini-"):
            return "gemini"
        if clean_lower.startswith(("mistral-", "codestral-", "pixtral-")):
            return "mistral"
        if clean_lower.startswith("deepseek-"):
            return "deepseek"
        if clean_lower.startswith("groq/"):
            return "groq"
        if clean_lower.startswith(("meta-llama/", "togethercomputer/")):
            return "together"
        if clean_lower.startswith("accounts/fireworks/"):
            return "fireworks"

        # 4. Check available API keys in environment
        available = self.detect_from_env()
        if "anthropic" in available and any(x in clean_lower for x in ["claude", "haiku", "sonnet", "opus"]):
            return "anthropic"
        if "openai" in available and "gpt" in clean_lower:
            return "openai"
        if "gemini" in available and "gemini" in clean_lower:
            return "gemini"
        if "deepseek" in available and "deepseek" in clean_lower:
            return "deepseek"

        # 5. Default provider
        return self._default_provider_name

    def get_provider(self, model: str) -> LLMProvider:
        """Resolve model string to a provider instance."""
        provider_name = self.detect_provider_name(model)
        provider = self._providers.get(provider_name)

        if not provider:
            provider = self._providers.get(self._default_provider_name)
            if not provider and self._providers:
                provider = next(iter(self._providers.values()))

        if not provider:
            raise RuntimeError(
                f"No LLM provider available for model '{model}'. "
                f"Registered providers: {list(self._providers.keys())}"
            )

        return provider

    def list_providers(self) -> list[ProviderInfo]:
        """List all registered providers and their metadata."""
        infos: list[ProviderInfo] = []
        for name, provider in self._providers.items():
            meta = self._metadata.get(name, {})
            caps = provider.get_capabilities(meta.get("default_model", ""))
            infos.append(
                ProviderInfo(
                    name=name,
                    display_name=meta.get("display_name", name),
                    description=meta.get("description", ""),
                    default_model=meta.get("default_model", ""),
                    env_vars=meta.get("env_vars", []),
                    is_configured=provider.is_configured(),
                    capabilities=caps,
                )
            )
        return infos


# ---------------------------------------------------------------------------
# Global Registry Singleton and Initialization
# ---------------------------------------------------------------------------

_REGISTRY: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get or initialize the global provider registry singleton."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    registry = ProviderRegistry()

    # 1. OpenAI
    registry.register_provider(
        name="openai",
        provider=OpenAIProvider(),
        display_name="OpenAI",
        description="OpenAI GPT-4o, GPT-4.1, o1, o3, o4 models",
        default_model="gpt-4o",
        env_vars=["OPENAI_API_KEY"],
        prefixes=["openai", "gpt", "o1", "o3", "o4"],
    )

    # 2. Anthropic
    registry.register_provider(
        name="anthropic",
        provider=AnthropicProvider(),
        display_name="Anthropic Claude",
        description="Anthropic Claude 3.5 Sonnet, Haiku, Opus models",
        default_model="claude-3-5-sonnet-20241022",
        env_vars=["ANTHROPIC_API_KEY"],
        prefixes=["anthropic", "claude"],
    )

    # 3. Google Gemini
    registry.register_provider(
        name="gemini",
        provider=GeminiProvider(),
        display_name="Google Gemini",
        description="Google Gemini Flash and Pro models",
        default_model="gemini-flash-latest",
        env_vars=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        prefixes=["gemini", "google"],
    )

    # 4. Groq
    registry.register_provider(
        name="groq",
        provider=GroqProvider(),
        display_name="Groq Cloud",
        description="Ultra-fast Llama, Mixtral, and Gemma inference on LPUs",
        default_model="llama-3.3-70b-versatile",
        env_vars=["GROQ_API_KEY"],
        prefixes=["groq"],
    )

    # 5. DeepSeek
    registry.register_provider(
        name="deepseek",
        provider=DeepSeekProvider(),
        display_name="DeepSeek",
        description="DeepSeek-Chat (V3) and DeepSeek-Reasoner (R1)",
        default_model="deepseek-chat",
        env_vars=["DEEPSEEK_API_KEY"],
        prefixes=["deepseek"],
    )

    # 6. Mistral AI
    registry.register_provider(
        name="mistral",
        provider=MistralProvider(),
        display_name="Mistral AI",
        description="Mistral Large, Codestral, and Pixtral models",
        default_model="mistral-large-latest",
        env_vars=["MISTRAL_API_KEY"],
        prefixes=["mistral", "codestral"],
    )

    # 7. OpenRouter
    registry.register_provider(
        name="openrouter",
        provider=OpenRouterProvider(),
        display_name="OpenRouter",
        description="Unified gateway to 200+ commercial and open models",
        default_model="anthropic/claude-3.5-sonnet",
        env_vars=["OPENROUTER_API_KEY"],
        prefixes=["openrouter"],
    )

    # 8. Together AI
    registry.register_provider(
        name="together",
        provider=TogetherProvider(),
        display_name="Together AI",
        description="Together AI open-source model inference API",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        env_vars=["TOGETHER_API_KEY"],
        prefixes=["together"],
    )

    # 9. Fireworks AI
    registry.register_provider(
        name="fireworks",
        provider=FireworksProvider(),
        display_name="Fireworks AI",
        description="Fireworks AI fast production model inference",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        env_vars=["FIREWORKS_API_KEY"],
        prefixes=["fireworks"],
    )

    # 10. Ollama
    registry.register_provider(
        name="ollama",
        provider=OllamaProvider(),
        display_name="Ollama (Local)",
        description="Local Ollama server models",
        default_model="llama3:latest",
        env_vars=["OLLAMA_BASE_URL", "OLLAMA_HOST"],
        prefixes=["ollama"],
        is_default=True,
    )

    # 11. llama.cpp server
    registry.register_provider(
        name="llamacpp",
        provider=LlamaCppProvider(),
        display_name="llama.cpp server",
        description="Local llama.cpp OpenAI-compatible HTTP server",
        default_model="local-model",
        env_vars=["LLAMACPP_HOST"],
        prefixes=["llamacpp", "llama"],
    )

    # 12. Hugging Face / GGUF local weights
    registry.register_provider(
        name="weights",
        provider=WeightsLLMProvider(),
        display_name="Local Model Weights",
        description="Directly loaded Hugging Face (.safetensors) or GGUF weights files",
        default_model="./models",
        env_vars=["WEIGHTS_DEVICE"],
        prefixes=["weights", "hf", "local_weights"],
    )

    # 13. Generic OpenAI-compatible local/remote
    registry.register_provider(
        name="openai_compat",
        provider=OpenAICompatProvider(),
        display_name="OpenAI-Compatible Endpoint",
        description="Generic endpoint compatible with OpenAI format (vLLM, LMStudio, etc.)",
        default_model="default",
        env_vars=["OPENAI_BASE_URL", "OPENAI_API_BASE"],
        prefixes=["openai_compat", "vllm", "lmstudio", "local"],
    )

    _REGISTRY = registry
    return _REGISTRY


def get_provider(model: str) -> LLMProvider:
    """Resolve model string to a provider instance."""
    return get_registry().get_provider(model)


def detect_provider(model: str) -> tuple[str, str]:
    """Return (provider_name, clean_model_name)."""
    registry = get_registry()
    provider_name = registry.detect_provider_name(model)
    provider = registry.get_provider(model)
    clean_name = getattr(provider, "clean_model_name", lambda m: m)(model)
    return provider_name, clean_name


def list_providers() -> list[ProviderInfo]:
    """List all registered providers."""
    return get_registry().list_providers()


def detect_from_env() -> list[str]:
    """List names of providers with credentials present in the environment."""
    return get_registry().detect_from_env()

