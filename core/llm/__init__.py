"""
Unified LLM Provider Framework for RAG_MONSTER.

Provides support for:
1. API-based models:
   - OpenAI (GPT-4o, GPT-4o-mini, o1, o3-mini, etc.)
   - Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Opus, etc.)
   - Google Gemini (Gemini 2.5 Flash, 2.5 Pro, 2.0, etc.)
   - Groq (ultra-fast LPU inference)
   - DeepSeek (deepseek-chat, deepseek-reasoner)
   - Mistral AI (Mistral Large, Codestral, Pixtral)
   - OpenRouter, Together AI, Fireworks AI
   - Local OpenAI-compatible servers (vLLM, LMStudio, LocalAI)
2. Local Downloaded Weights:
   - Hugging Face downloaded weights / safetensors directories (via transformers + torch, MPS/CUDA/CPU)
   - GGUF files (via llama-cpp-python)
3. Ollama & llama.cpp servers
"""

from __future__ import annotations

import config  # Ensure .env is loaded
from core.llm.base import LLMProvider, ProviderCapabilities
from core.llm.anthropic_provider import AnthropicProvider
from core.llm.deepseek_provider import DeepSeekProvider
from core.llm.fireworks_provider import FireworksProvider
from core.llm.gemini_provider import GeminiProvider
from core.llm.groq_provider import GroqProvider
from core.llm.langchain_adapter import UnifiedLLM, get_llm
from core.llm.llamacpp_provider import LlamaCppProvider
from core.llm.mistral_provider import MistralProvider
from core.llm.ollama_provider import OllamaProvider
from core.llm.openai_compat_provider import OpenAICompatProvider
from core.llm.openai_provider import OpenAIProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.registry import (
    ProviderInfo,
    ProviderRegistry,
    detect_from_env,
    detect_provider,
    get_provider,
    get_registry,
    list_providers,
)
from core.llm.together_provider import TogetherProvider
from core.llm.weights_provider import (
    GGUFProvider,
    TransformersProvider,
    WeightsLLMProvider,
    WeightsProvider,
)

__all__ = [
    "LLMProvider",
    "ProviderCapabilities",
    "UnifiedLLM",
    "get_llm",
    "get_provider",
    "detect_provider",
    "list_providers",
    "detect_from_env",
    "get_registry",
    "ProviderRegistry",
    "ProviderInfo",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "GroqProvider",
    "DeepSeekProvider",
    "MistralProvider",
    "OpenRouterProvider",
    "TogetherProvider",
    "FireworksProvider",
    "OllamaProvider",
    "LlamaCppProvider",
    "OpenAICompatProvider",
    "WeightsProvider",
    "WeightsLLMProvider",
    "TransformersProvider",
    "GGUFProvider",
]

