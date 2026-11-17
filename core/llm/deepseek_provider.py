"""
DeepSeek LLM provider.
"""

from __future__ import annotations

import os

from core.llm.base import ProviderCapabilities
from core.llm.openai_compat_provider import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    """Provider for DeepSeek API (deepseek-chat V3, deepseek-reasoner R1)."""

    name: str = "deepseek"
    default_base_url: str = "https://api.deepseek.com/v1"
    api_key_env_var: str = "DEEPSEEK_API_KEY"

    def get_base_url(self) -> str:
        if self._custom_base_url:
            return self._custom_base_url
        return os.environ.get("DEEPSEEK_BASE_URL", self.default_base_url)

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        clean = self.clean_model_name(model).lower()
        supports_thinking = "reasoner" in clean or "r1" in clean

        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=not supports_thinking,
            supports_vision=False,
            supports_thinking=supports_thinking,
            supports_system_prompt=True,
            supports_json_mode=True,
            max_context_window=64000,
            max_output_tokens=8192,
        )

