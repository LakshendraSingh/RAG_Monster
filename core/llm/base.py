"""
Base interfaces and capability definitions for LLM providers.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator


@dataclass
class ProviderCapabilities:
    """Capabilities supported by a specific provider or model."""

    supports_streaming: bool = True
    supports_tool_use: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    supports_system_prompt: bool = True
    supports_json_mode: bool = False
    max_context_window: int = 128000
    max_output_tokens: int = 16384


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    name: str = "base"

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        model: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        abort_signal: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream responses from the LLM provider.

        Yields dictionaries with event types:
          - {"type": "text_delta", "text": "..."}
          - {"type": "done", "text": "...", "model": "..."}
          - {"type": "error", "error": "..."}
        """

    @abstractmethod
    def get_capabilities(self, model: str) -> ProviderCapabilities:
        """Return capabilities supported by this provider for the model."""

    @abstractmethod
    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate tools schema into provider's format."""

    @abstractmethod
    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Translate message format for this provider."""

    def is_configured(self) -> bool:
        """Check if provider has the required credentials or endpoints configured."""
        return True

    def _normalize_prompt_to_messages(
        self,
        prompt_or_messages: str | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Helper to convert string prompts into standardized message lists."""
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    async def acomplete(
        self,
        prompt: str | list[dict[str, Any]],
        system_prompt: str = "",
        model: str = "",
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generate a complete text response."""
        messages = self._normalize_prompt_to_messages(prompt)
        full_text = ""
        error_msg = None

        async for chunk in self.stream(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        ):
            chunk_type = chunk.get("type")
            if chunk_type == "text_delta":
                full_text += chunk.get("text", "")
            elif chunk_type == "done" and not full_text:
                full_text = chunk.get("text", "")
            elif chunk_type == "error":
                error_msg = chunk.get("error", "Unknown provider error")

        if error_msg and not full_text:
            raise RuntimeError(f"LLM Generation Error ({self.name}): {error_msg}")

        return full_text

    def complete(
        self,
        prompt: str | list[dict[str, Any]],
        system_prompt: str = "",
        model: str = "",
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronously generate a complete text response."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.acomplete(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )

        if loop and loop.is_running():
            # In an existing event loop, run in a separate thread to prevent blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

