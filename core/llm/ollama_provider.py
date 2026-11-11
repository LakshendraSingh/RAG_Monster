"""
Ollama local model LLM provider using REST API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator

from core.llm.base import LLMProvider, ProviderCapabilities

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Provider for local Ollama server."""

    name: str = "ollama"
    default_base_url: str = "http://localhost:11434"
    default_model: str = "llama3:latest"

    def __init__(
        self,
        base_url: str | None = None,
    ) -> None:
        self._custom_base_url = base_url

    def get_base_url(self) -> str:
        if self._custom_base_url:
            return self._custom_base_url
        return os.environ.get("OLLAMA_BASE_URL", os.environ.get("OLLAMA_HOST", self.default_base_url))

    def is_configured(self) -> bool:
        return True

    def clean_model_name(self, model: str) -> str:
        if not model:
            return self.default_model
        if ":" in model and not model.startswith("http"):
            prefix, rest = model.split(":", 1)
            if prefix.lower() == "ollama":
                return rest
        return model

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        clean = self.clean_model_name(model).lower()
        supports_thinking = "r1" in clean or "deepseek" in clean
        supports_vision = "vision" in clean or "llava" in clean

        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_vision=supports_vision,
            supports_thinking=supports_thinking,
            supports_system_prompt=True,
            supports_json_mode=True,
            max_context_window=128000,
            max_output_tokens=8192,
        )

    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ollama_tools = []
        for t in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })
        return ollama_tools

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                content = "\n".join(text_parts)
            api_messages.append({"role": role, "content": str(content)})

        return api_messages

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
        clean_model = self.clean_model_name(model)
        base_url = self.get_base_url().rstrip("/")
        url = f"{base_url}/api/chat"

        api_messages = self.translate_messages(messages, system_prompt=system_prompt)
        payload: dict[str, Any] = {
            "model": clean_model,
            "messages": api_messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = self.translate_tools(tools)

        headers = {"Content-Type": "application/json"}

        try:
            import aiohttp
            has_aiohttp = True
        except ImportError:
            has_aiohttp = False

        if not has_aiohttp:
            async for chunk in self._stream_urllib(url, payload, headers, clean_model, abort_signal, **kwargs):
                yield chunk
            return

        try:
            timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", 180), sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        err_body = await resp.text()
                        yield {"type": "error", "error": f"Ollama error ({resp.status}): {err_body}"}
                        return

                    full_text = ""
                    async for line_bytes in resp.content:
                        if abort_signal and abort_signal.is_set():
                            yield {"type": "error", "error": "Request cancelled by user"}
                            return

                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg_obj = item.get("message", {})
                        text = msg_obj.get("content", "")
                        if text:
                            full_text += text
                            yield {"type": "text_delta", "text": text}

                        if item.get("done", False):
                            break

                    yield {"type": "done", "text": full_text, "model": clean_model}

        except asyncio.CancelledError:
            yield {"type": "error", "error": "Request cancelled by user"}
        except Exception as e:
            yield {"type": "error", "error": f"Ollama connection error: {e}"}

    async def _stream_urllib(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        clean_model: str,
        abort_signal: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        import queue
        import threading
        import urllib.error
        import urllib.request

        q: queue.Queue[tuple[str, Any]] = queue.Queue()

        def worker() -> None:
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=kwargs.get("timeout", 180)) as resp:
                    full_text = ""
                    for line_bytes in resp:
                        if abort_signal and abort_signal.is_set():
                            q.put(("chunk", {"type": "error", "error": "Request cancelled by user"}))
                            return

                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue

                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg_obj = item.get("message", {})
                        text = msg_obj.get("content", "")
                        if text:
                            full_text += text
                            q.put(("chunk", {"type": "text_delta", "text": text}))

                        if item.get("done", False):
                            break

                    q.put(("chunk", {"type": "done", "text": full_text, "model": clean_model}))
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                q.put(("chunk", {"type": "error", "error": f"Ollama error ({he.code}): {err_body}"}))
            except Exception as ex:
                q.put(("chunk", {"type": "error", "error": f"Ollama connection error: {ex}"}))
            finally:
                q.put(("done", None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            msg_type, val = item
            if msg_type == "done":
                break
            elif msg_type == "chunk":
                yield val

