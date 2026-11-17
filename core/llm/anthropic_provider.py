"""
Anthropic Claude LLM provider using REST API with aiohttp and urllib fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator

from core.llm.base import LLMProvider, ProviderCapabilities

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models (Claude 3.5 Sonnet, Haiku, Claude 4, etc.)."""

    name: str = "anthropic"
    default_base_url: str = "https://api.anthropic.com"
    default_model: str = "claude-3-5-sonnet-20241022"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._custom_api_key = api_key
        self._custom_base_url = base_url

    def get_api_key(self) -> str:
        if self._custom_api_key:
            return self._custom_api_key
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def get_base_url(self) -> str:
        if self._custom_base_url:
            return self._custom_base_url
        return os.environ.get("ANTHROPIC_BASE_URL", self.default_base_url)

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    def clean_model_name(self, model: str) -> str:
        if not model:
            return self.default_model
        if ":" in model and not model.startswith("http"):
            prefix, rest = model.split(":", 1)
            if prefix.lower() in ("anthropic", "claude"):
                return rest
        return model

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        clean = self.clean_model_name(model).lower()
        is_opus = "opus" in clean
        supports_thinking = "sonnet" in clean or "opus" in clean

        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_vision=True,
            supports_thinking=supports_thinking,
            supports_system_prompt=True,
            supports_json_mode=True,
            max_context_window=200000,
            max_output_tokens=32000 if is_opus else 8192,
        )

    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anthropic_tools = []
        for t in tools:
            if "name" in t and ("input_schema" in t or "parameters" in t):
                schema = t.get("input_schema", t.get("parameters", {}))
                anthropic_tools.append({
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "input_schema": schema,
                })
        return anthropic_tools

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Returns (system_prompt_str, anthropic_messages)."""
        sys_str = system_prompt or ""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

            if role == "system":
                if sys_str:
                    sys_str += "\n" + str(content)
                else:
                    sys_str = str(content)
            elif role in ("user", "assistant"):
                if isinstance(content, str):
                    api_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    text_parts = []
                    for b in content:
                        if isinstance(b, dict):
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            elif "text" in b:
                                text_parts.append(str(b["text"]))
                        else:
                            text_parts.append(str(b))
                    api_messages.append({"role": role, "content": "\n".join(text_parts)})
                else:
                    api_messages.append({"role": role, "content": str(content or "")})

        return sys_str, api_messages

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
        api_key = self.get_api_key()
        if not api_key:
            yield {"type": "error", "error": "Anthropic API key not configured. Set ANTHROPIC_API_KEY."}
            return

        clean_model = self.clean_model_name(model)
        base_url = self.get_base_url().rstrip("/")
        url = f"{base_url}/v1/messages"

        sys_str, api_messages = self.translate_messages(messages, system_prompt=system_prompt)

        payload: dict[str, Any] = {
            "model": clean_model,
            "messages": api_messages,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }
        if sys_str:
            payload["system"] = sys_str

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

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
                        yield {"type": "error", "error": f"Anthropic error ({resp.status}): {err_body}"}
                        return

                    full_text = ""
                    async for line_bytes in resp.content:
                        if abort_signal and abort_signal.is_set():
                            yield {"type": "error", "error": "Request cancelled by user"}
                            return

                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        try:
                            item = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = item.get("type", "")
                        if event_type == "content_block_delta":
                            delta = item.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    full_text += text
                                    yield {"type": "text_delta", "text": text}
                        elif event_type == "message_stop":
                            break

                    yield {"type": "done", "text": full_text, "model": clean_model}

        except asyncio.CancelledError:
            yield {"type": "error", "error": "Request cancelled by user"}
        except Exception as e:
            yield {"type": "error", "error": f"Anthropic connection error: {e}"}

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
                        if not line or not line.startswith("data:"):
                            continue

                        data_str = line[5:].strip()
                        try:
                            item = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = item.get("type", "")
                        if event_type == "content_block_delta":
                            delta = item.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    full_text += text
                                    q.put(("chunk", {"type": "text_delta", "text": text}))
                        elif event_type == "message_stop":
                            break

                    q.put(("chunk", {"type": "done", "text": full_text, "model": clean_model}))
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                q.put(("chunk", {"type": "error", "error": f"Anthropic error ({he.code}): {err_body}"}))
            except Exception as ex:
                q.put(("chunk", {"type": "error", "error": f"Anthropic connection error: {ex}"}))
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

