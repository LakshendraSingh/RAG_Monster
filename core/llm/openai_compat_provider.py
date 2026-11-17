"""
OpenAI-compatible LLM provider base class.

Provides common streaming, SSE parsing, message translation, and REST communication
for OpenAI and OpenAI-compatible APIs (Groq, Mistral, DeepSeek, Together, Fireworks, OpenRouter, llama.cpp, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator

from core.llm.base import LLMProvider, ProviderCapabilities

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """Base provider for OpenAI and OpenAI-compatible REST APIs."""

    name: str = "openai_compat"
    default_base_url: str = "http://localhost:1234/v1"
    default_model: str = "gpt-4o"
    api_key_env_var: str = "OPENAI_API_KEY"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._custom_base_url = base_url
        self._custom_api_key = api_key
        self.extra_headers: dict[str, str] = extra_headers or {}

    def get_base_url(self) -> str:
        """Resolve base URL from instance, environment, or default."""
        if self._custom_base_url:
            return self._custom_base_url
        env_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
        if env_url:
            return env_url
        return self.default_base_url

    def get_api_key(self) -> str:
        """Resolve API key from instance, environment, or fallback."""
        if self._custom_api_key:
            return self._custom_api_key
        if self.api_key_env_var:
            val = os.environ.get(self.api_key_env_var, "")
            if val:
                return val
        return os.environ.get("OPENAI_API_KEY", "")

    def is_configured(self) -> bool:
        """Return True if credentials or local host are available."""
        base_url = self.get_base_url()
        if "localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url:
            return True
        return bool(self.get_api_key())

    def clean_model_name(self, model: str) -> str:
        """Strip provider prefix from model string if present."""
        if not model:
            return self.default_model
        if ":" in model and not model.startswith("http"):
            _, model_name = model.split(":", 1)
            return model_name
        return model

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        """Return capabilities for the given model."""
        clean = self.clean_model_name(model).lower()
        supports_thinking = "r1" in clean or "reasoner" in clean or clean.startswith(("o1", "o3", "o4"))
        supports_vision = any(v in clean for v in ["vision", "vl", "pixtral", "4o", "4.1", "4.5"])
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_vision=supports_vision,
            supports_thinking=supports_thinking,
            supports_system_prompt=True,
            supports_json_mode=True,
            max_context_window=128000,
            max_output_tokens=16384,
        )

    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate tools schema into OpenAI function calling format."""
        openai_tools: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                openai_tools.append(t)
                continue

            name = t.get("name", "")
            description = t.get("description", "")
            parameters = t.get("input_schema", t.get("parameters", {"type": "object", "properties": {}}))

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return openai_tools

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Translate messages into standard OpenAI message format."""
        api_messages: list[dict[str, Any]] = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

            if role in ("system", "user"):
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

            elif role == "assistant":
                if isinstance(content, str):
                    api_messages.append({"role": "assistant", "content": content})
                else:
                    api_messages.append({"role": "assistant", "content": str(content or "")})

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
        """Stream responses from the OpenAI-compatible endpoint."""
        clean_model = self.clean_model_name(model)
        base_url = self.get_base_url().rstrip("/")
        url = f"{base_url}/chat/completions"
        api_key = self.get_api_key()

        api_messages = self.translate_messages(messages, system_prompt=system_prompt)

        payload: dict[str, Any] = {
            "model": clean_model,
            "messages": api_messages,
            "stream": True,
        }

        if max_tokens:
            if clean_model.startswith(("o1", "o3", "o4")):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        if not clean_model.startswith(("o1", "o3", "o4")):
            payload["temperature"] = kwargs.get("temperature", 0.2)

        if tools:
            payload["tools"] = self.translate_tools(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"

        logger.debug(
            "%s request: url=%s, model=%s, messages=%d",
            self.name, url, clean_model, len(api_messages)
        )

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
                        err_msg = err_body
                        try:
                            parsed_err = json.loads(err_body)
                            if "error" in parsed_err:
                                err_obj = parsed_err["error"]
                                err_msg = err_obj.get("message", err_body) if isinstance(err_obj, dict) else str(err_obj)
                        except Exception:
                            pass
                        yield {"type": "error", "error": f"{self.name} error ({resp.status}): {err_msg}"}
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
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            yield {"type": "text_delta", "text": content}

                    yield {"type": "done", "text": full_text, "model": clean_model}

        except asyncio.CancelledError:
            yield {"type": "error", "error": "Request cancelled by user"}
        except Exception as e:
            logger.exception("%s streaming error", self.name)
            yield {"type": "error", "error": f"{self.name} connection error: {e}"}

    async def _stream_urllib(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        clean_model: str,
        abort_signal: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Fallback streaming via standard library urllib and threading."""
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
                        if data_str == "[DONE]":
                            break

                        try:
                            item = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = item.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text += content
                            q.put(("chunk", {"type": "text_delta", "text": content}))

                    q.put(("chunk", {"type": "done", "text": full_text, "model": clean_model}))
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                q.put(("chunk", {"type": "error", "error": f"{self.name} error ({he.code}): {err_body}"}))
            except Exception as ex:
                q.put(("chunk", {"type": "error", "error": f"{self.name} connection error: {ex}"}))
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

