"""
Google Gemini LLM provider using REST API with aiohttp and urllib fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator

from core.llm.base import LLMProvider, ProviderCapabilities

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini models (Gemini 2.5 Flash, Gemini 2.0, Gemini 1.5 Pro, etc.)."""

    name: str = "gemini"
    default_base_url: str = "https://generativelanguage.googleapis.com"
    default_model: str = "gemini-flash-latest"

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
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.get_api_key())

    def clean_model_name(self, model: str) -> str:
        if not model:
            return self.default_model

        if ":" in model and not model.startswith("http"):
            _, model_name = model.split(":", 1)
        else:
            model_name = model

        # Clean common prefixes/aliases
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        if model_name in ("", "gemini", "gemini-flash", "gemini-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
            return "gemini-flash-latest"
        if model_name in ("gemini-pro", "gemini-2.5-pro"):
            return "gemini-pro-latest"
        return model_name

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        clean = self.clean_model_name(model).lower()
        supports_thinking = "2.5" in clean or "pro" in clean

        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_vision=True,
            supports_thinking=supports_thinking,
            supports_system_prompt=True,
            supports_json_mode=True,
            max_context_window=1000000,
            max_output_tokens=65536 if "2.5" in clean else 8192,
        )

    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for t in tools:
            declarations.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {})),
            })
        return [{"functionDeclarations": declarations}] if declarations else []

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Returns (system_instruction, contents)."""
        sys_instruction = None
        if system_prompt:
            sys_instruction = {"parts": [{"text": system_prompt}]}

        raw_contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

            if role == "system":
                if not sys_instruction:
                    sys_instruction = {"parts": [{"text": str(content)}]}
                else:
                    sys_instruction["parts"].append({"text": str(content)})
                continue

            gemini_role = "user" if role == "user" else "model"
            parts: list[dict[str, Any]] = []

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            parts.append({"text": b.get("text", "")})
                        elif "text" in b:
                            parts.append({"text": str(b["text"])})
                    else:
                        parts.append({"text": str(b)})
            else:
                parts.append({"text": str(content or "")})

            if parts:
                raw_contents.append({"role": gemini_role, "parts": parts})

        # Merge consecutive turns with the same role (Gemini API requirement)
        merged_contents: list[dict[str, Any]] = []
        for turn in raw_contents:
            if merged_contents and merged_contents[-1]["role"] == turn["role"]:
                merged_contents[-1]["parts"].extend(turn["parts"])
            else:
                merged_contents.append(turn)

        return sys_instruction, merged_contents

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
            yield {"type": "error", "error": "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."}
            return

        clean_model = self.clean_model_name(model)
        base_url = (self._custom_base_url or self.default_base_url).rstrip("/")
        url = f"{base_url}/v1beta/models/{clean_model}:streamGenerateContent?alt=sse&key={api_key}"

        sys_inst, contents = self.translate_messages(messages, system_prompt=system_prompt)
        payload: dict[str, Any] = {"contents": contents}
        if sys_inst:
            payload["systemInstruction"] = sys_inst

        gen_config: dict[str, Any] = {}
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        gen_config["temperature"] = kwargs.get("temperature", 0.2)
        payload["generationConfig"] = gen_config

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
                        yield {"type": "error", "error": f"Gemini error ({resp.status}): {err_body}"}
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

                        candidates = item.get("candidates", [])
                        if not candidates:
                            continue

                        candidate_parts = candidates[0].get("content", {}).get("parts", [])
                        for p in candidate_parts:
                            text = p.get("text", "")
                            if text:
                                full_text += text
                                yield {"type": "text_delta", "text": text}

                    yield {"type": "done", "text": full_text, "model": clean_model}

        except asyncio.CancelledError:
            yield {"type": "error", "error": "Request cancelled by user"}
        except Exception as e:
            yield {"type": "error", "error": f"Gemini connection error: {e}"}

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
            import time
            max_retries = 3
            for attempt in range(max_retries):
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

                            candidates = item.get("candidates", [])
                            if not candidates:
                                continue

                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                text = p.get("text", "")
                                if text:
                                    full_text += text
                                    q.put(("chunk", {"type": "text_delta", "text": text}))

                        q.put(("chunk", {"type": "done", "text": full_text, "model": clean_model}))
                        break
                except urllib.error.HTTPError as he:
                    if he.code in (429, 503) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    err_body = he.read().decode("utf-8", errors="replace")
                    q.put(("chunk", {"type": "error", "error": f"Gemini error ({he.code}): {err_body}"}))
                    break
                except Exception as ex:
                    q.put(("chunk", {"type": "error", "error": f"Gemini connection error: {ex}"}))
                    break
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

