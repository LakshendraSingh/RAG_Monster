"""
LangChain adapter and unified LLM wrapper for RAG_MONSTER.

Enables seamless drop-in replacement for LangChain's Ollama or other LLMs,
with full support for API providers and local Hugging Face / GGUF weights.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, List, Optional

from core.llm.base import LLMProvider
from core.llm.registry import get_provider, detect_provider

try:
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
    from langchain_core.outputs import GenerationChunk
    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
        from langchain.schema.output import GenerationChunk
        _HAS_LANGCHAIN = True
    except ImportError:
        _HAS_LANGCHAIN = False


if _HAS_LANGCHAIN:
    class UnifiedLLM(LLM):
        """LangChain LLM wrapper that routes calls to RAG_MONSTER providers."""

        model: str
        system_prompt: str = ""
        temperature: float = 0.2
        max_tokens: Optional[int] = None

        @property
        def _llm_type(self) -> str:
            provider_name, _ = detect_provider(self.model)
            return f"unified_{provider_name}"

        def _get_provider(self) -> LLMProvider:
            return get_provider(self.model)

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            provider = self._get_provider()
            call_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
            if stop:
                call_kwargs["stop"] = stop

            return provider.complete(
                prompt=prompt,
                system_prompt=self.system_prompt,
                model=self.model,
                **call_kwargs,
            )

        async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            provider = self._get_provider()
            call_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
            if stop:
                call_kwargs["stop"] = stop

            return await provider.acomplete(
                prompt=prompt,
                system_prompt=self.system_prompt,
                model=self.model,
                **call_kwargs,
            )

        def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> Iterator[GenerationChunk]:
            provider = self._get_provider()
            call_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
            if stop:
                call_kwargs["stop"] = stop

            messages = [{"role": "user", "content": prompt}]

            # Bridge async stream to synchronous generator
            import queue
            import threading

            q: queue.Queue[tuple[str, Any]] = queue.Queue()

            async def _run_stream():
                try:
                    async for chunk in provider.stream(
                        messages=messages,
                        system_prompt=self.system_prompt,
                        model=self.model,
                        **call_kwargs,
                    ):
                        if chunk.get("type") == "text_delta":
                            q.put(("chunk", chunk.get("text", "")))
                        elif chunk.get("type") == "error":
                            q.put(("error", chunk.get("error", "")))
                except Exception as ex:
                    q.put(("error", str(ex)))
                finally:
                    q.put(("done", None))

            def _thread_worker():
                asyncio.run(_run_stream())

            t = threading.Thread(target=_thread_worker, daemon=True)
            t.start()

            while True:
                msg_type, val = q.get()
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"LLM Stream Error: {val}")
                elif msg_type == "chunk":
                    if run_manager:
                        run_manager.on_llm_new_token(val)
                    yield GenerationChunk(text=val)

else:
    class UnifiedLLM:
        """Standalone fallback LLM wrapper when langchain is not installed."""

        def __init__(
            self,
            model: str,
            system_prompt: str = "",
            temperature: float = 0.2,
            max_tokens: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self.model = model
            self.system_prompt = system_prompt
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.extra_kwargs = kwargs

        def _get_provider(self) -> LLMProvider:
            return get_provider(self.model)

        def invoke(self, prompt: Any, **kwargs: Any) -> str:
            if hasattr(prompt, "to_string"):
                prompt_str = prompt.to_string()
            elif hasattr(prompt, "text"):
                prompt_str = prompt.text
            elif isinstance(prompt, str):
                prompt_str = prompt
            else:
                prompt_str = str(prompt)

            provider = self._get_provider()
            return provider.complete(
                prompt=prompt_str,
                system_prompt=self.system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **{**self.extra_kwargs, **kwargs},
            )

        async def ainvoke(self, prompt: Any, **kwargs: Any) -> str:
            if hasattr(prompt, "to_string"):
                prompt_str = prompt.to_string()
            elif isinstance(prompt, str):
                prompt_str = prompt
            else:
                prompt_str = str(prompt)

            provider = self._get_provider()
            return await provider.acomplete(
                prompt=prompt_str,
                system_prompt=self.system_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **{**self.extra_kwargs, **kwargs},
            )

        def stream(self, prompt: Any, **kwargs: Any) -> Iterator[str]:
            if hasattr(prompt, "to_string"):
                prompt_str = prompt.to_string()
            elif isinstance(prompt, str):
                prompt_str = prompt
            else:
                prompt_str = str(prompt)

            provider = self._get_provider()
            import queue
            import threading

            q: queue.Queue[tuple[str, Any]] = queue.Queue()

            async def _run_stream():
                try:
                    async for chunk in provider.stream(
                        messages=[{"role": "user", "content": prompt_str}],
                        system_prompt=self.system_prompt,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **{**self.extra_kwargs, **kwargs},
                    ):
                        if chunk.get("type") == "text_delta":
                            q.put(("chunk", chunk.get("text", "")))
                        elif chunk.get("type") == "error":
                            q.put(("error", chunk.get("error", "")))
                except Exception as ex:
                    q.put(("error", str(ex)))
                finally:
                    q.put(("done", None))

            def _thread_worker():
                asyncio.run(_run_stream())

            t = threading.Thread(target=_thread_worker, daemon=True)
            t.start()

            while True:
                msg_type, val = q.get()
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"LLM Stream Error: {val}")
                elif msg_type == "chunk":
                    yield val

        def __call__(self, prompt: Any, **kwargs: Any) -> str:
            return self.invoke(prompt, **kwargs)


def get_llm(
    model: str | None = None,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> UnifiedLLM:
    """Factory function to build a unified LLM compatible with LangChain."""
    from config import LLM_MODEL
    selected_model = model or LLM_MODEL
    return UnifiedLLM(
        model=selected_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

