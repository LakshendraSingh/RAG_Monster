"""
Direct model weight loader using Transformers (Hugging Face) or llama-cpp-python.

Provides a unified interface for loading:
1. Hugging Face directories / .safetensors (via transformers + torch)
2. GGUF files (via llama-cpp-python)
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from typing import Any, AsyncGenerator

from core.llm.base import LLMProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

# Global cache to keep models in memory across requests
_MODEL_CACHE: dict[str, Any] = {}


class WeightsProvider:
    """Dispatcher for direct model weight loading."""

    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))

    def _get_impl(self):
        """Detect model type and return appropriate provider implementation from cache."""
        if self.model_path in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_path]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights path does not exist: {self.model_path}")

        impl: Any = None
        # Detect Hugging Face directory or safetensors file
        if os.path.isdir(self.model_path):
            hf_markers = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer_config.json"]
            if any(os.path.exists(os.path.join(self.model_path, m)) for m in hf_markers):
                impl = TransformersProvider(self.model_path)
        elif self.model_path.endswith((".safetensors", ".bin")):
            parent_dir = os.path.dirname(self.model_path)
            if any(os.path.exists(os.path.join(parent_dir, m)) for m in ["config.json", "tokenizer_config.json"]):
                impl = TransformersProvider(parent_dir)
            else:
                impl = TransformersProvider(self.model_path)

        if not impl:
            # Default to GGUF
            impl = GGUFProvider(self.model_path)

        _MODEL_CACHE[self.model_path] = impl
        return impl

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion from the detected backend."""
        try:
            impl = self._get_impl()
            async for event in impl.stream(messages, system_prompt, **kwargs):
                yield event
        except Exception as e:
            logger.error("Model weights execution error: %s", e)
            yield {"type": "error", "error": f"Failed to load or run local weights: {e}"}


class GGUFProvider:
    """Provider for GGUF files using llama-cpp-python."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._llm = None

    def _get_llm(self, **kwargs: Any) -> Any:
        if self._llm:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required to run GGUF weights directly. "
                "Install with: pip install llama-cpp-python"
            )

        n_gpu_layers = kwargs.get("n_gpu_layers", -1 if "METAL" in os.environ or "CUDA" in os.environ else 0)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=kwargs.get("n_ctx", 8192),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            chat_format=kwargs.get("chat_format", "chatml"),
        )
        return self._llm

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        loop = asyncio.get_running_loop()
        api_messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages
        abort_signal = kwargs.get("abort_signal")

        def sync_stream():
            llm = self._get_llm(**kwargs)
            return llm.create_chat_completion(
                messages=api_messages,
                stream=True,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.1),
            )

        stream_iter = await loop.run_in_executor(None, sync_stream)
        full_text = ""
        for chunk in stream_iter:
            if abort_signal and abort_signal.is_set():
                yield {"type": "error", "error": "Request cancelled by user"}
                return
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                full_text += content
                yield {"type": "text_delta", "text": content}
            if chunk["choices"][0].get("finish_reason"):
                break
        yield {"type": "done", "text": full_text, "model": self.model_path}


class TransformersProvider:
    """Provider for Hugging Face models using transformers + torch (MPS / CUDA / CPU supported)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        except ImportError:
            raise ImportError(
                "torch and transformers are required to run Hugging Face / Safetensors weights directly. "
                "Install with: pip install torch transformers"
            )

        forced_device = os.environ.get("WEIGHTS_DEVICE", "auto").lower()
        if forced_device in ("mps", "cuda", "cpu"):
            device = forced_device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info("Loading Hugging Face model from %s onto %s...", self.model_path, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        try:
            config = AutoConfig.from_pretrained(self.model_path)
            model_type = getattr(config, "model_type", "").lower()
        except Exception:
            model_type = ""

        # Fallback template if missing
        if self._tokenizer.chat_template is None:
            if "llama" in model_type or "llama" in self.model_path.lower():
                logger.info("Injecting Llama-3 fallback template.")
                self._tokenizer.chat_template = (
                    "{% set loop_messages = messages %}"
                    "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
                    "{{ content }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
                )
            elif "phi" in model_type or "phi" in self.model_path.lower():
                logger.info("Injecting Phi fallback template (Instruct/Output).")
                self._tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ 'Instruct: ' + message['content'] + '\\n' if message['role'] == 'user' else 'Output: ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ 'Output: ' }}{% endif %}"
                )
            else:
                logger.info("Injecting Generic fallback template.")
                self._tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['role'].capitalize() + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
                )

        dtype = torch.float16 if device != "cpu" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._load()
        from transformers import TextIteratorStreamer
        from threading import Thread

        api_messages = [{"role": "system", "content": system_prompt}] + messages if system_prompt else messages
        prompt = self._tokenizer.apply_chat_template(
            api_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=False)

        # Determine EOS tokens
        eos_token_id = [self._tokenizer.eos_token_id]
        for token in ["<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>"]:
            if token in self._tokenizer.all_special_tokens:
                eid = self._tokenizer.convert_tokens_to_ids(token)
                if eid not in eos_token_id:
                    eos_token_id.append(eid)

        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = eos_token_id[0]

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_tokens", 4096),
            do_sample=True,
            temperature=kwargs.get("temperature", 0.1),
            eos_token_id=eos_token_id,
            pad_token_id=self._model.config.pad_token_id,
            repetition_penalty=1.1,
        )

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        full_text = ""
        loop = asyncio.get_running_loop()
        abort_signal = kwargs.get("abort_signal")

        def get_next():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while thread.is_alive() or not streamer.text_queue.empty():
            if abort_signal and abort_signal.is_set():
                yield {"type": "error", "error": "Request cancelled by user"}
                return
            new_text = await loop.run_in_executor(None, get_next)
            if new_text is None:
                break

            # Filter common system tokens if skip_special_tokens is False
            clean_text = new_text
            for token in self._tokenizer.all_special_tokens:
                if token in clean_text:
                    clean_text = clean_text.replace(token, "")

            if clean_text:
                full_text += clean_text
                yield {"type": "text_delta", "text": clean_text}

        yield {"type": "done", "text": full_text, "model": self.model_path}


class WeightsLLMProvider(LLMProvider):
    """Provider adapter registered for directly loaded local model weights (GGUF or Hugging Face)."""

    name: str = "weights"

    def is_configured(self) -> bool:
        return True

    def get_capabilities(self, model: str) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tool_use=False,
            supports_vision=False,
            supports_thinking=False,
            supports_system_prompt=True,
            supports_json_mode=False,
            max_context_window=32768,
            max_output_tokens=4096,
        )

    def translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools

    def translate_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                text_parts: list[str] = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                        elif "text" in b:
                            text_parts.append(str(b["text"]))
                    else:
                        text = getattr(b, "text", "") or getattr(b, "content", "")
                        if text:
                            text_parts.append(str(text))
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content or "")
            api_messages.append({"role": role, "content": content_str})
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
        path = model
        if path.startswith("weights:"):
            path = path[8:]

        provider = WeightsProvider(path)
        norm_messages = self.translate_messages(messages, system_prompt)
        async for chunk in provider.stream(
            norm_messages,
            system_prompt,
            max_tokens=max_tokens,
            abort_signal=abort_signal,
            **kwargs,
        ):
            if abort_signal and abort_signal.is_set():
                yield {"type": "error", "error": "Request cancelled by user"}
                return
            yield chunk

