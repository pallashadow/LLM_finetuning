from __future__ import annotations

import os
import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _ensure_transformers_tokenizer_compat() -> None:
    """Backfill tokenizer attributes expected by newer vLLM code paths."""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        # vLLM may access this field when adapting HF tokenizers.
        def _all_special_tokens_extended(self: PreTrainedTokenizerBase) -> list[str]:
            tokens = getattr(self, "all_special_tokens", [])
            return list(tokens) if isinstance(tokens, list) else list(tokens or [])

        setattr(PreTrainedTokenizerBase, "all_special_tokens_extended", property(_all_special_tokens_extended))


def _ensure_tqdm_asyncio_compat() -> None:
    """Patch tqdm.asyncio init for duplicate `disable` keyword incompatibilities."""
    try:
        from tqdm.asyncio import tqdm_asyncio
        from tqdm.std import tqdm as std_tqdm
    except Exception:
        return

    if getattr(tqdm_asyncio, "_opensupport_disable_patch", False):
        return

    def _patched_init(self, *args, **kwargs):
        # Some dependency combinations can pass `disable` twice and trigger a TypeError.
        disable = kwargs.pop("disable", False)
        # `disable` can also be supplied positionally (11th arg after `self`).
        # Only inject keyword `disable` when not already present positionally.
        if len(args) <= 10:
            kwargs.setdefault("disable", disable)
        std_tqdm.__init__(self, *args, **kwargs)

    setattr(tqdm_asyncio, "__init__", _patched_init)
    setattr(tqdm_asyncio, "_opensupport_disable_patch", True)


class InferenceBackend:
    model_name: str

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class TransformersBackend(InferenceBackend):
    def __init__(self, model_name_or_path: str, adapter_path: str | None = None, **_: Any) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model_name = f"{model_name_or_path}+{adapter_path}"
        else:
            self.model = base_model
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text or "No answer generated."


class VLLMBackend(InferenceBackend):
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str | None = None,
        max_model_len: int | None = None,
        **_: Any,
    ) -> None:
        _ensure_transformers_tokenizer_compat()
        _ensure_tqdm_asyncio_compat()
        from vllm import LLM

        self.model_name = model_name_or_path
        requested_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8"))
        gpu_memory_utilization = max(0.1, min(requested_utilization, 0.99))
        configured_max_model_len = max(
            512,
            int(max_model_len) if max_model_len is not None else int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        )
        enforce_eager = str(os.getenv("VLLM_ENFORCE_EAGER", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        # vLLM can fail at startup when requested utilization exceeds currently
        # free memory ratio. Clamp to a safe, runtime-derived ceiling.
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_ratio = free_bytes / total_bytes if total_bytes > 0 else 0.0
            safe_ceiling = max(0.1, min(0.99, free_ratio - 0.02))
            gpu_memory_utilization = min(gpu_memory_utilization, safe_ceiling)

        # On small GPUs, KV cache can fail after model load. Retry with shorter
        # context windows before giving up.
        max_len_candidates: list[int] = [configured_max_model_len]
        current = configured_max_model_len
        while current > 1024:
            current = max(1024, current // 2)
            if current not in max_len_candidates:
                max_len_candidates.append(current)

        last_error: Exception | None = None
        for candidate_max_len in max_len_candidates:
            try:
                self.llm = LLM(
                    model=model_name_or_path,
                    enable_lora=bool(adapter_path),
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=candidate_max_len,
                    enforce_eager=enforce_eager,
                )
                logger.info(
                    "Initialized vLLM with max_model_len=%s, gpu_memory_utilization=%s, enforce_eager=%s",
                    candidate_max_len,
                    gpu_memory_utilization,
                    enforce_eager,
                )
                break
            except ValueError as exc:
                msg = str(exc)
                # Retry for memory-related startup failures.
                if (
                    "desired GPU memory utilization" in msg
                    or "No available memory for the cache blocks" in msg
                ):
                    last_error = exc
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to initialize vLLM backend.")
        self.adapter_path = adapter_path

    def generate(self, prompt: str, max_tokens: int) -> str:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = self.llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return "No answer generated."
        return outputs[0].outputs[0].text.strip() or "No answer generated."


BackendFactory = Callable[..., InferenceBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Backend name cannot be empty.")
        self._factories[key] = factory

    def create(
        self,
        name: str,
        model_name_or_path: str,
        adapter_path: str | None = None,
        **backend_kwargs: Any,
    ) -> InferenceBackend:
        key = name.strip().lower()
        if key not in self._factories:
            raise KeyError(f"Backend `{name}` is not registered.")
        return self._factories[key](model_name_or_path, adapter_path, **backend_kwargs)

    def create_with_fallback(
        self,
        preferred_names: list[str],
        model_name_or_path: str,
        adapter_path: str | None = None,
        **backend_kwargs: Any,
    ) -> InferenceBackend:
        if not preferred_names:
            raise ValueError("preferred_names cannot be empty.")

        errors: list[tuple[str, Exception]] = []
        for name in preferred_names:
            try:
                return self.create(
                    name=name,
                    model_name_or_path=model_name_or_path,
                    adapter_path=adapter_path,
                    **backend_kwargs,
                )
            except Exception as exc:
                errors.append((name, exc))

        detail = ", ".join([f"{name}: {err}" for name, err in errors])
        raise RuntimeError(f"No available backend in fallback chain. Errors -> {detail}")
