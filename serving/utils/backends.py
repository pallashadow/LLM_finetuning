from __future__ import annotations

from typing import Callable

import torch


class InferenceBackend:
    model_name: str

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class TransformersBackend(InferenceBackend):
    def __init__(self, model_name_or_path: str, adapter_path: str | None = None) -> None:
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
    def __init__(self, model_name_or_path: str, adapter_path: str | None = None) -> None:
        from vllm import LLM

        self.model_name = model_name_or_path
        self.llm = LLM(model=model_name_or_path, enable_lora=bool(adapter_path))
        self.adapter_path = adapter_path

    def generate(self, prompt: str, max_tokens: int) -> str:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outputs = self.llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return "No answer generated."
        return outputs[0].outputs[0].text.strip() or "No answer generated."


BackendFactory = Callable[[str, str | None], InferenceBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Backend name cannot be empty.")
        self._factories[key] = factory

    def create(self, name: str, model_name_or_path: str, adapter_path: str | None = None) -> InferenceBackend:
        key = name.strip().lower()
        if key not in self._factories:
            raise KeyError(f"Backend `{name}` is not registered.")
        return self._factories[key](model_name_or_path, adapter_path)

    def create_with_fallback(
        self,
        preferred_names: list[str],
        model_name_or_path: str,
        adapter_path: str | None = None,
    ) -> InferenceBackend:
        if not preferred_names:
            raise ValueError("preferred_names cannot be empty.")

        errors: list[tuple[str, Exception]] = []
        for name in preferred_names:
            try:
                return self.create(name=name, model_name_or_path=model_name_or_path, adapter_path=adapter_path)
            except Exception as exc:
                errors.append((name, exc))

        detail = ", ".join([f"{name}: {err}" for name, err in errors])
        raise RuntimeError(f"No available backend in fallback chain. Errors -> {detail}")
