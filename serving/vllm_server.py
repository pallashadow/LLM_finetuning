from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch


app = FastAPI(title="OpenSupport-LLM Inference API", version="0.1.0")
APP_CONFIG: dict[str, Any] = {
    "backend": "auto",
    "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    "adapter_path": None,
}


class AnswerRequest(BaseModel):
    question: str = Field(..., description="User question.")
    refs: list[str] = Field(..., description="Reference chunks used for grounded answering.")
    query_context_txt: str = Field("", description="Optional prior Q&A history text.")
    max_tokens: int = Field(256, ge=32, le=1024)


class AnswerResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float


def build_prompt(question: str, refs: list[str], query_context_txt: str = "") -> str:
    from training.prompt_templates import render_inference

    return render_inference(
        {
            "question": question,
            "refs": refs,
            "query_context_txt": query_context_txt,
        }
    )


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


def get_backend() -> InferenceBackend:
    backend = getattr(app.state, "backend", None)
    if backend is not None:
        return backend

    backend_name = str(APP_CONFIG.get("backend", "auto"))
    model_name_or_path = str(APP_CONFIG["model_name_or_path"])
    adapter_path = APP_CONFIG.get("adapter_path")

    if backend_name in {"vllm", "auto"}:
        try:
            app.state.backend = VLLMBackend(model_name_or_path=model_name_or_path, adapter_path=adapter_path)
            return app.state.backend
        except Exception:
            if backend_name == "vllm":
                raise

    app.state.backend = TransformersBackend(model_name_or_path=model_name_or_path, adapter_path=adapter_path)
    return app.state.backend


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": str(APP_CONFIG.get("backend", "auto"))}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    start = time.perf_counter()
    backend = get_backend()
    answer = backend.generate(
        build_prompt(
            question=req.question,
            refs=req.refs,
            query_context_txt=req.query_context_txt,
        ),
        max_tokens=req.max_tokens,
    )

    latency_ms = (time.perf_counter() - start) * 1000.0
    return AnswerResponse(
        answer=answer or "No answer generated.",
        model=backend.model_name,
        latency_ms=latency_ms,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenSupport-LLM vLLM server scaffold.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    APP_CONFIG["backend"] = args.backend
    APP_CONFIG["model_name_or_path"] = args.model_name_or_path
    APP_CONFIG["adapter_path"] = args.adapter_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


