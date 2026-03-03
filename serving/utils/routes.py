from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    input: str = Field(..., description="Generic input text for inference.")
    max_tokens: int = Field(256, ge=32, le=1024)


class AnswerResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=32, le=1024)
    temperature: float | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(256, ge=32, le=1024)
    temperature: float | None = None
    stream: bool = False


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if p.strip()])
    return str(content)


def _messages_to_input_text(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _message_content_to_text(message.content).strip()
        if not text:
            continue
        role = str(message.role or "user").strip().lower()
        lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


async def _parse_invocation_payload(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty request body.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object.")
    return payload


def _normalize_invocation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: Any = payload
    if isinstance(normalized.get("instances"), list) and normalized["instances"]:
        normalized = normalized["instances"][0]

    if isinstance(normalized, dict) and "inputs" in normalized and "input" not in normalized:
        inputs = normalized.get("inputs")
        if isinstance(inputs, dict):
            normalized = inputs
        elif isinstance(inputs, str):
            normalized = {"input": inputs}

    if isinstance(normalized, str):
        normalized = {"input": normalized}

    if not isinstance(normalized, dict):
        raise HTTPException(status_code=400, detail="Unsupported invocation payload format.")
    return normalized


def _run_generation_or_503(
    run_generation: Callable[[str, int], AnswerResponse], input_text: str, max_tokens: int
) -> AnswerResponse:
    try:
        return run_generation(input_text=input_text, max_tokens=max_tokens)
    except RuntimeError as exc:
        # Surface backend initialization/runtime errors as service-unavailable
        # instead of uncaught 500 responses.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def register_local_routes(
    app: FastAPI,
    run_generation: Callable[[str, int], AnswerResponse],
    config: dict[str, Any],
) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": str(config.get("backend", "auto"))}

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest) -> AnswerResponse:
        return _run_generation_or_503(run_generation, req.input, req.max_tokens)


def register_openai_routes(
    app: FastAPI,
    run_generation: Callable[[str, int], AnswerResponse],
    config: dict[str, Any],
) -> None:
    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        model_id = str(config.get("model_name_or_path"))
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported.")
        if not req.messages:
            raise HTTPException(status_code=400, detail="`messages` cannot be empty.")

        input_text = _messages_to_input_text(req.messages)
        if not input_text:
            raise HTTPException(status_code=400, detail="No usable text found in `messages`.")

        result = _run_generation_or_503(run_generation, input_text, req.max_tokens)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_tokens = max(1, len(input_text.split()))
        completion_tokens = max(1, len(result.answer.split()))
        model_name = req.model or result.model

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported.")

        if isinstance(req.prompt, list):
            input_text = "\n".join([str(p) for p in req.prompt if str(p).strip()]).strip()
        else:
            input_text = str(req.prompt).strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="`prompt` must contain text.")

        result = _run_generation_or_503(run_generation, input_text, req.max_tokens)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        prompt_tokens = max(1, len(input_text.split()))
        completion_tokens = max(1, len(result.answer.split()))
        model_name = req.model or result.model

        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": result.answer,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def register_sagemaker_routes(
    app: FastAPI,
    run_generation: Callable[[str, int], AnswerResponse],
    config: dict[str, Any],
) -> None:
    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invocations")
    async def invocations(request: Request) -> dict[str, Any]:
        payload = _normalize_invocation_payload(await _parse_invocation_payload(request))

        input_text = payload.get("input")
        if not isinstance(input_text, str) or not input_text.strip():
            raise HTTPException(status_code=400, detail="`input` must be a non-empty string.")

        try:
            max_tokens = int(payload.get("max_tokens", 256))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="`max_tokens` must be an integer.") from exc

        if max_tokens < 32 or max_tokens > 1024:
            raise HTTPException(status_code=400, detail="`max_tokens` must be between 32 and 1024.")

        result = _run_generation_or_503(run_generation, input_text, max_tokens)
        return result.model_dump()
