# Serving / Unified Inference Guide

This document explains how to run and test `serving/vllm_server.py` after refactoring to a registry-based architecture.

## 1. Architecture

The server now uses registry classes to compose backends and API protocols.

- Backend registry: `serving/utils/backends.py`
  - `vllm` backend
  - `transformers` backend
- Route registry: `serving/utils/registries.py`
- Route adapters: `serving/utils/routes.py`
  - `local` routes (`/health`, `/answer`)
  - `openai` routes (`/v1/models`, `/v1/chat/completions`, `/v1/completions`)
  - `sagemaker` routes (`/ping`, `/invocations`)

## 2. Start the Server

Default startup:

```bash
python serving/vllm_server.py
```

Explicit startup:

```bash
python serving/vllm_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --route_sets local,openai,sagemaker
```

Optional arguments:

- `--backend`: `auto` / `vllm` / `transformers`
- `--adapter_path`: optional LoRA/PEFT adapter path
- `--route_sets`: comma-separated route sets (`local`, `openai`, `sagemaker`)

Backend behavior:

- `auto`: try `vllm` first, fallback to `transformers` if `vllm` init fails
- `vllm`: force vLLM backend (raise error if unavailable)
- `transformers`: force Hugging Face Transformers backend

## 3. API Endpoints

When `--route_sets local,openai,sagemaker` is enabled (default), these endpoints are available:

- Local:
  - `GET /health`
  - `POST /answer`
- OpenAI-compatible:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- SageMaker-compatible:
  - `GET /ping`
  - `POST /invocations`

## 4. Quick Self-Check

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Local inference:

```bash
curl -s http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Explain what RAG is in one paragraph.",
    "max_tokens": 128
  }'
```

SageMaker-style inference:

```bash
curl -s http://127.0.0.1:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is LoRA?",
    "max_tokens": 128
  }'
```

## 5. Notes

- First request may be slower because model/backend initialization is lazy.
- If you pass `--adapter_path`, ensure the adapter path is accessible at runtime.
- For SageMaker deployment, use `--port 8080` and keep route set `sagemaker` enabled.

