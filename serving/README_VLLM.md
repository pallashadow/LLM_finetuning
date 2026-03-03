# Serving / Unified Inference Guide

This document explains how to run and test `serving/vllm_server.py` after refactoring to a registry-based architecture.

Run commands in WSL (bash). Example entry command from Windows host:

```bash
wsl -u pallas -- bash -lc "cd /mnt/d/PROJECT/LLM_finetuning && lic python -m serving.vllm_server"
```

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
python -m serving.vllm_server
```

Explicit startup:

```bash
python -m serving.vllm_server \
  --host 0.0.0.0 \
  --port 8000 \
  --backend auto \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 4096 \
  --route_sets local,openai,sagemaker
```

Optional arguments:

- `--backend`: `auto` / `vllm` / `transformers`
- `--adapter_path`: optional LoRA/PEFT adapter path
- `--max-model-len`: vLLM context length upper bound (default: `4096`)
- `--route_sets`: comma-separated route sets (`local`, `openai`, `sagemaker`)

Backend behavior:

- `auto`: prefer `vllm` backend, then fallback to `transformers` if vLLM initialization fails
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

Local inference (RAG prompt builder client):

```bash
python -m serving.client.client \
  --question "Who is Satya Nadella?" \
  --query-context-txt "User previously asked about Microsoft leadership." \
  --search-results-txt "[1] Satya Nadella is the CEO of Microsoft." \
  --model-name vllm
```

This client builds the inference prompt from `prompts/prompt_rag.yaml` using
`question`, `query_context_txt`, and `search_results_txt`, then calls LiteLLM
via `serving/client/litellm_api.py`.

Set local vLLM endpoint/model with `VLLM_API_BASE`, `VLLM_MODEL`, `VLLM_API_KEY`.
The client auto-loads these values from project-root `.env` (existing process env still has higher priority).

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
- vLLM memory behavior:
  - Default requested `gpu_memory_utilization` is `0.8`.
  - Default `max_model_len` is `4096` (reduced for 8GB-class GPUs).
  - Runtime auto-clamps to current free GPU memory to avoid startup failure.
  - If KV cache is still insufficient, runtime auto-retries with smaller `max_model_len`.
  - You can override utilization via env var: `VLLM_GPU_MEMORY_UTILIZATION` (for example `0.75`).
  - You can override max length via env var: `VLLM_MAX_MODEL_LEN` (used when CLI value is not provided).
  - To re-enable torch.compile path, set `VLLM_ENFORCE_EAGER=0`.

