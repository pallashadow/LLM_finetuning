# Serving / SageMaker Deployment Guide

This project now uses a single unified server: `serving/vllm_server.py`.

Run commands in WSL (bash). Example entry command from Windows host:

```bash
wsl -u pallas -- bash -lc "cd /mnt/d/PROJECT/LLM_finetuning && lic python serving/vllm_server.py --port 8080 --route_sets local,openai,sagemaker"
```

## 1. Start Command (local and container)

For SageMaker-compatible startup, run:

```bash
python serving/vllm_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --backend auto \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --route_sets local,openai,sagemaker
```

Optional arguments:

- `--backend`: `auto` / `vllm` / `transformers`
- `--adapter_path`: optional LoRA/PEFT adapter path
- `--route_sets`: comma-separated route sets

## 2. SageMaker Endpoint Contract

When `sagemaker` route set is enabled, the server exposes:

- `GET /ping`: health check
- `POST /invocations`: inference entry point

## 3. Request Format (`/invocations`)

Recommended minimal payload:

```json
{
  "input": "What is LoRA?",
  "max_tokens": 256
}
```

Compatible wrapped payload formats:

1. `instances`:

```json
{
  "instances": [
    {
      "input": "What is LoRA?"
    }
  ]
}
```

2. `inputs`:

```json
{
  "inputs": "What is LoRA?"
}
```

Response example:

```json
{
  "answer": "LoRA is a parameter-efficient fine-tuning method ...",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "latency_ms": 123.45
}
```

## 4. Quick Self-Check

Health check:

```bash
curl -s http://127.0.0.1:8080/ping
```

Inference request:

```bash
curl -s http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is RAG?",
    "max_tokens": 128
  }'
```

## 5. Key SageMaker Configuration Points

1. The inference container must listen on `0.0.0.0:8080`.
2. Use `/ping` as the health check path.
3. Use `/invocations` as the online inference path.
4. Keep `sagemaker` in `--route_sets`.

