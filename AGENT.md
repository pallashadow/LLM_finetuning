# AGENT Guide

This document is for coding agents working on this repository.

## Project Name

OpenSupport-LLM: Low-Latency RAG Answer System with Small Open-Source Models

## Mission

Build a reproducible, benchmarkable **RAG answer** system: the model answers only from provided references, cites as [1][2], respects entity matching (no wrong names from refs), and uses a **minimal-token prompt** defined in `training/prompt_template.yaml`.

Primary comparison target:

- Closed-source GPT-4 baseline vs fine-tuned 7B/8B open-source model variants

## Prompt Template (Single Source of Truth)

- **File**: `training/prompt_template.yaml`
- **Purpose**: Generic RAG prompt; minimize tokens. Used for both training data rendering and inference prompt construction.
- **Placeholders**: `{question}`, `{query_context_txt}` (prior Q&A history), `{search_results_txt}` (reference texts).
- **Body keys**: `prompt_before`, `topic_line_template`, `history_line_template`, `refs_line_template`, `prompt_after`.

All prompt text and layout must stay in this YAML; avoid duplicating instructions in code.

## Core Workstreams

### 1) Instruction Tuning Pipeline

Deliver:

- Dataset pipeline compatible with the **RAG template** in `training/prompt_template.yaml` (question, history, refs → grounded answer with citations).
- SFT pipeline (LoRA / QLoRA).
- **Single source of truth**: training and inference prompt rendering driven by `prompt_template.yaml`.
- Multi-turn support via `history_line_template` and `{query_context_txt}`.

Stack:

- Qwen3-8B (or equivalent 7B/8B base)
- QLoRA + PEFT, Hugging Face Trainer, DeepSpeed

Outputs:

- Fine-tuned checkpoint, end-to-end training script, clear dataset schema.

### 2) Evaluation Framework

Must include:

- Response quality and faithfulness to references.
- Citation presence and correctness.
- Entity matching (no wrong names from refs).
- Hallucination rate.
- Instruction-following (no conclusion phrase, answer only from refs).

Optional: LLM-as-a-judge, GPT-4 evaluator.

## Detailed TODO List

### A. Data and Prompting

- [x] Define JSONL schema for training/eval (compatible with RAG: question, history, refs, answer).
- [x] Build synthetic or converted dataset pipeline.
- [x] Create generic RAG prompt in `training/prompt_template.yaml` (minimal tokens).
- [x] Add data quality checks (malformed rows, empty answers, required fields).

### B. Training (SFT / LoRA / QLoRA)

- [x] Real HF Trainer + PEFT for RAG SFT in `training/sft_pipeline.py`.
- [x] RAG config wiring from `configs/train_config.yaml` (RAG train/eval JSONL).
- [x] Keep training as minimum executable baseline.
- [x] Checkpoint save/load and resume.
- [x] Validation loop and metric logging.
- [x] Export adapter weights and merged model option.

### C. Serving and Deployment

- [x] Replace stub in `serving/vllm_server.py` with real model inference.
- [ ] Build inference prompt from `prompt_template.yaml` (question, history, refs).
- [x] Request validation (max length, payload size).
- [ ] SageMaker deployment artifacts.
- [ ] Endpoint metrics (latency, error rate, throughput).

### D. Benchmarking

- [x] Expand `evaluation/latency_test.py` (warmup, concurrency).
- [ ] Cost estimation for local and API models.
- [x] Connect to `report/generated/benchmark_results.csv`.
- [ ] Multi-model benchmark automation and markdown tables.

### E. Evaluation and Reporting

- [ ] Quality evaluation (faithfulness, citations, entity matching).
- [ ] Hallucination detection and error categorization.
- [ ] Instruction-following checks aligned with `prompt_template.yaml`.
- [ ] Optional LLM-judge; update `report/benchmark_results.md`.

### F. Engineering Hygiene

- [ ] Unit tests (dataset, benchmark, latency stats).
- [ ] Pre-commit (format, lint, static checks).
- [ ] CI smoke tests.
- [ ] Version configs and experiment IDs; troubleshooting section.

## Project Status (Updated)

Last updated: 2026-03-02

Completed:

- Repository skeleton (data, training, serving, evaluation, config, report).
- `training/sft_pipeline.py`: executable SFT with LoRA/QLoRA + PEFT, checkpoint resume support, eval metrics, and optional merged-model export.
- Adapter and run metadata output (`run_metadata.json`) with train/eval metrics.
- Generic RAG prompt in `training/prompt_template.yaml` (minimal tokens; question, history, refs).
- `serving/vllm_server.py`: unified FastAPI inference server with backend registry (`vllm`/`transformers`) and route sets (`local`, `openai`, `sagemaker`).
- Request validation and payload normalization are in place for local/OpenAI/SageMaker-style endpoints.
- `evaluation/latency_test.py`: warmup rounds, concurrency test mode, JSON output, and benchmark CSV appending.

In progress:

- Wiring inference prompt construction from `prompt_template.yaml` in serving path (question, history, refs).
- Cost estimation and multi-model benchmark automation/reporting.

Not started / missing:

- Full production training hardening (larger-scale runs, richer eval, experiment tracking).
- Production vLLM deployment and capacity tuning.
- SageMaker deployment pipeline.
- Full evaluation suite (quality, hallucination, citations, entity matching, retrieval metrics).

## Project Architecture (Logical)

```text
data/
training/
   sft_pipeline.py
   prompt_template.yaml   # single source of truth for RAG prompt
   prompt_templates.py    # load YAML, render (training or RAG)
models/
   lora_checkpoint/
serving/
   vllm_server.py
   utils/
      backends.py
      registries.py
      routes.py
evaluation/
   benchmark.py
   latency_test.py
report/
   benchmark_results.md
```
