# OpenSupport-LLM

Low-latency RAG answer system: grounded Q&A from references using small open-source models.

## Project Goals

- Reproducible instruction tuning for **RAG-style answers**: answer only from references, cite as [1][2], entity matching, minimal prompt tokens.
- Compare closed-source GPT-4 baseline vs fine-tuned 7B/8B open model variants.
- Evaluate latency, cost, and quality with a shared benchmark protocol.

## Prompt Template

All prompts (training and inference) are driven by **`training/prompt_template.yaml`** (single source of truth). Generic RAG template with placeholders:

- `{question}` — user question
- `{query_context_txt}` — prior Q&A history (optional)
- `{search_results_txt}` — reference texts

Template is kept short to minimize token usage.

## Repository Layout

```text
data/
training/
models/
serving/
evaluation/
configs/
report/
```

## Quick Start

1. Create environment and install dependencies (WSL / bash, from project root):

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Prepare dataset files (or configure a data source):

   ```bash
   .venv/bin/python training/sft_pipeline.py --mode check_quality --config configs/train_config.yaml
   ```

3. Run SFT pipeline:

   ```bash
   .venv/bin/python training/sft_pipeline.py --mode train --config configs/train_config.yaml
   ```

4. Start local inference service (auto-select `vLLM`, fallback to `Transformers`):

   ```bash
   .venv/bin/python serving/vllm_server.py --host 0.0.0.0 --port 8000 --backend auto --model_name_or_path Qwen/Qwen3-4B-Instruct-2507
   ```

   For more serving options and OpenAI/SageMaker-compatible routes, see `serving/README_VLLM.md`.

5. Run latency benchmark:

   ```bash
   .venv/bin/python evaluation/latency_test.py --endpoint http://127.0.0.1:8000/answer --rounds 20 --warmup_rounds 3 --concurrency 1 --output_json report/generated/latest_latency.json --benchmark_csv report/generated/benchmark_results.csv --model local-open-model --cost_per_1k_queries_usd 0.02
   ```

## Current Status

- Repository scaffold is complete for data, training, serving, evaluation, config, and report modules.
- Training: LoRA/PEFT SFT, adapter export, metadata output. Prompt rendering can be wired to `prompt_template.yaml` (RAG: question, history, refs).
- Inference and benchmark scripts are runnable; production hardening and deployment workflows are pending.
- SageMaker deployment, full evaluation framework, and CI/test coverage are not complete yet.

## Dataset Loader Abstraction

`training/sft_pipeline.py` supports source abstraction via `data/loaders.py`.

- Backward-compatible file config:
  - `dataset.train_file`
  - `dataset.eval_file`
- Source-based config:
  - `dataset.train_source`
  - `dataset.eval_source`
- Optional transform pipeline:
  - `dataset.pipeline`
  - `dataset.train_pipeline`
  - `dataset.eval_pipeline`

Example (single JSONL file source):

```yaml
dataset:
  train_source:
    type: jsonl_file
    config:
      path: data/synthetic_rag_train.jsonl
  eval_source:
    type: jsonl_file
    config:
      path: data/synthetic_rag_eval.jsonl
  pipeline:
    - name: strip_fields
      config:
        fields: [question, answer]
    - name: drop_empty_answer
```

Example (Kafka source):

```yaml
dataset:
  train_source:
    type: kafka
    config:
      topic: rag-train
      bootstrap_servers: 127.0.0.1:9092
      group_id: opensupport-train
      auto_offset_reset: earliest
      max_messages: 10000
  eval_source:
    type: jsonl_file
    config:
      path: data/synthetic_rag_eval.jsonl
```

## Detailed TODO / Roadmap

High-level roadmap; see `AGENT.md` for the up-to-date, engineering-focused checklist.

- [ ] Production-ready training pipeline (resume, richer eval, experiment tracking).
- [ ] vLLM-backed serving, adapter loading, and SageMaker deployment assets.
- [ ] Expanded latency and cost benchmarking across GPT-4 and open-model variants.
- [ ] Quality evaluation suite (faithfulness, citations, entity matching, hallucination rate).
- [ ] Tests and CI for dataset building, benchmark utilities, and inference API smoke checks.
