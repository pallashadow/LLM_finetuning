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

1. Create environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Build synthetic dataset (if using pipeline dataset builder):

   ```bash
   python training/sft_pipeline.py --mode build_dataset --config configs/train_config.yaml
   ```

3. Run SFT pipeline:

   ```bash
   python training/sft_pipeline.py --mode train --config configs/train_config.yaml
   ```

4. Start local inference service (auto-select `vLLM`, fallback to `Transformers`):

   ```bash
   python serving/vllm_server.py --host 0.0.0.0 --port 8000 --backend auto --model_name_or_path Qwen/Qwen2.5-7B-Instruct
   ```

5. Run latency benchmark:

   ```bash
   python evaluation/latency_test.py --endpoint http://127.0.0.1:8000/answer --rounds 20 --warmup_rounds 3 --concurrency 1 --output_json report/generated/latest_latency.json --benchmark_csv report/generated/benchmark_results.csv --model local-open-model --cost_per_1k_queries_usd 0.02
   ```

## Current Status

- Repository scaffold is complete for data, training, serving, evaluation, config, and report modules.
- Training: LoRA/PEFT SFT, adapter export, metadata output. Prompt rendering can be wired to `prompt_template.yaml` (RAG: question, history, refs).
- Inference and benchmark scripts are runnable; production hardening and deployment workflows are pending.
- SageMaker deployment, full evaluation framework, and CI/test coverage are not complete yet.

## Detailed TODO List

- [ ] Keep training as a stable minimum executable baseline.
- [ ] Add training resume support and explicit checkpoint selection.
- [ ] Add training smoke test for one short end-to-end run.
- [ ] Replace serving placeholder with real vLLM-backed RAG answer and adapter loading; build inference prompt from `prompt_template.yaml`.
- [ ] Add SageMaker deployment assets (container, endpoint config, deployment script).
- [ ] Expand latency benchmark (warmup, concurrency, reproducible metadata).
- [ ] Add cost tracking and benchmark comparison across GPT-4 and open-model variants.
- [ ] Build quality evaluation (faithfulness to refs, citation presence, entity matching, hallucination rate).
- [ ] Add tests and CI for dataset building, benchmark utilities, and inference API smoke checks.
