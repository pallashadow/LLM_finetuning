## Training & Quantization Pipelines

This directory contains the training and quantization scripts for **OpenSupport-LLM**.  
All pipelines are driven by YAML configs in `configs/` and share the same dataset abstraction in `data/`.

### Files in this folder

- `sft_pipeline.py`: Supervised fine-tuning (SFT) with LoRA/QLoRA.
- `prompt_templates.py`: Loads `prompt_template.yaml` and renders RAG prompts for training/inference.
- `quantize_pipeline.py`: Post-training quantization pipeline (AutoAWQ int4) for serving-optimized models.

---

### 1. SFT Pipeline (`sft_pipeline.py`)

**Purpose**: Train a LoRA/QLoRA adapter on RAG-style data using the shared prompt template.

**Key features**

- Uses `training/prompt_template.yaml` as the single source of truth for prompt layout.
- Supports both simple JSONL files and advanced data sources via `data/loaders.py`.
- Runs data quality checks before training.
- Exports LoRA adapter and (optionally) a merged full model.

**Core modes**

- `train`: run SFT training.
- `check_quality`: validate dataset schema and basic data quality.

**Typical config fields (YAML)**

- `project_name`: experiment name.
- `base_model`: HF model ID (e.g. `Qwen/Qwen3-4B-Instruct-2507`).
- `output_dir`: where checkpoints and metadata are written.
- `dataset`: data source + pipeline definition (see `README.md` and `data/loaders.py`).
- `training`: Trainer hyperparameters (lr, steps, batch size, DeepSpeed, 4-bit, etc.).
- `export`: adapter / merged-model export options.
- `lora`: LoRA rank, alpha, target modules, etc.

**Example commands**

```bash
python training/sft_pipeline.py --mode check_quality --config configs/train_config_qwen3_4b_qlora_3070.yaml

python training/sft_pipeline.py --mode train --config configs/train_config_qwen3_4b_qlora_3070.yaml
```

Outputs:

- LoRA adapter weights and tokenizer (under `<output_dir>/adapter` by default).
- Optional merged full model (if `export.merge_adapter: true`).
- `run_metadata.json` with training/eval metrics and export summary.

---

### 2. Prompt Rendering (`prompt_templates.py`)

**Purpose**: Keep all prompt text in `prompt_template.yaml` and render it consistently for:

- Training examples (`render`)
- Inference prompts (`render_inference`)

**Concepts**

- Body keys: `prompt_before`, `topic_line_template`, `history_line_template`, `refs_line_template`, `prompt_after`.
- Placeholders: `{question}`, `{query_context_txt}`, `{search_results_txt}`.
- The same template is used to generate training text (prompt + answer) and inference-only prompts.

You normally don’t call this module directly; it is used by `sft_pipeline.py` and serving code.

---

### 3. Quantization Pipeline (`quantize_pipeline.py`)

**Purpose**: Turn a trained model (or merged model) into an **int4 AutoAWQ** variant for low-latency serving,  
while reusing the same dataset config to build a calibration set.

**High-level flow**

1. Load a training config YAML (same style as SFT).
2. Resolve which model to quantize:
   - Merged full model directory (preferred, if present).
   - Base model + LoRA adapter (merged on the fly into a temp dir).
   - Base model only (fallback).
3. Sample calibration texts from the dataset via `data/loaders.py`.
4. Run AutoAWQ int4 quantization and export artifacts.
5. Write `quantize_metadata.json` next to the quantized model.

**CLI**

- `--config`: path to YAML config (reuses `base_model`, `dataset`, and an optional `quantization` block).
- `--mode`: currently only `quantize`.
- `--dry_run`: resolve model + calibration sampling, but skip the actual quantization call.

**Config: `quantization` block (optional but recommended)**

```yaml
quantization:
  backend: autoawq          # currently only AutoAWQ is implemented
  output_dir: outputs/qwen3_4b_qlora_3070/quantized_awq

  model:
    # Optional overrides; defaults come from training config export:
    # base_model: Qwen/Qwen3-4B-Instruct-2507
    # merged_model_path: outputs/qwen3_4b_qlora_3070/merged_model
    # adapter_path: outputs/qwen3_4b_qlora_3070/adapter
    input_variant: auto     # auto | merged | base_adapter | base

  params:
    w_bit: 4
    q_group_size: 128
    zero_point: true
    version: GEMM

  calibration:
    split: train
    max_samples: 128
    max_tokens: 1024
    seed: 42
```

If the `quantization` block is omitted, the pipeline will:

- Use `base_model` from the root config.
- Look for a merged model under `<output_dir>/merged_model`.
- Fall back to base + adapter or base-only according to availability.
- Write quantized weights under `<output_dir>/quantized_awq`.

**Example commands**

```bash
# Dry-run: check plan and calibration sampling only
python training/quantize_pipeline.py \
  --config configs/train_config_qwen3_4b_qlora_3070.yaml \
  --dry_run

# Full quantization run
python training/quantize_pipeline.py \
  --config configs/train_config_qwen3_4b_qlora_3070.yaml \
  --mode quantize
```

Outputs:

- Quantized AWQ weights (e.g. `outputs/.../quantized_awq`).
- Tokenizer files for serving.
- `quantize_metadata.json` with the plan, calibration details, and status.

---

### 4. Recommended workflow

1. **Validate data**
   - `python training/sft_pipeline.py --mode check_quality --config <train_config.yaml>`
2. **Run SFT training**
   - `python training/sft_pipeline.py --mode train --config <train_config.yaml>`
3. **(Optional) Export merged full model**
   - Enable `export.merge_adapter: true` in the training config if you want an explicit merged directory.
4. **Quantize for serving**
   - `python training/quantize_pipeline.py --config <train_config.yaml> --mode quantize`
5. **Serve quantized model**
   - Point your serving backend (e.g. `vLLM`/`transformers`) at the quantized model directory.

