from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is on path for `data` package
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_training_text(example: dict[str, Any]) -> str:
    from training.prompt_templates import render
    return render(example)


def _resolve_resume_checkpoint(training_cfg: dict[str, Any], output_dir: Path) -> str | bool | None:
    """Resolve resume checkpoint from config. Supports True/'latest'/explicit path."""
    from transformers.trainer_utils import get_last_checkpoint

    resume_cfg = training_cfg.get("resume_from_checkpoint")
    if resume_cfg in (None, False):
        return None
    if resume_cfg is True or str(resume_cfg).lower() == "latest":
        last_ckpt = get_last_checkpoint(str(output_dir))
        if last_ckpt is None:
            print("[train] resume_from_checkpoint requested but no checkpoint found. Starting fresh.")
            return None
        print(f"[train] Resuming from latest checkpoint: {last_ckpt}")
        return last_ckpt

    checkpoint_path = Path(str(resume_cfg))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Configured resume checkpoint does not exist: {checkpoint_path}")
    print(f"[train] Resuming from configured checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


def _export_artifacts(
    *,
    cfg: dict[str, Any],
    trainer: Any,
    tokenizer: Any,
    output_dir: Path,
    base_model: str,
    use_cuda: bool,
) -> dict[str, Any]:
    """Export adapter weights and optional merged full model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch

    export_cfg = cfg.get("export", {})
    adapter_dir = output_dir / str(export_cfg.get("adapter_dir", "adapter"))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    export_summary: dict[str, Any] = {
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": None,
        "merged_model_exported": False,
    }

    if not bool(export_cfg.get("merge_adapter", False)):
        return export_summary

    merged_model_dir = output_dir / str(export_cfg.get("merged_model_dir", "merged_model"))
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    safe_serialization = bool(export_cfg.get("safe_serialization", True))
    merge_dtype = torch.bfloat16 if use_cuda else torch.float32

    # Merge via a fresh base model to avoid quantization-state edge cases.
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=merge_dtype,
        device_map="auto" if use_cuda else None,
    )
    peft_model = PeftModel.from_pretrained(base_for_merge, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(merged_model_dir), safe_serialization=safe_serialization)
    tokenizer.save_pretrained(str(merged_model_dir))

    export_summary["merged_model_dir"] = str(merged_model_dir)
    export_summary["merged_model_exported"] = True
    return export_summary


def run_training(cfg: dict[str, Any]) -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    import torch

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    resume_from_checkpoint = _resolve_resume_checkpoint(training_cfg, output_dir)

    from data.loaders import load_split_rows

    train_rows = load_split_rows(dataset_cfg, split="train")
    eval_rows = load_split_rows(dataset_cfg, split="eval")
    if not train_rows or not eval_rows:
        raise ValueError("Train/eval dataset cannot be empty.")

    max_seq_length = int(dataset_cfg.get("max_seq_length", 2048))
    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    requested_4bit = bool(training_cfg.get("load_in_4bit", True))
    use_4bit = requested_4bit and use_cuda
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(training_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = Dataset.from_list([{"text": _build_training_text(row)} for row in train_rows])
    eval_ds = Dataset.from_list([{"text": _build_training_text(row)} for row in eval_rows])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokens["labels"] = [ids[:] for ids in tokens["input_ids"]]
        return tokens

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        logging_strategy=str(training_cfg.get("logging_strategy", "steps")),
        save_steps=int(training_cfg.get("save_steps", 200)),
        eval_steps=int(training_cfg.get("eval_steps", 200)),
        eval_strategy=str(training_cfg.get("eval_strategy", "steps")),
        save_strategy=str(training_cfg.get("save_strategy", "steps")),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", False)),
        metric_for_best_model=str(training_cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        report_to=list(training_cfg.get("report_to", [])),
        deepspeed=training_cfg.get("deepspeed_config"),
        bf16=bool(training_cfg.get("bf16", use_cuda)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_output = trainer.evaluate()
    if "eval_loss" in eval_output:
        eval_loss = float(eval_output["eval_loss"])
        eval_output["eval_perplexity"] = math.exp(eval_loss) if eval_loss < 50 else float("inf")

    trainer.log_metrics("train", train_output.metrics)
    trainer.save_metrics("train", train_output.metrics)
    trainer.save_state()
    trainer.log_metrics("eval", eval_output)
    trainer.save_metrics("eval", eval_output)

    export_summary = _export_artifacts(
        cfg=cfg,
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        base_model=base_model,
        use_cuda=use_cuda,
    )

    metadata = {
        "project_name": cfg["project_name"],
        "base_model": cfg["base_model"],
        "status": "sft_completed",
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "global_step": train_output.global_step,
        "train_loss": train_output.training_loss,
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_output,
        "quantization_4bit": use_4bit,
        "resume_from_checkpoint": resume_from_checkpoint,
        "deepspeed_config": training_cfg.get("deepspeed_config"),
        "export": export_summary,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[train] SFT training completed.")
    print(f"[train] Metadata saved to: {output_dir / 'run_metadata.json'}")
    print(f"[train] Adapter saved to: {export_summary['adapter_dir']}")
    if export_summary["merged_model_exported"]:
        print(f"[train] Merged model saved to: {export_summary['merged_model_dir']}")


def check_quality(cfg: dict[str, Any]) -> None:
    """Run data quality checks on train and eval files from config."""
    from data.loaders import load_split_rows
    from data.quality_checks import run_quality_checks_and_print
    from data.schema import REQUIRED_KEYS

    dataset_cfg = cfg["dataset"]

    # Keep fast path for legacy file-based config.
    if "train_file" in dataset_cfg and "eval_file" in dataset_cfg:
        train_path = Path(dataset_cfg["train_file"])
        eval_path = Path(dataset_cfg["eval_file"])
        train_ok = train_path.exists() and run_quality_checks_and_print(train_path)
        if not train_path.exists():
            print(f"[check_quality] Train file not found: {train_path}")
        eval_ok = eval_path.exists() and run_quality_checks_and_print(eval_path)
        if not eval_path.exists():
            print(f"[check_quality] Eval file not found: {eval_path}")
        if train_ok and eval_ok and train_path.exists() and eval_path.exists():
            print("[check_quality] All checks passed.")
            return
        raise SystemExit(1)

    # Generic source-mode validation (Kafka and other non-file sources).
    def _validate_rows(rows: list[dict[str, Any]], split: str) -> bool:
        ok = True
        if not rows:
            print(f"[check_quality] {split}: empty dataset")
            return False
        for i, row in enumerate(rows):
            missing = [k for k in REQUIRED_KEYS if k not in row]
            if missing:
                print(f"[check_quality] {split}: row {i} missing required keys: {missing}")
                ok = False
                continue
            refs = row.get("refs")
            if not isinstance(refs, list) or len(refs) == 0 or not all(isinstance(r, str) for r in refs):
                print(f"[check_quality] {split}: row {i} has invalid refs (must be non-empty list[str])")
                ok = False
            answer = (row.get("answer") or "").strip()
            if not answer:
                print(f"[check_quality] {split}: row {i} has empty answer")
                ok = False
        print(f"[check_quality] {split}: rows={len(rows)} passed={ok}")
        return ok

    train_rows = load_split_rows(dataset_cfg, split="train")
    eval_rows = load_split_rows(dataset_cfg, split="eval")
    if _validate_rows(train_rows, "train") and _validate_rows(eval_rows, "eval"):
        print("[check_quality] All checks passed.")
        return
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT pipeline scaffold for OpenSupport-LLM.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "check_quality"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def _cleanup_distributed() -> None:
    """Best-effort distributed cleanup to avoid NCCL shutdown warnings."""
    try:
        import torch.distributed as dist
    except Exception:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Do not fail process shutdown on cleanup path.
        pass


def main() -> None:
    try:
        args = parse_args()
        cfg = load_config(args.config)

        if args.mode == "train":
            run_training(cfg)
        elif args.mode == "check_quality":
            check_quality(cfg)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()


