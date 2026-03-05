from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is on path for `data` package imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class QuantizationPlan:
    base_model: str
    input_variant: str
    model_path_for_quant: Path
    tokenizer_path_for_quant: Path
    adapter_path: Path | None
    output_dir: Path
    temp_merged_dir: Path | None
    quant_backend: str
    quant_config: dict[str, Any]
    calibration_split: str
    calibration_max_samples: int
    calibration_seed: int
    calibration_max_tokens: int


def _resolve_output_dir(cfg: dict[str, Any], quant_cfg: dict[str, Any]) -> Path:
    out_cfg = quant_cfg.get("output_dir")
    if out_cfg:
        return Path(str(out_cfg))
    train_output = Path(str(cfg.get("output_dir", "outputs/model")))
    return train_output / "quantized_awq"


def _resolve_default_merged_model_dir(cfg: dict[str, Any]) -> Path:
    train_output = Path(str(cfg.get("output_dir", "outputs/model")))
    export_cfg = cfg.get("export", {}) if isinstance(cfg.get("export"), dict) else {}
    merged_name = str(export_cfg.get("merged_model_dir", "merged_model"))
    return train_output / merged_name


def _resolve_quantization_plan(cfg: dict[str, Any]) -> QuantizationPlan:
    quant_cfg = cfg.get("quantization", {}) if isinstance(cfg.get("quantization"), dict) else {}
    model_cfg = quant_cfg.get("model", {}) if isinstance(quant_cfg.get("model"), dict) else {}
    calib_cfg = quant_cfg.get("calibration", {}) if isinstance(quant_cfg.get("calibration"), dict) else {}

    base_model = str(model_cfg.get("base_model") or cfg.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("Missing base model. Set `base_model` or `quantization.model.base_model` in config.")

    merged_model_path = model_cfg.get("merged_model_path")
    adapter_path_cfg = model_cfg.get("adapter_path")
    adapter_path = Path(str(adapter_path_cfg)) if adapter_path_cfg else None

    input_variant = str(model_cfg.get("input_variant", "auto")).strip().lower()
    if input_variant not in {"auto", "merged", "base_adapter", "base"}:
        raise ValueError("`quantization.model.input_variant` must be one of: auto, merged, base_adapter, base")

    default_merged = _resolve_default_merged_model_dir(cfg)
    merged_candidate = Path(str(merged_model_path)) if merged_model_path else default_merged
    merged_exists = merged_candidate.exists()

    if input_variant == "auto":
        if merged_exists:
            resolved_variant = "merged"
        elif adapter_path is not None:
            resolved_variant = "base_adapter"
        else:
            resolved_variant = "base"
    else:
        resolved_variant = input_variant

    if resolved_variant == "merged":
        if not merged_exists:
            raise FileNotFoundError(
                f"Merged model not found: {merged_candidate}. "
                "Set `quantization.model.merged_model_path` or export merged model first."
            )
        model_path_for_quant = merged_candidate
        tokenizer_path_for_quant = merged_candidate
        temp_merged_dir = None
    elif resolved_variant == "base_adapter":
        if adapter_path is None:
            raise ValueError("`base_adapter` variant requires `quantization.model.adapter_path`.")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        temp_merged_dir = _resolve_output_dir(cfg, quant_cfg) / "_tmp_merged_for_quant"
        model_path_for_quant = temp_merged_dir
        tokenizer_path_for_quant = temp_merged_dir
    else:
        temp_merged_dir = None
        model_path_for_quant = Path(base_model)
        tokenizer_path_for_quant = Path(base_model)

    quant_backend = str(quant_cfg.get("backend", "autoawq")).strip().lower()
    if quant_backend != "autoawq":
        raise ValueError("Only `autoawq` backend is currently implemented.")

    quant_params = quant_cfg.get("params", {}) if isinstance(quant_cfg.get("params"), dict) else {}
    quant_config: dict[str, Any] = {
        "w_bit": int(quant_params.get("w_bit", 4)),
        "q_group_size": int(quant_params.get("q_group_size", 128)),
        "zero_point": bool(quant_params.get("zero_point", True)),
        "version": str(quant_params.get("version", "GEMM")),
    }

    return QuantizationPlan(
        base_model=base_model,
        input_variant=resolved_variant,
        model_path_for_quant=model_path_for_quant,
        tokenizer_path_for_quant=tokenizer_path_for_quant,
        adapter_path=adapter_path,
        output_dir=_resolve_output_dir(cfg, quant_cfg),
        temp_merged_dir=temp_merged_dir,
        quant_backend=quant_backend,
        quant_config=quant_config,
        calibration_split=str(calib_cfg.get("split", "train")),
        calibration_max_samples=int(calib_cfg.get("max_samples", 128)),
        calibration_seed=int(calib_cfg.get("seed", 42)),
        calibration_max_tokens=int(calib_cfg.get("max_tokens", 1024)),
    )


def _render_calibration_text(row: dict[str, Any]) -> str:
    from training.prompt_templates import render_inference

    try:
        return render_inference(row)
    except Exception:
        question = str(row.get("question", "")).strip()
        refs = row.get("refs") or []
        refs_txt = "\n".join(r for r in refs if isinstance(r, str))
        return "\n\n".join([p for p in [question, refs_txt] if p]).strip()


def _sample_calibration_texts(
    dataset_cfg: dict[str, Any],
    split: str,
    max_samples: int,
    seed: int,
    tokenizer: Any,
    max_tokens: int,
) -> list[str]:
    from data.loaders import load_split_rows

    rows = load_split_rows(dataset_cfg, split=split)
    if not rows:
        raise ValueError(f"Calibration split '{split}' is empty.")
    rng = random.Random(seed)
    picked_rows = rows if len(rows) <= max_samples else rng.sample(rows, k=max_samples)

    texts: list[str] = []
    for row in picked_rows:
        txt = _render_calibration_text(row).strip()
        if not txt:
            continue
        input_ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(input_ids) > max_tokens:
            txt = tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True)
        if txt:
            texts.append(txt)
    if not texts:
        raise ValueError("No valid calibration text generated from dataset rows.")
    return texts


def _merge_base_and_adapter(plan: QuantizationPlan) -> None:
    if plan.input_variant != "base_adapter":
        return
    if plan.temp_merged_dir is None or plan.adapter_path is None:
        raise ValueError("Internal error: missing temp merge path or adapter path.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if plan.temp_merged_dir.exists():
        shutil.rmtree(plan.temp_merged_dir)
    plan.temp_merged_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        plan.base_model,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    peft_model = PeftModel.from_pretrained(model, str(plan.adapter_path))
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(plan.temp_merged_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(plan.base_model, use_fast=True)
    tokenizer.save_pretrained(str(plan.temp_merged_dir))


def _cleanup_temp_merge(plan: QuantizationPlan) -> None:
    if plan.temp_merged_dir is None:
        return
    if plan.temp_merged_dir.exists():
        shutil.rmtree(plan.temp_merged_dir)


def _autoawq_quantize(
    model_path: Path,
    output_dir: Path,
    quant_config: dict[str, Any],
    calibration_texts: list[str],
) -> None:
    from transformers import AutoTokenizer

    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise ImportError(
            "AutoAWQ is not installed. Install it in your environment first, e.g. `pip install autoawq`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoAWQForCausalLM.from_pretrained(str(model_path))

    quantize_attempts = [
        {"quant_config": quant_config, "calib_data": calibration_texts},
        {"quant_config": quant_config},
    ]
    last_error: Exception | None = None
    for kwargs in quantize_attempts:
        try:
            model.quantize(tokenizer, **kwargs)
            last_error = None
            break
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def run_quantization(cfg: dict[str, Any], dry_run: bool = False) -> None:
    from transformers import AutoTokenizer

    plan = _resolve_quantization_plan(cfg)
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    if plan.input_variant == "base_adapter":
        print("[quantize] Merging base model and adapter before quantization...")
        _merge_base_and_adapter(plan)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(plan.tokenizer_path_for_quant), use_fast=True)
        dataset_cfg = cfg.get("dataset")
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Missing `dataset` config used for calibration sampling.")
        calibration_texts = _sample_calibration_texts(
            dataset_cfg=dataset_cfg,
            split=plan.calibration_split,
            max_samples=plan.calibration_max_samples,
            seed=plan.calibration_seed,
            tokenizer=tokenizer,
            max_tokens=plan.calibration_max_tokens,
        )
        print(
            f"[quantize] Calibration data ready: split={plan.calibration_split}, "
            f"samples={len(calibration_texts)}, max_tokens={plan.calibration_max_tokens}"
        )

        if dry_run:
            print("[quantize] Dry-run enabled; skip quantization execution.")
            status = "dry_run"
        else:
            if plan.quant_backend == "autoawq":
                _autoawq_quantize(
                    model_path=plan.model_path_for_quant,
                    output_dir=plan.output_dir,
                    quant_config=plan.quant_config,
                    calibration_texts=calibration_texts,
                )
            else:
                raise ValueError(f"Unsupported quantization backend: {plan.quant_backend}")
            status = "quantization_completed"

        metadata = {
            "project_name": cfg.get("project_name"),
            "status": status,
            "quant_backend": plan.quant_backend,
            "input_variant": plan.input_variant,
            "base_model": plan.base_model,
            "model_path_for_quant": str(plan.model_path_for_quant),
            "adapter_path": str(plan.adapter_path) if plan.adapter_path else None,
            "output_dir": str(plan.output_dir),
            "quant_config": plan.quant_config,
            "calibration": {
                "split": plan.calibration_split,
                "sample_count": len(calibration_texts),
                "max_tokens": plan.calibration_max_tokens,
                "seed": plan.calibration_seed,
            },
        }
        with open(plan.output_dir / "quantize_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[quantize] Status: {status}")
        print(f"[quantize] Artifacts dir: {plan.output_dir}")
        print(f"[quantize] Metadata: {plan.output_dir / 'quantize_metadata.json'}")
    finally:
        _cleanup_temp_merge(plan)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization pipeline for OpenSupport-LLM.")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to YAML config. Reads `quantization` section plus base/dataset defaults.",
    )
    parser.add_argument(
        "--mode",
        default="quantize",
        choices=["quantize"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve model path and calibration sampling without running quantization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode == "quantize":
        run_quantization(cfg, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
