"""
Extensible dataset loading abstractions for training pipelines.

- Source registry: plug in JSONL files, Kafka streams, and other data backends.
- Pipeline registry: row-level transforms/filters shared across sources.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

Row = dict[str, Any]
SourceFactory = Callable[[dict[str, Any]], Iterable[Row]]
PipelineStep = Callable[[Row, dict[str, Any]], Row | None]


class LoaderRegistry:
    """Registry container for data sources and pipeline steps."""

    def __init__(self) -> None:
        self.source_registry: dict[str, SourceFactory] = {}
        self.pipeline_registry: dict[str, PipelineStep] = {}

    def register_source(self, name: str, factory: SourceFactory) -> None:
        """Register a dataset source factory by name."""
        self.source_registry[name] = factory

    def register_pipeline_step(self, name: str, step: PipelineStep) -> None:
        """Register a row-level pipeline step by name."""
        self.pipeline_registry[name] = step

    def source(self, name: str) -> Callable[[SourceFactory], SourceFactory]:
        """Return a decorator that registers a source factory."""

        def decorator(factory: SourceFactory) -> SourceFactory:
            self.register_source(name, factory)
            return factory

        return decorator

    def pipeline_step(self, name: str) -> Callable[[PipelineStep], PipelineStep]:
        """Return a decorator that registers a pipeline step."""

        def decorator(step: PipelineStep) -> PipelineStep:
            self.register_pipeline_step(name, step)
            return step

        return decorator


REGISTRY = LoaderRegistry()
SOURCE_REGISTRY = REGISTRY.source_registry
PIPELINE_REGISTRY = REGISTRY.pipeline_registry


def register_source(name: str, factory: SourceFactory) -> None:
    """Register a dataset source factory by name.

    Args:
        name: Source type key used in dataset config, such as ``jsonl_file``.
        factory: Callable that accepts a source config and yields row dicts.
    """
    REGISTRY.register_source(name, factory)


def register_pipeline_step(name: str, step: PipelineStep) -> None:
    """Register a row-level pipeline step by name.

    Args:
        name: Pipeline step name used in pipeline specs.
        step: Callable that transforms one row or returns ``None`` to drop it.
    """
    REGISTRY.register_pipeline_step(name, step)


@REGISTRY.source("jsonl_file")
def _jsonl_file_source(config: dict[str, Any]) -> Iterable[Row]:
    """Yield rows from a JSONL file.

    Args:
        config: Source config that must include ``path``.

    Yields:
        Parsed JSON object rows as dictionaries.

    Raises:
        FileNotFoundError: If the configured file does not exist.
        ValueError: If any JSONL line is not a JSON object.
    """
    path = Path(str(config.get("path", "")))
    if not path.exists():
        raise FileNotFoundError(f"JSONL source file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Each JSONL row must be an object. Found: {type(obj).__name__}")
            yield obj


@REGISTRY.source("kafka")
def _kafka_source(config: dict[str, Any]) -> Iterable[Row]:
    """
    Read JSON messages from Kafka topic.
    Requires `kafka-python` package when used.

    Args:
        config: Kafka settings. Typical keys include ``topic``,
            ``bootstrap_servers``, ``group_id``, ``auto_offset_reset``,
            and optional ``max_messages``.

    Yields:
        Decoded JSON message values when they are dict objects.

    Raises:
        ImportError: If ``kafka-python`` is not installed.
        ValueError: If ``topic`` is empty.
    """
    try:
        from kafka import KafkaConsumer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Kafka source requires kafka-python. Install with: pip install kafka-python"
        ) from exc

    topic = str(config.get("topic", "")).strip()
    if not topic:
        raise ValueError("Kafka source requires non-empty 'topic'.")
    bootstrap_servers = config.get("bootstrap_servers", "127.0.0.1:9092")
    group_id = config.get("group_id")
    auto_offset_reset = str(config.get("auto_offset_reset", "earliest"))
    max_messages = config.get("max_messages")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    try:
        count = 0
        for msg in consumer:
            value = msg.value
            if not isinstance(value, dict):
                continue
            yield value
            count += 1
            if isinstance(max_messages, int) and max_messages > 0 and count >= max_messages:
                break
    finally:
        consumer.close()


@REGISTRY.pipeline_step("identity")
def _identity(row: Row, config: dict[str, Any]) -> Row:
    """Return the input row unchanged."""
    _ = config
    return row


@REGISTRY.pipeline_step("strip_fields")
def _strip_fields(row: Row, config: dict[str, Any]) -> Row:
    """Strip leading/trailing spaces for configured string fields.

    Args:
        row: Input row to mutate in place.
        config: Step config with ``fields`` list.

    Returns:
        The same row instance after in-place normalization.
    """
    fields = config.get("fields", [])
    if not isinstance(fields, list):
        return row
    for key in fields:
        if key in row and isinstance(row[key], str):
            row[key] = row[key].strip()
    return row


@REGISTRY.pipeline_step("drop_empty_answer")
def _drop_empty_answer(row: Row, config: dict[str, Any]) -> Row | None:
    """Drop rows with empty ``answer`` field after trimming spaces.

    Args:
        row: Input row.
        config: Unused step config, kept for uniform step signature.

    Returns:
        Original row if ``answer`` is non-empty, otherwise ``None``.
    """
    _ = config
    answer = (row.get("answer") or "").strip()
    return row if answer else None


def _resolve_source_spec(dataset_cfg: dict[str, Any], split: str) -> dict[str, Any]:
    """Resolve source config for a split, including legacy fallback.

    The preferred format is ``{split}_source``:
    ``{"type": "...", "config": {...}}``.
    For backward compatibility, ``{split}_file`` is mapped to a
    ``jsonl_file`` source automatically.

    Args:
        dataset_cfg: Dataset-level config dictionary.
        split: Data split name, such as ``train`` or ``eval``.

    Returns:
        A normalized source spec dictionary.

    Raises:
        ValueError: If ``{split}_source`` exists but is not a dict.
        KeyError: If neither ``{split}_source`` nor ``{split}_file`` exists.
    """
    source_key = f"{split}_source"
    legacy_file_key = f"{split}_file"
    if source_key in dataset_cfg:
        spec = dataset_cfg[source_key]
        if not isinstance(spec, dict):
            raise ValueError(f"{source_key} must be a mapping with 'type' and optional 'config'.")
        return spec
    if legacy_file_key in dataset_cfg:
        return {"type": "jsonl_file", "config": {"path": dataset_cfg[legacy_file_key]}}
    raise KeyError(f"Missing dataset source config: expected '{source_key}' or '{legacy_file_key}'.")


def _apply_pipeline(rows: Iterable[Row], pipeline_specs: list[dict[str, Any]]) -> list[Row]:
    """Apply a sequence of pipeline steps to rows.

    Each pipeline spec should contain ``name`` and optional ``config``.
    If a step returns ``None``, that row is dropped and remaining steps are
    skipped for that row.

    Args:
        rows: Source rows.
        pipeline_specs: Ordered list of step specs.

    Returns:
        Processed rows that are not dropped by any step.

    Raises:
        ValueError: If a step name is empty.
        KeyError: If a step name is not registered.
    """
    out: list[Row] = []
    for raw_row in rows:
        row: Row | None = raw_row
        for step_spec in pipeline_specs:
            name = str(step_spec.get("name", "")).strip()
            if not name:
                raise ValueError("Pipeline step requires non-empty 'name'.")
            if name not in PIPELINE_REGISTRY:
                raise KeyError(f"Unknown pipeline step: {name}. Available: {sorted(PIPELINE_REGISTRY)}")
            step = PIPELINE_REGISTRY[name]
            step_cfg = step_spec.get("config", {}) if isinstance(step_spec, dict) else {}
            row = step(row, step_cfg) if row is not None else None
            if row is None:
                break
        if row is not None:
            out.append(row)
    return out


def load_split_rows(dataset_cfg: dict[str, Any], split: str) -> list[Row]:
    """Load and process one dataset split based on config.

    It resolves a source via registry and then executes pipeline steps in this
    order: dataset-level ``pipeline`` first, then ``{split}_pipeline``.

    Example:
        >>> cfg = {
        ...     "train_file": "data/synthetic_rag_train.jsonl",
        ...     "pipeline": [{"name": "strip_fields", "config": {"fields": ["question", "answer"]}}],
        ...     "train_pipeline": [{"name": "drop_empty_answer"}],
        ... }
        >>> rows = load_split_rows(cfg, "train")
        >>> isinstance(rows, list)
        True

    Args:
        dataset_cfg: Full dataset configuration.
        split: Split name, e.g. ``train``/``eval``.

    Returns:
        Materialized rows after source loading and pipeline processing.

    Raises:
        KeyError: If source type is unknown or source config is missing.
    """
    spec = _resolve_source_spec(dataset_cfg, split)
    source_type = str(spec.get("type", "")).strip()
    source_cfg = spec.get("config", {}) if isinstance(spec, dict) else {}
    if source_type not in SOURCE_REGISTRY:
        raise KeyError(f"Unknown source type: {source_type}. Available: {sorted(SOURCE_REGISTRY)}")
    source_factory = SOURCE_REGISTRY[source_type]
    rows = source_factory(source_cfg)

    global_pipeline = dataset_cfg.get("pipeline", [])
    split_pipeline = dataset_cfg.get(f"{split}_pipeline", [])
    pipeline_specs: list[dict[str, Any]] = []
    if isinstance(global_pipeline, list):
        pipeline_specs.extend([p for p in global_pipeline if isinstance(p, dict)])
    if isinstance(split_pipeline, list):
        pipeline_specs.extend([p for p in split_pipeline if isinstance(p, dict)])
    return _apply_pipeline(rows, pipeline_specs)
