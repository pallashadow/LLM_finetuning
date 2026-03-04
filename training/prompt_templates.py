"""
Load prompt template from training/prompt_template.yaml and render RAG prompts.

Expects YAML body: prompt_before, topic_line_template, history_line_template, refs_line_template, prompt_after.
Placeholders: {question}, {query_context_txt}, {search_results_txt}.
Sample schema: question, refs (list[str]), answer; optional query_context_txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = _TEMPLATE_DIR / "prompt_template.yaml"


def _load_body() -> dict[str, Any]:
    """Load body section from prompt_template.yaml."""
    if not _DEFAULT_YAML.exists():
        return {}
    with open(_DEFAULT_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("body") or {}


def render(example: dict[str, Any]) -> str:
    """
    Render one RAG training example using prompt_template.yaml.
    Uses example["question"], example["refs"] (→ search_results_txt), example["answer"];
    optional example["query_context_txt"] for history.
    """
    prompt = render_inference(example)
    answer = (example.get("answer") or "").strip()
    return "\n\n".join(p for p in [prompt, answer] if p)


def render_inference(example: dict[str, Any]) -> str:
    """
    Render one RAG inference prompt using prompt_template.yaml.
    Uses example["question"], example["refs"] (→ search_results_txt);
    optional example["query_context_txt"] for history.
    """
    body = _load_body()
    prompt_before = (body.get("prompt_before") or "").strip()
    prompt_after = (body.get("prompt_after") or "").strip()
    topic_tpl = body.get("topic_line_template") or "Q: {question}\n"
    history_tpl = body.get("history_line_template") or "History: {query_context_txt}\n"
    refs_tpl = body.get("refs_line_template") or "Refs: {search_results_txt}\n"

    question = (example.get("question") or "").strip()
    query_context_txt = (example.get("query_context_txt") or "").strip()
    refs = example.get("refs") or []
    search_results_txt = "\n".join(r.strip() for r in refs if isinstance(r, str))
    parts: list[str] = []
    if prompt_before:
        parts.append(prompt_before)
    parts.append(topic_tpl.format(question=question))
    if query_context_txt:
        parts.append(history_tpl.format(query_context_txt=query_context_txt))
    parts.append(refs_tpl.format(search_results_txt=search_results_txt))
    if prompt_after:
        parts.append(prompt_after)
    return "\n\n".join(p for p in parts if p)
