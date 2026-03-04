"""
JSONL schema for RAG training and evaluation samples.

Aligns with training/prompt_template.yaml: question, query_context_txt, search_results_txt → answer.
"""

from __future__ import annotations

from typing import Any, TypedDict


# --- Required and optional keys ---
REQUIRED_KEYS = ("sample_id", "question", "refs", "answer")
OPTIONAL_KEYS = ("query_context_txt", "labels", "metadata")
LABEL_KEYS = ("topic",)
METADATA_KEYS = ("source", "version", "split")


class Labels(TypedDict, total=False):
    topic: str


class SampleMetadata(TypedDict, total=False):
    source: str
    version: str
    split: str


class RAGSample(TypedDict, total=False):
    sample_id: str
    question: str
    query_context_txt: str
    refs: list[str]
    answer: str
    labels: Labels
    metadata: SampleMetadata


def schema_doc() -> str:
    """Return schema documentation for README and validation messages."""
    return """
RAG JSONL schema (one JSON object per line):

Required:
  - sample_id (str): Unique id, e.g. "rag_0001"
  - question (str): User question
  - refs (list[str]): Reference text chunks (rendered as search_results_txt in prompt)
  - answer (str): Ground-truth answer; must be non-empty for training

Optional:
  - query_context_txt (str): Prior Q&A history for multi-turn
  - labels (object): e.g. topic (str)
  - metadata (object): source, version, split (e.g. "train"|"eval")
"""
