from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

from serving.client.litellm_api import call_llm_with_fallback


_PROMPT_YAML_PATH = Path(__file__).resolve().parents[2] / "prompts" / "prompt_rag.yaml"


def _load_prompt_body(path: Path = _PROMPT_YAML_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    body = data.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"Invalid prompt YAML body in: {path}")
    return body


def build_inference_prompt(
    question: str,
    query_context_txt: str = "",
    search_results_txt: str = "",
    *,
    prompt_path: Path = _PROMPT_YAML_PATH,
) -> str:
    body = _load_prompt_body(prompt_path)
    prompt_before = str(body.get("prompt_before") or "").strip()
    prompt_after = str(body.get("prompt_after") or "").strip()
    topic_tpl = str(body.get("topic_line_template") or "{question}")
    history_tpl = str(body.get("history_line_template") or "{query_context_txt}")
    refs_tpl = str(body.get("refs_line_template") or "{search_results_txt}")

    safe_question = (question or "").strip()
    safe_query_context = (query_context_txt or "").strip()
    safe_search_results = (search_results_txt or "").strip()

    parts: list[str] = []
    if prompt_before:
        parts.append(prompt_before)
    parts.append(topic_tpl.format(question=safe_question))
    if safe_query_context:
        parts.append(history_tpl.format(query_context_txt=safe_query_context))
    parts.append(refs_tpl.format(search_results_txt=safe_search_results))
    if prompt_after:
        parts.append(prompt_after)
    return "\n\n".join([p for p in parts if p])


async def answer_with_rag_prompt(
    *,
    question: str,
    query_context_txt: str = "",
    search_results_txt: str = "",
    model_name: str = "vllm",
    response_format: dict[str, Any] | None = None,
) -> Any:
    prompt = build_inference_prompt(
        question=question,
        query_context_txt=query_context_txt,
        search_results_txt=search_results_txt,
    )
    return await call_llm_with_fallback(
        prompt,
        model_name=model_name,
        response_format=response_format,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG prompt and call LiteLLM fallback client.")
    parser.add_argument("--question", required=True, help="User question/topic")
    parser.add_argument("--query-context-txt", default="", help="Prior Q&A history as text")
    parser.add_argument("--search-results-txt", default="", help="Retrieved references as text")
    parser.add_argument(
        "--model-name",
        default="vllm",
        help="LiteLLM router model alias (default: vllm -> serving.vllm_server)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(
        answer_with_rag_prompt(
            question=args.question,
            query_context_txt=args.query_context_txt,
            search_results_txt=args.search_results_txt,
            model_name=args.model_name,
        )
    )
    print(result)


if __name__ == "__main__":
    main()

