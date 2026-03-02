from __future__ import annotations

import argparse
import json
import os

from litellm import completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call local vllm_server.py through LiteLLM (OpenAI-compatible API)."
    )
    parser.add_argument("--api_base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--input", required=True, help="User input text.")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # LiteLLM requires an API key for OpenAI-compatible flows.
    # Local server can ignore it, so use env var or a dummy fallback.
    api_key = os.getenv("OPENAI_API_KEY", "dummy")

    response = completion(
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        custom_llm_provider="openai",
        messages=[{"role": "user", "content": args.input}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    output = response["choices"][0]["message"]["content"]
    print(output)
    print("\n--- raw response ---")
    print(json.dumps(response, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
