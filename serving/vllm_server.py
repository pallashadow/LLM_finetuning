from __future__ import annotations

import argparse
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from serving.utils import (
    AnswerResponse,
    BackendRegistry,
    RouteRegistry,
    build_default_backend_registry,
    build_default_route_registry,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "auto",
    "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
    "adapter_path": None,
    "max_model_len": 4096,
    "route_sets": "local,openai,sagemaker",
}


def _parse_route_sets(raw_route_sets: str) -> list[str]:
    items = [item.strip().lower() for item in raw_route_sets.split(",")]
    return [item for item in items if item]


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="OpenSupport-LLM Inference API", version="0.2.0")
    app.state.app_config = dict(config)
    app.state.backend_registry = build_default_backend_registry()
    app.state.route_registry = build_default_route_registry()
    app.state.backend = None

    def get_backend():
        backend = getattr(app.state, "backend", None)
        if backend is not None:
            return backend

        app_config = app.state.app_config
        backend_name = str(app_config.get("backend", "auto"))
        model_name_or_path = str(app_config["model_name_or_path"])
        adapter_path = app_config.get("adapter_path")
        max_model_len = int(app_config.get("max_model_len", 4096))
        backend_registry: BackendRegistry = app.state.backend_registry

        if backend_name == "auto":
            # Prefer vLLM for performance, then fall back to Transformers
            # if vLLM cannot initialize in the current runtime.
            app.state.backend = backend_registry.create_with_fallback(
                preferred_names=["vllm", "transformers"],
                model_name_or_path=model_name_or_path,
                adapter_path=adapter_path,
                max_model_len=max_model_len,
            )
        else:
            app.state.backend = backend_registry.create(
                name=backend_name,
                model_name_or_path=model_name_or_path,
                adapter_path=adapter_path,
                max_model_len=max_model_len,
            )
        return app.state.backend

    def run_generation(input_text: str, max_tokens: int) -> AnswerResponse:
        start = time.perf_counter()
        backend = get_backend()
        answer = backend.generate(input_text, max_tokens=max_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return AnswerResponse(
            answer=answer or "No answer generated.",
            model=backend.model_name,
            latency_ms=latency_ms,
        )

    route_sets = _parse_route_sets(str(app.state.app_config.get("route_sets", "local")))
    route_registry: RouteRegistry = app.state.route_registry
    route_registry.apply(route_sets, app=app, run_generation=run_generation, config=app.state.app_config)
    return app


app = create_app(DEFAULT_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenSupport-LLM unified inference server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--route_sets",
        default="local,openai,sagemaker",
        help="Comma-separated route sets: local,openai,sagemaker",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = {
        "backend": args.backend,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "max_model_len": args.max_model_len,
        "route_sets": args.route_sets,
    }
    runtime_app = create_app(runtime_config)
    uvicorn.run(runtime_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


