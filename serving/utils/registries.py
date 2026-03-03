from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI


RouteRegistrar = Callable[[FastAPI, Callable[[str, int], Any], dict[str, Any]], None]


class RouteRegistry:
    def __init__(self) -> None:
        self._registrars: dict[str, RouteRegistrar] = {}

    def register(self, name: str, registrar: RouteRegistrar) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Route set name cannot be empty.")
        self._registrars[key] = registrar

    def apply(
        self,
        names: list[str],
        app: FastAPI,
        run_generation: Callable[[str, int], Any],
        config: dict[str, Any],
    ) -> None:
        for name in names:
            key = name.strip().lower()
            if not key:
                continue
            registrar = self._registrars.get(key)
            if registrar is None:
                raise KeyError(f"Route set `{name}` is not registered.")
            registrar(app, run_generation, config)


def build_default_backend_registry():
    from serving.utils.backends import BackendRegistry, TransformersBackend, VLLMBackend

    registry = BackendRegistry()
    registry.register("vllm", VLLMBackend)
    registry.register("transformers", TransformersBackend)
    return registry


def build_default_route_registry() -> RouteRegistry:
    from serving.utils.routes import (
        register_local_routes,
        register_openai_routes,
        register_sagemaker_routes,
    )

    registry = RouteRegistry()
    registry.register("local", register_local_routes)
    registry.register("openai", register_openai_routes)
    registry.register("sagemaker", register_sagemaker_routes)
    return registry
