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
