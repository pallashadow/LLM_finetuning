from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    fraction = rank - low
    return values[low] * (1 - fraction) + values[high] * fraction


def _single_request(endpoint: str, payload: dict[str, Any], timeout: float) -> tuple[float, int]:
    start = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, response.status_code


def run_latency_test(
    endpoint: str,
    rounds: int,
    timeout: float,
    warmup_rounds: int = 3,
    concurrency: int = 1,
) -> dict[str, Any]:
    samples_ms: list[float] = []
    status_codes: list[int] = []
    run_id = uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "input": "I was charged twice this month. How can I get a refund?",
        "max_tokens": 128,
    }

    for _ in range(warmup_rounds):
        _single_request(endpoint=endpoint, payload=payload, timeout=timeout)

    if concurrency <= 1:
        for _ in range(rounds):
            elapsed_ms, status = _single_request(endpoint=endpoint, payload=payload, timeout=timeout)
            samples_ms.append(elapsed_ms)
            status_codes.append(status)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(_single_request, endpoint=endpoint, payload=payload, timeout=timeout)
                for _ in range(rounds)
            ]
            for future in as_completed(futures):
                elapsed_ms, status = future.result()
                samples_ms.append(elapsed_ms)
                status_codes.append(status)

    samples_ms.sort()
    return {
        "run_id": run_id,
        "started_at_utc": started_at,
        "endpoint": endpoint,
        "warmup_rounds": warmup_rounds,
        "rounds": rounds,
        "concurrency": concurrency,
        "status_codes": sorted(list(set(status_codes))),
        "avg_ms": statistics.mean(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark scaffold.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--warmup_rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--benchmark_csv", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cost_per_1k_queries_usd", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_latency_test(
        endpoint=args.endpoint,
        rounds=args.rounds,
        timeout=args.timeout,
        warmup_rounds=args.warmup_rounds,
        concurrency=args.concurrency,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[latency] Wrote JSON report to {output_path}")

    if args.benchmark_csv and args.model:
        from evaluation.benchmark import add_row, init_benchmark_file

        csv_path = Path(args.benchmark_csv)
        init_benchmark_file(csv_path)
        add_row(
            path=csv_path,
            model=args.model,
            avg_latency=float(stats["avg_ms"]),
            p95_latency=float(stats["p95_ms"]),
            cost_1k=args.cost_per_1k_queries_usd,
            run_id=str(stats["run_id"]),
            timestamp_utc=str(stats["started_at_utc"]),
        )
        print(f"[latency] Appended benchmark row to {csv_path}")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


