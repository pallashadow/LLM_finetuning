from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


HEADER = [
    "timestamp_utc",
    "run_id",
    "model",
    "avg_latency_ms",
    "p95_latency_ms",
    "cost_per_1k_queries_usd",
]


def init_benchmark_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def add_row(
    path: Path,
    model: str,
    avg_latency: float,
    p95_latency: float,
    cost_1k: float,
    run_id: str | None = None,
    timestamp_utc: str | None = None,
) -> None:
    run_id = run_id or uuid4().hex
    timestamp_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_utc, run_id, model, avg_latency, p95_latency, cost_1k])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark table updater.")
    parser.add_argument("--result_file", default="report/generated/benchmark_results.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--avg_latency_ms", required=True, type=float)
    parser.add_argument("--p95_latency_ms", required=True, type=float)
    parser.add_argument("--cost_per_1k_queries_usd", required=True, type=float)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--timestamp_utc", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.result_file)
    init_benchmark_file(path)
    add_row(
        path,
        model=args.model,
        avg_latency=args.avg_latency_ms,
        p95_latency=args.p95_latency_ms,
        cost_1k=args.cost_per_1k_queries_usd,
        run_id=args.run_id,
        timestamp_utc=args.timestamp_utc,
    )
    print(f"[benchmark] Appended result row to {path}")


if __name__ == "__main__":
    main()


