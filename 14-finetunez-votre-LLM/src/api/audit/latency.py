from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * (p / 100)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run(args: argparse.Namespace) -> dict:
    endpoint = f"{args.base_url.rstrip('/')}/v1/generate"
    headers = {"Content-Type": "application/json"}

    latencies: list[float] = []
    statuses: list[int] = []
    errors: list[str] = []

    with httpx.Client(timeout=args.timeout_seconds) as client:
        for _ in range(args.requests):
            payload = {
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            start = time.perf_counter()
            try:
                response = client.post(endpoint, headers=headers, json=payload)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(round(elapsed_ms, 2))
                statuses.append(response.status_code)
                if response.status_code >= 400:
                    errors.append(response.text[:200])
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(round(elapsed_ms, 2))
                statuses.append(0)
                errors.append(str(exc))

    success_count = sum(1 for s in statuses if s == 200)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "requests": args.requests,
        "success_count": success_count,
        "error_count": args.requests - success_count,
        "error_rate": round((args.requests - success_count) / args.requests, 4) if args.requests else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else None,
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(percentile(latencies, 50), 2) if percentile(latencies, 50) is not None else None,
            "p95": round(percentile(latencies, 95), 2) if percentile(latencies, 95) is not None else None,
            "max": max(latencies) if latencies else None,
        },
        "statuses": statuses,
        "errors": errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple latency test for p14 API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--prompt", default="Douleur thoracique depuis 2 jours.")
    parser.add_argument("--output", default="artifacts/api/reports/latency_report.json")
    args = parser.parse_args()

    report = run(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
