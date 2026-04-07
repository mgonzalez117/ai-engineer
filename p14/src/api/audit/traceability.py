from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple traceability audit for p14 API traces.")
    parser.add_argument("--trace-log", default="artifacts/api/traces/interactions.jsonl")
    parser.add_argument("--output", default="artifacts/api/reports/traceability_audit.json")
    args = parser.parse_args()

    trace_path = Path(args.trace_log)
    if not trace_path.exists():
        print(f"Trace log not found: {trace_path}")
        return 1

    required_fields = {
        "timestamp",
        "request_id",
        "endpoint",
        "status_code",
        "latency_ms",
        "model",
    }

    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    malformed = 0
    records: list[dict] = []

    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1

    missing_required_count = 0
    request_ids: list[str] = []
    success_latencies: list[float] = []
    status_counts: dict[str, int] = {}

    for record in records:
        if not required_fields.issubset(record.keys()):
            missing_required_count += 1

        request_id = str(record.get("request_id", "")).strip()
        if request_id:
            request_ids.append(request_id)

        status = str(record.get("status_code", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        status_code = record.get("status_code")
        latency = record.get("latency_ms")
        if isinstance(status_code, int) and status_code < 400 and isinstance(latency, (int, float)):
            success_latencies.append(float(latency))

    duplicate_request_ids = max(len(request_ids) - len(set(request_ids)), 0)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "trace_log": str(trace_path),
        "line_count": len(lines),
        "record_count": len(records),
        "malformed_lines": malformed,
        "missing_required_field_records": missing_required_count,
        "duplicate_request_ids": duplicate_request_ids,
        "status_counts": status_counts,
        "latency_ms_success": {
            "p95": round(percentile(success_latencies, 95), 2) if percentile(success_latencies, 95) is not None else None,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
