from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


CASES = [
    {
        "name": "missing_prompt",
        "payload": {"max_tokens": 100},
        "expected_status": 422,
    },
    {
        "name": "empty_prompt",
        "payload": {"prompt": "", "max_tokens": 100},
        "expected_status": 422,
    },
    {
        "name": "too_long_prompt",
        "payload": {"prompt": "a" * 9000, "max_tokens": 100},
        "expected_status": 422,
    },
    {
        "name": "invalid_max_tokens",
        "payload": {"prompt": "douleur thoracique", "max_tokens": 0},
        "expected_status": 422,
    },
    {
        "name": "valid_request",
        "payload": {"prompt": "Douleur thoracique depuis 2 jours", "max_tokens": 120, "temperature": 0.3},
        "expected_status": 200,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple robustness checks for p14 API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--output", default="artifacts/api/reports/robustness_report.json")
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/v1/generate"
    results = []

    with httpx.Client(timeout=args.timeout_seconds) as client:
        for case in CASES:
            try:
                response = client.post(endpoint, json=case["payload"])
                status = response.status_code
                passed = status == case["expected_status"]
                body_excerpt = response.text[:200]
            except Exception as exc:  # noqa: BLE001
                status = 0
                passed = False
                body_excerpt = str(exc)

            results.append(
                {
                    "name": case["name"],
                    "expected_status": case["expected_status"],
                    "actual_status": status,
                    "passed": passed,
                    "body_excerpt": body_excerpt,
                }
            )

    passed_count = sum(1 for item in results if item["passed"])
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "total_cases": len(results),
        "passed_cases": passed_count,
        "failed_cases": len(results) - passed_count,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output_path}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
