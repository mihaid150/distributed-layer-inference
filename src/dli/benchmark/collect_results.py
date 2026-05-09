from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten_raw_result(row: Dict[str, Any]) -> Dict[str, Any]:
    response = row.get("gateway_response") or {}

    token_metrics = response.get("token_metrics", [])

    per_token_latencies = [
        item.get("latency_ms", 0.0)
        for item in token_metrics
    ]

    all_stage_metrics = []
    for token in token_metrics:
        all_stage_metrics.extend(token.get("stage_metrics", []))

    compute_times = [
        metric.get("compute_time_ms", 0.0)
        for metric in all_stage_metrics
    ]

    transfer_times = [
        metric.get("transfer_time_ms", 0.0)
        for metric in all_stage_metrics
    ]

    memory_values = [
        metric.get("process_memory_mb", 0.0)
        for metric in all_stage_metrics
    ]

    cpu_values = [
        metric.get("process_cpu_percent", metric.get("cpu_percent", 0.0))
        for metric in all_stage_metrics
    ]

    network_delta_bytes = [
        (metric.get("network_delta") or {}).get(
            "bytes_total",
            (metric.get("network_delta") or {}).get("bytes_sent", 0)
            + (metric.get("network_delta") or {}).get("bytes_recv", 0),
        )
        for metric in all_stage_metrics
    ]

    payload_b64_bytes = [
        metric.get("outbound_payload_b64_bytes", 0)
        for metric in all_stage_metrics
    ]

    return {
        "status": row.get("status"),
        "run_index": row.get("run_index"),
        "prompt_id": row.get("prompt_id"),
        "prompt_category": row.get("prompt_category"),
        "prompt_length": row.get("prompt_length"),
        "max_new_tokens": row.get("max_new_tokens"),
        "temperature": row.get("temperature"),
        "wall_latency_ms": row.get("wall_latency_ms"),
        "gateway_total_latency_ms": response.get("total_latency_ms"),
        "tokens_per_second": response.get("tokens_per_second"),
        "generated_token_count": len(response.get("generated_token_ids", [])),
        "mean_token_latency_ms": mean(per_token_latencies),
        "mean_stage_compute_time_ms": mean(compute_times),
        "mean_stage_transfer_time_ms": mean(transfer_times),
        "max_stage_memory_mb": max(memory_values) if memory_values else None,
        "max_stage_cpu_percent": max(cpu_values) if cpu_values else None,
        "sum_network_delta_mb": (sum(network_delta_bytes) / (1024.0 * 1024.0)) if network_delta_bytes else None,
        "sum_payload_b64_mb": (sum(payload_b64_bytes) / (1024.0 * 1024.0)) if payload_b64_bytes else None,
        "error": row.get("error"),
    }


def mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert benchmark JSONL results into CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw benchmark JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".csv")

    rows = [flatten_raw_result(row) for row in read_jsonl(input_path)]

    write_csv(output_path, rows)

    print(f"CSV written to: {output_path}")


if __name__ == "__main__":
    main()
