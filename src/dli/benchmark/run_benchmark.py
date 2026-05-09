from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

from dli.benchmark.prompt_sets import BenchmarkPrompt, get_all_prompts


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def call_gateway(
    *,
    gateway_url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> Dict[str, Any]:
    response = requests.post(
        f"{gateway_url.rstrip('/')}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def run_single_case(
    *,
    gateway_url: str,
    prompt_item: BenchmarkPrompt,
    max_new_tokens: int,
    temperature: float,
    run_index: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    wall_start = time.perf_counter()

    try:
        response = call_gateway(
            gateway_url=gateway_url,
            prompt=prompt_item.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

        wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0

        return {
            "status": "ok",
            "run_index": run_index,
            "prompt_id": prompt_item.prompt_id,
            "prompt_category": prompt_item.category,
            "prompt_length": prompt_item.approximate_length,
            "prompt": prompt_item.prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "wall_latency_ms": wall_latency_ms,
            "gateway_response": response,
            "error": None,
        }

    except Exception as exc:
        wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0

        return {
            "status": "error",
            "run_index": run_index,
            "prompt_id": prompt_item.prompt_id,
            "prompt_category": prompt_item.category,
            "prompt_length": prompt_item.approximate_length,
            "prompt": prompt_item.prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "wall_latency_ms": wall_latency_ms,
            "gateway_response": None,
            "error": str(exc),
        }


def summarize_case(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [item for item in results if item["status"] == "ok"]

    if not successful:
        return {
            "num_runs": len(results),
            "num_successful": 0,
            "num_failed": len(results),
        }

    wall_latencies = [item["wall_latency_ms"] for item in successful]
    gateway_latencies = [
        item["gateway_response"]["total_latency_ms"] for item in successful
    ]
    tokens_per_second = [
        item["gateway_response"]["tokens_per_second"] for item in successful
    ]

    generated_tokens = [
        len(item["gateway_response"]["generated_token_ids"]) for item in successful
    ]

    return {
        "num_runs": len(results),
        "num_successful": len(successful),
        "num_failed": len(results) - len(successful),
        "wall_latency_ms_mean": statistics.mean(wall_latencies),
        "wall_latency_ms_stdev": statistics.stdev(wall_latencies)
        if len(wall_latencies) > 1
        else 0.0,
        "gateway_latency_ms_mean": statistics.mean(gateway_latencies),
        "gateway_latency_ms_stdev": statistics.stdev(gateway_latencies)
        if len(gateway_latencies) > 1
        else 0.0,
        "tokens_per_second_mean": statistics.mean(tokens_per_second),
        "tokens_per_second_stdev": statistics.stdev(tokens_per_second)
        if len(tokens_per_second) > 1
        else 0.0,
        "generated_tokens_mean": statistics.mean(generated_tokens),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark against the Inference Gateway."
    )

    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--runs-per-case",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    experiment_id = f"benchmark_{utc_timestamp()}"

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    prompts = get_all_prompts()

    for prompt_item in prompts:
        for max_new_tokens in args.max_new_tokens:
            case_rows: List[Dict[str, Any]] = []

            print(
                f"Running case prompt_id={prompt_item.prompt_id} "
                f"max_new_tokens={max_new_tokens}"
            )

            for run_index in range(args.runs_per_case):
                result = run_single_case(
                    gateway_url=args.gateway_url,
                    prompt_item=prompt_item,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    run_index=run_index,
                    timeout_seconds=args.timeout_seconds,
                )

                case_rows.append(result)
                all_rows.append(result)

                print(
                    f"  run={run_index} status={result['status']} "
                    f"wall_latency_ms={result['wall_latency_ms']:.2f}"
                )

            case_summary = summarize_case(case_rows)
            case_summary.update(
                {
                    "experiment_id": experiment_id,
                    "prompt_id": prompt_item.prompt_id,
                    "prompt_category": prompt_item.category,
                    "prompt_length": prompt_item.approximate_length,
                    "max_new_tokens": max_new_tokens,
                    "temperature": args.temperature,
                }
            )
            summary_rows.append(case_summary)

    raw_path = output_dir / f"{experiment_id}.jsonl"
    summary_path = output_dir / f"{experiment_id}_summary.jsonl"

    write_jsonl(raw_path, all_rows)
    write_jsonl(summary_path, summary_rows)

    print(f"Raw results written to: {raw_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()