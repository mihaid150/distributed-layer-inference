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


FEATURE_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "baseline-json-fp32": {},
    "stage4-fastpath": {},
    "transport-raw-fp16": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "forward_dedupe_enabled": False,
        "persistent_sessions_enabled": True,
    },
    "binary-fp16": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "forward_dedupe_enabled": False,
        "persistent_sessions_enabled": True,
    },
    "decode-raw-fp16-kv": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "persistent_sessions_enabled": True,
    },
    "binary-fp16-kv": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "persistent_sessions_enabled": True,
    },
    "rebalance-raw-fp16-kv": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "rebalance_profile": "latency_balanced_v1",
        "persistent_sessions_enabled": True,
    },
    "binary-fp16-kv-rebalance": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "rebalance_profile": "latency_balanced_v1",
        "persistent_sessions_enabled": True,
    },
    "topology-raw-fp16-kv": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "rebalance_profile": "latency_balanced_v1",
        "topology_aware_routing": True,
        "persistent_sessions_enabled": True,
    },
    "binary-fp16-kv-rebalance-topology": {
        "transport_mode": "binary_octet_stream",
        "activation_precision": "fp16",
        "kv_cache_enabled": True,
        "forward_dedupe_enabled": False,
        "rebalance_profile": "latency_balanced_v1",
        "topology_aware_routing": True,
        "persistent_sessions_enabled": True,
    },
}


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
    feature_flags: Dict[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    response = requests.post(
        f"{gateway_url.rstrip('/')}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "temperature": temperature,
            "feature_flags": feature_flags,
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
    profile_name: str,
    feature_flags: Dict[str, Any],
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
            feature_flags=feature_flags,
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
            "profile_name": profile_name,
            "feature_flags": feature_flags,
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
            "profile_name": profile_name,
            "feature_flags": feature_flags,
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
    aggregates = [
        item["gateway_response"].get("summary_metrics", {}).get("aggregate", {})
        for item in successful
    ]

    summary = {
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

    for source_key, output_key in {
        "comm_compute_ratio": "comm_compute_ratio_mean",
        "rpc_compute_ratio": "rpc_compute_ratio_mean",
        "true_comm_compute_ratio": "true_comm_compute_ratio_mean",
        "rpc_wall_time_ms_sum": "rpc_wall_time_ms_mean",
        "true_comm_ms_sum": "true_comm_ms_mean",
        "payload_mebibytes_sum": "payload_mebibytes_mean",
        "network_delta_mebibytes_sum": "network_mebibytes_mean",
        "stage4_compute_time_ms_per_token": "stage4_compute_ms_per_token_mean",
        "token_latency_ms_p50": "token_latency_ms_p50_mean",
        "token_latency_ms_p95": "token_latency_ms_p95_mean",
        "token_latency_ms_p99": "token_latency_ms_p99_mean",
    }.items():
        values = [
            float(aggregate.get(source_key, 0.0))
            for aggregate in aggregates
            if source_key in aggregate
        ]
        summary[output_key] = statistics.mean(values) if values else 0.0

    return summary


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
        "--profiles",
        nargs="+",
        default=["baseline"],
        choices=sorted(FEATURE_PROFILES.keys()),
        help="Named feature profiles to compare.",
    )
    parser.add_argument(
        "--feature-flags-json",
        default=None,
        help="Optional JSON object merged into each selected profile.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per case/profile. Warmups are discarded from output files.",
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
    feature_flag_override = (
        json.loads(args.feature_flags_json) if args.feature_flags_json else {}
    )

    for prompt_item in prompts:
        for max_new_tokens in args.max_new_tokens:
            for profile_name in args.profiles:
                feature_flags = dict(FEATURE_PROFILES[profile_name])
                feature_flags.update(feature_flag_override)
                case_rows: List[Dict[str, Any]] = []

                print(
                    f"Running case profile={profile_name} prompt_id={prompt_item.prompt_id} "
                    f"max_new_tokens={max_new_tokens}"
                )

                for warmup_index in range(max(0, args.warmup_runs)):
                    warmup = run_single_case(
                        gateway_url=args.gateway_url,
                        prompt_item=prompt_item,
                        max_new_tokens=max_new_tokens,
                        temperature=args.temperature,
                        profile_name=profile_name,
                        feature_flags=feature_flags,
                        run_index=-(warmup_index + 1),
                        timeout_seconds=args.timeout_seconds,
                    )
                    print(
                        f"  warmup={warmup_index} status={warmup['status']} "
                        f"wall_latency_ms={warmup['wall_latency_ms']:.2f}"
                    )

                for run_index in range(args.runs_per_case):
                    result = run_single_case(
                        gateway_url=args.gateway_url,
                        prompt_item=prompt_item,
                        max_new_tokens=max_new_tokens,
                        temperature=args.temperature,
                        profile_name=profile_name,
                        feature_flags=feature_flags,
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
                        "profile_name": profile_name,
                        "feature_flags": feature_flags,
                        "prompt_id": prompt_item.prompt_id,
                        "prompt_category": prompt_item.category,
                        "prompt_length": prompt_item.approximate_length,
                        "max_new_tokens": max_new_tokens,
                        "min_new_tokens": max_new_tokens,
                        "temperature": args.temperature,
                        "warmup_runs_discarded": max(0, args.warmup_runs),
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
