from __future__ import annotations

import os
import socket
import threading
from typing import Any, Dict, Optional

import psutil
from dli.common.timing import now_ms


_METRICS_LOCK = threading.Lock()
_LAST_NETWORK_COUNTERS: Optional[Dict[str, int]] = None
_LAST_NETWORK_TS_MS: Optional[float] = None
_LAST_PROCESS_IO_COUNTERS: Optional[Dict[str, int]] = None
_LAST_PROCESS_CPU_TOTAL_S: Optional[float] = None
_LAST_PROCESS_CPU_TS_MS: Optional[float] = None
_LAST_PROCESS_CPU_PERCENT: float = 0.0
_LAST_SYSTEM_CPU_TOTAL_S: Optional[float] = None
_LAST_SYSTEM_CPU_IDLE_S: Optional[float] = None
_LAST_SYSTEM_CPU_PERCENT: float = 0.0
_CPU_COUNT = max(1, int(psutil.cpu_count() or 1))


def _initialize_cpu_sampling_state() -> None:
    global _LAST_PROCESS_CPU_TOTAL_S, _LAST_PROCESS_CPU_TS_MS, _LAST_PROCESS_CPU_PERCENT
    global _LAST_SYSTEM_CPU_TOTAL_S, _LAST_SYSTEM_CPU_IDLE_S, _LAST_SYSTEM_CPU_PERCENT

    process = psutil.Process(os.getpid())
    process_times = process.cpu_times()
    system_times = psutil.cpu_times()

    _LAST_PROCESS_CPU_TOTAL_S = float(process_times.user + process_times.system)
    _LAST_PROCESS_CPU_TS_MS = now_ms()
    _LAST_PROCESS_CPU_PERCENT = max(
        0.0,
        min(float(_CPU_COUNT * 100), float(process.cpu_percent(interval=None))),
    )

    _LAST_SYSTEM_CPU_TOTAL_S = float(sum(system_times))
    _LAST_SYSTEM_CPU_IDLE_S = float(
        getattr(system_times, "idle", 0.0) + getattr(system_times, "iowait", 0.0)
    )
    _LAST_SYSTEM_CPU_PERCENT = max(
        0.0,
        min(100.0, float(psutil.cpu_percent(interval=None))),
    )

def get_hostname() -> str:
    return socket.gethostname()

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_system_memory_usage() -> Dict[str, float]:
    memory = psutil.virtual_memory()

    return {
        "total_mb": memory.total / (1024 * 1024),
        "available_mb": memory.available / (1024 * 1024),
        "used_mb": memory.used / (1024 * 1024),
        "free_mb": memory.free / (1024 * 1024),
        "percent": memory.percent,
    }

def get_cpu_usage_percent(interval: Optional[float] = None) -> float:
    if interval is not None:
        return psutil.cpu_percent(interval=interval)

    current = psutil.cpu_times()
    total_s = float(sum(current))
    idle_s = float(getattr(current, "idle", 0.0) + getattr(current, "iowait", 0.0))
    fallback_percent = float(psutil.cpu_percent(interval=None))

    global _LAST_SYSTEM_CPU_TOTAL_S, _LAST_SYSTEM_CPU_IDLE_S, _LAST_SYSTEM_CPU_PERCENT
    with _METRICS_LOCK:
        prev_total_s = _LAST_SYSTEM_CPU_TOTAL_S
        prev_idle_s = _LAST_SYSTEM_CPU_IDLE_S
        _LAST_SYSTEM_CPU_TOTAL_S = total_s
        _LAST_SYSTEM_CPU_IDLE_S = idle_s

        if prev_total_s is None or prev_idle_s is None:
            _LAST_SYSTEM_CPU_PERCENT = max(0.0, min(100.0, fallback_percent))
            return _LAST_SYSTEM_CPU_PERCENT

        total_delta = max(0.0, total_s - prev_total_s)
        idle_delta = max(0.0, idle_s - prev_idle_s)

        if total_delta <= 1e-9:
            return _LAST_SYSTEM_CPU_PERCENT

        used_ratio = 1.0 - (idle_delta / total_delta)
        _LAST_SYSTEM_CPU_PERCENT = max(0.0, min(100.0, used_ratio * 100.0))
        return _LAST_SYSTEM_CPU_PERCENT

def get_process_cpu_percent() -> float:
    process = psutil.Process(os.getpid())
    cpu_times = process.cpu_times()
    process_total_s = float(cpu_times.user + cpu_times.system)
    sample_ts_ms = now_ms()
    fallback_percent = float(process.cpu_percent(interval=None))

    global _LAST_PROCESS_CPU_TOTAL_S, _LAST_PROCESS_CPU_TS_MS, _LAST_PROCESS_CPU_PERCENT
    with _METRICS_LOCK:
        prev_total_s = _LAST_PROCESS_CPU_TOTAL_S
        prev_ts_ms = _LAST_PROCESS_CPU_TS_MS
        _LAST_PROCESS_CPU_TOTAL_S = process_total_s
        _LAST_PROCESS_CPU_TS_MS = sample_ts_ms

        if prev_total_s is None or prev_ts_ms is None:
            _LAST_PROCESS_CPU_PERCENT = max(
                0.0,
                min(float(_CPU_COUNT * 100), fallback_percent),
            )
            return _LAST_PROCESS_CPU_PERCENT

        elapsed_s = max(1e-9, (sample_ts_ms - prev_ts_ms) / 1000.0)
        cpu_delta_s = max(0.0, process_total_s - prev_total_s)
        cpu_percent = (cpu_delta_s / elapsed_s) * 100.0
        _LAST_PROCESS_CPU_PERCENT = max(
            0.0,
            min(float(_CPU_COUNT * 100), cpu_percent),
        )
        return _LAST_PROCESS_CPU_PERCENT

def get_process_threads_count() -> int:
    process = psutil.Process(os.getpid())
    return int(process.num_threads())

def get_process_ctx_switches() -> Dict[str, int]:
    process = psutil.Process(os.getpid())
    ctx = process.num_ctx_switches()
    return {
        "voluntary": int(ctx.voluntary),
        "involuntary": int(ctx.involuntary),
    }

def get_network_counters() -> Dict[str, int]:
    counters = psutil.net_io_counters()

    return {
        "bytes_sent": counters.bytes_sent,
        "bytes_recv": counters.bytes_recv,
        "packets_sent": counters.packets_sent,
        "packets_recv": counters.packets_recv,
    }

def get_process_io_counters() -> Dict[str, int]:
    process = psutil.Process(os.getpid())
    try:
        io = process.io_counters()
        return {
            "read_bytes": int(io.read_bytes),
            "write_bytes": int(io.write_bytes),
            "read_count": int(io.read_count),
            "write_count": int(io.write_count),
        }
    except Exception:
        return {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
        }

def _compute_counter_delta(
    *,
    previous: Optional[Dict[str, int]],
    current: Dict[str, int],
) -> Dict[str, int]:
    if previous is None:
        return {key: 0 for key in current}
    delta: Dict[str, int] = {}
    for key, value in current.items():
        prev_value = previous.get(key, value)
        delta[key] = int(max(0, value - prev_value))
    return delta

def _compute_network_delta(
    current_network: Dict[str, int],
) -> Dict[str, Any]:
    global _LAST_NETWORK_COUNTERS, _LAST_NETWORK_TS_MS

    current_ts_ms = now_ms()
    with _METRICS_LOCK:
        previous = _LAST_NETWORK_COUNTERS
        previous_ts = _LAST_NETWORK_TS_MS
        _LAST_NETWORK_COUNTERS = dict(current_network)
        _LAST_NETWORK_TS_MS = current_ts_ms

    delta = _compute_counter_delta(previous=previous, current=current_network)

    elapsed_ms = 0.0
    if previous_ts is not None:
        elapsed_ms = max(0.0, current_ts_ms - previous_ts)

    elapsed_s = elapsed_ms / 1000.0 if elapsed_ms > 0.0 else 0.0
    total_delta_bytes = int(delta["bytes_sent"] + delta["bytes_recv"])
    bandwidth_mbps_total = (total_delta_bytes * 8.0 / elapsed_s / 1_000_000.0) if elapsed_s > 0.0 else 0.0
    bandwidth_mbps_tx = (delta["bytes_sent"] * 8.0 / elapsed_s / 1_000_000.0) if elapsed_s > 0.0 else 0.0
    bandwidth_mbps_rx = (delta["bytes_recv"] * 8.0 / elapsed_s / 1_000_000.0) if elapsed_s > 0.0 else 0.0

    return {
        "elapsed_ms": elapsed_ms,
        "bytes_sent": int(delta["bytes_sent"]),
        "bytes_recv": int(delta["bytes_recv"]),
        "packets_sent": int(delta["packets_sent"]),
        "packets_recv": int(delta["packets_recv"]),
        "bytes_total": total_delta_bytes,
        "bandwidth_mbps_total": bandwidth_mbps_total,
        "bandwidth_mbps_tx": bandwidth_mbps_tx,
        "bandwidth_mbps_rx": bandwidth_mbps_rx,
    }

def _compute_process_io_delta(
    current_io: Dict[str, int],
) -> Dict[str, int]:
    global _LAST_PROCESS_IO_COUNTERS
    with _METRICS_LOCK:
        previous = _LAST_PROCESS_IO_COUNTERS
        _LAST_PROCESS_IO_COUNTERS = dict(current_io)
    return _compute_counter_delta(previous=previous, current=current_io)


def make_stage_metric(
    *,
    service_name: str,
    stage_id: Optional[int],
    token_index: int,
    compute_time_ms: float,
    transfer_time_ms: float = 0.0,
    rpc_wall_time_ms: Optional[float] = None,
    true_comm_ms: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_memory = get_system_memory_usage()
    process_memory_mb = get_process_memory_mb()
    system_cpu_percent = get_cpu_usage_percent(interval=None)
    process_cpu_percent = get_process_cpu_percent()
    process_threads = get_process_threads_count()
    process_ctx_switches = get_process_ctx_switches()
    network_totals = get_network_counters()
    network_delta = _compute_network_delta(network_totals)
    process_io_totals = get_process_io_counters()
    process_io_delta = _compute_process_io_delta(process_io_totals)
    rpc_wall = transfer_time_ms if rpc_wall_time_ms is None else rpc_wall_time_ms

    metric: Dict[str, Any] = {
        "timestamp_ms": now_ms(),
        "hostname": get_hostname(),
        "service_name": service_name,
        "stage_id": stage_id,
        "token_index": token_index,
        "compute_time_ms": compute_time_ms,
        "rpc_wall_time_ms": rpc_wall,
        "true_comm_ms": true_comm_ms,
        # Backward-compatible alias for previous consumers. This is RPC wall
        # time, not pure network transfer time.
        "transfer_time_ms": transfer_time_ms,
        "process_memory_mb": process_memory_mb,
        # Keep backward-compatible key name for previous consumers:
        "cpu_percent": process_cpu_percent,
        "process_cpu_percent": process_cpu_percent,
        "system_cpu_percent": system_cpu_percent,
        "process_threads": process_threads,
        "process_context_switches": process_ctx_switches,
        "system_memory": system_memory,
        "network": network_totals,
        "network_delta": network_delta,
        "process_io": process_io_totals,
        "process_io_delta": process_io_delta,
    }

    if extra:
        metric.update(extra)

    return metric


def compute_tokens_per_second(num_tokens: int, total_latency_ms: float) -> float:
    if total_latency_ms <= 0:
        return 0.0

    return num_tokens / (total_latency_ms / 1000.0)


_initialize_cpu_sampling_state()
