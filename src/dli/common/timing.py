from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    return now_ms() - start_ms


@contextmanager
def timed_block(name: str) -> Iterator[Dict[str, float | str]]:
    start = now_ms()
    result: Dict[str, float | str] = {
        "name": name,
        "start_ms": start,
        "duration_ms": 0.0,
    }

    try:
        yield result
    finally:
        result["duration_ms"] = elapsed_ms(start)