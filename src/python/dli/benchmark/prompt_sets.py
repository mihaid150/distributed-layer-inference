from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkPrompt:
    prompt_id: str
    prompt: str
    category: str
    approximate_length: str


PROMPTS: List[BenchmarkPrompt] = [
    BenchmarkPrompt(
        prompt_id="short_001",
        category="definition",
        approximate_length="short",
        prompt="Explain distributed inference in one sentence.",
    ),
    BenchmarkPrompt(
        prompt_id="short_002",
        category="edge_ai",
        approximate_length="short",
        prompt="Define edge AI for a computer science student.",
    ),
    BenchmarkPrompt(
        prompt_id="medium_001",
        category="systems",
        approximate_length="medium",
        prompt=(
            "Explain why running a language model on a Raspberry Pi cluster "
            "can be more challenging than running it on a single GPU server."
        ),
    ),
    BenchmarkPrompt(
        prompt_id="medium_002",
        category="distributed_systems",
        approximate_length="medium",
        prompt=(
            "Describe the trade-off between memory usage and network communication "
            "in a distributed layer inference system."
        ),
    ),
    BenchmarkPrompt(
        prompt_id="long_001",
        category="research",
        approximate_length="long",
        prompt=(
            "A research team deploys a small language model across four Raspberry Pi "
            "devices. Each device owns a subset of transformer layers, and intermediate "
            "activations are transferred from one device to the next during inference. "
            "Discuss the expected advantages, bottlenecks, and evaluation metrics for "
            "this architecture."
        ),
    ),
    BenchmarkPrompt(
        prompt_id="long_002",
        category="architecture",
        approximate_length="long",
        prompt=(
            "Consider a resource-constrained edge cluster composed of single-board "
            "computers connected through Gigabit Ethernet. The system uses a static "
            "pipeline in which each stage executes a subset of model layers. Explain how "
            "the architecture should be evaluated with respect to latency, throughput, "
            "memory, CPU utilization, and network transfer overhead."
        ),
    ),
]


def get_prompts_by_length(length: str) -> List[BenchmarkPrompt]:
    return [prompt for prompt in PROMPTS if prompt.approximate_length == length]


def get_all_prompts() -> List[BenchmarkPrompt]:
    return list(PROMPTS)