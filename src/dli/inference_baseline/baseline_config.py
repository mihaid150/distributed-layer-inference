from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class BaselineConfig:
    service_name: str
    model_name: str
    device: str
    torch_dtype: str


def load_baseline_config() -> BaselineConfig:
    config_path = os.getenv("STAGE_MAP_PATH", "/app/configs/stage_map.yaml")

    config_data: Dict[str, Any] = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}

    model_name = os.getenv(
        "MODEL_NAME",
        config_data.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    )

    return BaselineConfig(
        service_name=os.getenv("SERVICE_NAME", "inference-baseline"),
        model_name=model_name,
        device=os.getenv("DEVICE", "cpu"),
        torch_dtype=os.getenv("TORCH_DTYPE", "float32"),
    )