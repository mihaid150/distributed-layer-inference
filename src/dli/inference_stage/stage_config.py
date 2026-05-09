from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class StageComponents:
    embedding: bool
    layers: List[int]
    norm: bool
    lm_head: bool


@dataclass(frozen=True)
class StageConfig:
    stage_id: int
    service_name: str
    physical_node: str
    partition_file: str
    components: StageComponents
    next_stage_url: Optional[str]


def load_stage_config(stage_id: Optional[int] = None) -> StageConfig:
    if stage_id is None:
        raw_stage_id = os.getenv("STAGE_ID")
        if raw_stage_id is None:
            raise RuntimeError("STAGE_ID environment variable is required.")
        stage_id = int(raw_stage_id)

    config_path = os.getenv("STAGE_MAP_PATH", "/app/configs/stage_map.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    stages = config.get("inference_stages", [])

    for stage in stages:
        if int(stage["stage_id"]) == stage_id:
            components = stage["components"]

            next_stage_url = os.getenv("NEXT_STAGE_URL", stage.get("next_stage_url"))

            if next_stage_url == "":
                next_stage_url = None

            return StageConfig(
                stage_id=stage_id,
                service_name=stage["service_name"],
                physical_node=stage["physical_node"],
                partition_file=stage["partition_file"],
                components=StageComponents(
                    embedding=bool(components.get("embedding", False)),
                    layers=list(components.get("layers", [])),
                    norm=bool(components.get("norm", False)),
                    lm_head=bool(components.get("lm_head", False)),
                ),
                next_stage_url=next_stage_url,
            )

    raise RuntimeError(f"No stage configuration found for STAGE_ID={stage_id}.")