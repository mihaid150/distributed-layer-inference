from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml
from torch import nn

from dli.model_splitter.hf_model_utils import (
    download_snapshot_if_needed,
    get_decoder_stack,
    load_causal_lm_and_tokenizer,
    sanitize_model_id,
)


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    layer_indices: List[int]
    include_embedding: bool
    include_norm: bool
    include_lm_head: bool
    output_file: Path


@dataclass(frozen=True)
class ModelSplitPlan:
    model_id: str
    model_source: str
    output_dir: Path
    num_stages: int
    num_layers: int
    stages: List[StageSpec]


def build_balanced_layer_ranges(num_layers: int, num_stages: int) -> List[List[int]]:
    if num_stages <= 0:
        raise ValueError("num_stages must be greater than zero.")
    if num_layers <= 0:
        raise ValueError("num_layers must be greater than zero.")
    if num_stages > num_layers:
        raise ValueError(
            f"num_stages={num_stages} cannot be greater than num_layers={num_layers}."
        )

    base = num_layers // num_stages
    remainder = num_layers % num_stages

    ranges: List[List[int]] = []
    cursor = 0

    for stage_idx in range(num_stages):
        size = base + (1 if stage_idx < remainder else 0)
        layer_ids = list(range(cursor, cursor + size))
        ranges.append(layer_ids)
        cursor += size

    return ranges


def parse_layer_ranges(raw_ranges: Sequence[str], num_layers: int) -> List[List[int]]:
    """
    Parse ranges such as:
      --layer-ranges 0-2 3-5 6-8 9-11
      --layer-ranges 0,1,2 3,4,5 6,7,8 9,10,11
    """
    parsed: List[List[int]] = []
    seen: set[int] = set()

    for raw in raw_ranges:
        raw = raw.strip()
        if not raw:
            raise ValueError("Empty layer range is not allowed.")

        if "-" in raw:
            start_s, end_s = raw.split("-", maxsplit=1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid decreasing layer range: {raw}")
            layer_ids = list(range(start, end + 1))
        else:
            layer_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]

        if not layer_ids:
            raise ValueError(f"Layer range produced no layers: {raw}")

        for layer_id in layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"Layer index {layer_id} is outside valid range [0, {num_layers - 1}]."
                )
            if layer_id in seen:
                raise ValueError(f"Layer index {layer_id} appears in multiple stages.")
            seen.add(layer_id)

        parsed.append(layer_ids)

    expected = set(range(num_layers))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            f"Layer ranges must cover all layers exactly once. Missing={missing}, extra={extra}."
        )

    return parsed


def build_split_plan(
    *,
    model_id: str,
    model_source: Path | str,
    output_dir: Path,
    num_layers: int,
    num_stages: int,
    layer_ranges: Optional[Sequence[str]] = None,
) -> ModelSplitPlan:
    if layer_ranges:
        ranges = parse_layer_ranges(layer_ranges, num_layers)
        if len(ranges) != num_stages:
            raise ValueError(
                f"len(layer_ranges)={len(ranges)} must match num_stages={num_stages}."
            )
    else:
        ranges = build_balanced_layer_ranges(num_layers, num_stages)

    stages: List[StageSpec] = []

    for index, layer_ids in enumerate(ranges):
        stage_id = index + 1
        stages.append(
            StageSpec(
                stage_id=stage_id,
                layer_indices=layer_ids,
                include_embedding=(stage_id == 1),
                include_norm=(stage_id == num_stages),
                include_lm_head=(stage_id == num_stages),
                output_file=output_dir / f"stage_{stage_id}.pt",
            )
        )

    return ModelSplitPlan(
        model_id=model_id,
        model_source=str(model_source),
        output_dir=output_dir,
        num_stages=num_stages,
        num_layers=num_layers,
        stages=stages,
    )


def _to_cpu_eval(module: Optional[nn.Module]) -> Optional[nn.Module]:
    if module is None:
        return None
    module = copy.deepcopy(module)
    module.to(torch.device("cpu"))
    module.eval()
    return module


def _layers_to_cpu_eval(layers: Sequence[nn.Module]) -> List[nn.Module]:
    prepared: List[nn.Module] = []
    for layer in layers:
        copied = copy.deepcopy(layer)
        copied.to(torch.device("cpu"))
        copied.eval()
        prepared.append(copied)
    return prepared


def save_stage_checkpoint(
    *,
    stage_spec: StageSpec,
    embedding: nn.Module,
    decoder_layers: Sequence[nn.Module],
    norm: nn.Module,
    lm_head: nn.Module,
    model_id: str,
    num_layers: int,
    dtype_name: str,
    force: bool = False,
) -> None:
    if stage_spec.output_file.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {stage_spec.output_file}. "
            "Use --force to overwrite."
        )

    stage_spec.output_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "embedding": _to_cpu_eval(embedding) if stage_spec.include_embedding else None,
        "layers": _layers_to_cpu_eval([decoder_layers[idx] for idx in stage_spec.layer_indices]),
        "norm": _to_cpu_eval(norm) if stage_spec.include_norm else None,
        "lm_head": _to_cpu_eval(lm_head) if stage_spec.include_lm_head else None,
        "metadata": {
            "model_id": model_id,
            "stage_id": stage_spec.stage_id,
            "num_layers_total": num_layers,
            "layer_indices": stage_spec.layer_indices,
            "include_embedding": stage_spec.include_embedding,
            "include_norm": stage_spec.include_norm,
            "include_lm_head": stage_spec.include_lm_head,
            "torch_dtype": dtype_name,
            "checkpoint_format": "dli_stage_module_v1",
        },
    }

    torch.save(checkpoint, stage_spec.output_file)


def write_manifest(plan: ModelSplitPlan, output_file: Path) -> None:
    data = {
        "model_id": plan.model_id,
        "model_source": plan.model_source,
        "num_layers": plan.num_layers,
        "num_stages": plan.num_stages,
        "output_dir": str(plan.output_dir),
        "stages": [
            {
                "stage_id": stage.stage_id,
                "checkpoint": str(stage.output_file),
                "layers": stage.layer_indices,
                "include_embedding": stage.include_embedding,
                "include_norm": stage.include_norm,
                "include_lm_head": stage.include_lm_head,
            }
            for stage in plan.stages
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_stage_map_yaml(
    *,
    plan: ModelSplitPlan,
    output_file: Path,
    gateway_port: int = 8000,
    stage_port: int = 8000,
    stage_service_prefix: str = "inference-stage",
    physical_nodes: Optional[Sequence[str]] = None,
    container_model_dir: str = "/app/models",
) -> None:
    stages = []

    for index, stage in enumerate(plan.stages):
        stage_id = stage.stage_id
        next_url = None
        if stage_id < plan.num_stages:
            next_url = f"http://{stage_service_prefix}-{stage_id + 1}:{stage_port}/forward"

        physical_node = None
        if physical_nodes and index < len(physical_nodes):
            physical_node = physical_nodes[index]

        stages.append(
            {
                "stage_id": stage_id,
                "service_name": f"{stage_service_prefix}-{stage_id}",
                "physical_node": physical_node,
                "partition_file": f"{container_model_dir}/stage_{stage_id}.pt",
                "components": {
                    "embedding": stage.include_embedding,
                    "layers": stage.layer_indices,
                    "norm": stage.include_norm,
                    "lm_head": stage.include_lm_head,
                },
                "next_stage_url": next_url,
            }
        )

    data = {
        "model_name": plan.model_id,
        "num_layers": plan.num_layers,
        "inference_gateway": {
            "service_name": "inference-gateway",
            "port": gateway_port,
            "first_stage_url": f"http://{stage_service_prefix}-1:{stage_port}/forward",
        },
        "inference_stages": stages,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def copy_tokenizer_and_config(
    *,
    tokenizer: Any,
    model: nn.Module,
    output_dir: Path,
) -> None:
    tokenizer_dir = output_dir / "tokenizer"
    config_dir = output_dir / "config"

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(tokenizer_dir)
    if hasattr(model, "config"):
        model.config.save_pretrained(config_dir)


def split_model_into_stages(
    *,
    model_id: str,
    output_dir: Path,
    models_root: Path,
    num_stages: int,
    layer_ranges: Optional[Sequence[str]] = None,
    local_model_dir: Optional[Path] = None,
    dtype_name: str = "float32",
    device: str = "cpu",
    trust_remote_code: bool = False,
    force_download: bool = False,
    force: bool = False,
    copy_tokenizer: bool = True,
    manifest_file: Optional[Path] = None,
    stage_map_file: Optional[Path] = None,
    physical_nodes: Optional[Sequence[str]] = None,
) -> ModelSplitPlan:
    """
    Main high-level API used by the CLI.

    If the model is not already under models/<sanitized_model_id>, this function
    downloads it from Hugging Face into that folder. Then it writes stage_1.pt,
    stage_2.pt, ... into output_dir.
    """
    model_source = download_snapshot_if_needed(
        model_id=model_id,
        models_root=models_root,
        local_model_dir=local_model_dir,
        force_download=force_download,
    )

    tokenizer, model = load_causal_lm_and_tokenizer(
        model_source=model_source,
        dtype_name=dtype_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    embedding, decoder_layers, norm, lm_head = get_decoder_stack(model)
    num_layers = len(decoder_layers)

    plan = build_split_plan(
        model_id=model_id,
        model_source=model_source,
        output_dir=output_dir,
        num_layers=num_layers,
        num_stages=num_stages,
        layer_ranges=layer_ranges,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in plan.stages:
        save_stage_checkpoint(
            stage_spec=stage,
            embedding=embedding,
            decoder_layers=decoder_layers,
            norm=norm,
            lm_head=lm_head,
            model_id=model_id,
            num_layers=num_layers,
            dtype_name=dtype_name,
            force=force,
        )

    if copy_tokenizer:
        copy_tokenizer_and_config(tokenizer=tokenizer, model=model, output_dir=output_dir)

    if manifest_file is None:
        manifest_file = output_dir / "split_manifest.json"
    write_manifest(plan, manifest_file)

    if stage_map_file is not None:
        write_stage_map_yaml(
            plan=plan,
            output_file=stage_map_file,
            physical_nodes=physical_nodes,
        )

    return plan
