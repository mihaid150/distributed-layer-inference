from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn


@dataclass
class ModelPartition:
    embedding: Optional[nn.Module]
    layers: List[nn.Module]
    norm: Optional[nn.Module]
    lm_head: Optional[nn.Module]


class ModelPartitionLoader:
    """
    Loads a stage-specific model partition.

    Expected checkpoint format:

    {
        "embedding": optional nn.Module,
        "layers": list[nn.Module],
        "norm": optional nn.Module,
        "lm_head": optional nn.Module,
        "metadata": optional dict
    }

    For a real paper implementation, the splitting script should generate
    stage_1.pt, stage_2.pt, stage_3.pt, stage_4.pt using this format.
    """

    def __init__(self, partition_file: str, device: str = "cpu") -> None:
        self.partition_file = partition_file
        self.device = torch.device(device)

    def load(self) -> ModelPartition:
        checkpoint = torch.load(
            self.partition_file,
            map_location=self.device,
            weights_only=False,
        )

        embedding = checkpoint.get("embedding")
        layers = checkpoint.get("layers", [])
        norm = checkpoint.get("norm")
        lm_head = checkpoint.get("lm_head")

        if embedding is not None:
            embedding = embedding.to(self.device)
            embedding.eval()

        prepared_layers: List[nn.Module] = []
        for layer in layers:
            layer = layer.to(self.device)
            layer.eval()
            prepared_layers.append(layer)

        if norm is not None:
            norm = norm.to(self.device)
            norm.eval()

        if lm_head is not None:
            lm_head = lm_head.to(self.device)
            lm_head.eval()

        return ModelPartition(
            embedding=embedding,
            layers=prepared_layers,
            norm=norm,
            lm_head=lm_head,
        )