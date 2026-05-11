from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    service_name: str = "inference-gateway"
    first_stage_url: str = Field(
        default="http://inference-stage-1:8000/forward",
        description="Internal URL of the first Inference Stage.",
    )
    model_name: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Tokenizer model name.",
    )
    topology_route_candidates: Dict[str, List[str]] = Field(default_factory=dict)
    topology_probe_timeout_seconds: float = Field(default=0.25, ge=0.01, le=5.0)
