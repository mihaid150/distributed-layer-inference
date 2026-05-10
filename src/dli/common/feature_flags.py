from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TransportMode = Literal["json_base64", "binary_octet_stream"]
ActivationPrecision = Literal["fp32", "fp16", "bf16", "int8"]
RebalanceProfile = Literal["baseline", "latency_balanced_v1"]
PayloadCompression = Literal["none", "zlib"]


class FeatureFlags(BaseModel):
    """
    Runtime-selectable optimization modules.

    Defaults keep baseline behavior unchanged.
    """

    transport_mode: TransportMode = Field(
        default="json_base64",
        description="Activation transport between stages.",
    )
    activation_precision: ActivationPrecision = Field(
        default="fp32",
        description=(
            "Tensor precision/compression for inter-stage activations. FP16 is the"
            " optimized module-profile default; INT8 is experimental."
        ),
    )
    payload_compression: PayloadCompression = Field(
        default="none",
        description="Optional lightweight compression for binary activation payloads.",
    )
    kv_cache_enabled: bool = Field(
        default=False,
        description="Enable stage-side caching module (dedupe/retry cache).",
    )
    rebalance_profile: RebalanceProfile = Field(
        default="baseline",
        description="Partition profile selector.",
    )
    topology_aware_routing: bool = Field(
        default=False,
        description="Enable topology-aware next-hop routing policy.",
    )
    persistent_sessions_enabled: bool = Field(
        default=False,
        description="Reuse persistent HTTP sessions for stage calls.",
    )
    backpressure_enabled: bool = Field(
        default=False,
        description="Enable gateway single-flight backpressure guard.",
    )
    backpressure_queue_size: int = Field(
        default=0,
        ge=0,
        le=256,
        description=(
            "Gateway backpressure queue capacity. 0 means reject concurrent requests"
            " when module is active."
        ),
    )

    @staticmethod
    def optimized_module_profile() -> dict[str, object]:
        return {
            "transport_mode": "binary_octet_stream",
            "activation_precision": "fp16",
            "payload_compression": "none",
            "persistent_sessions_enabled": True,
        }

    def enabled_module_keys(self) -> list[str]:
        keys: list[str] = []
        if self.transport_mode != "json_base64":
            keys.append("binary_transport")
        if self.activation_precision != "fp32":
            keys.append("activation_precision")
        if self.payload_compression != "none":
            keys.append("payload_compression")
        if self.kv_cache_enabled:
            keys.append("kv_cache")
        if self.rebalance_profile != "baseline":
            keys.append("rebalance")
        if self.topology_aware_routing:
            keys.append("topology_aware")
        if self.persistent_sessions_enabled or self.backpressure_enabled:
            keys.append("persistent_sessions_backpressure")
        return keys
