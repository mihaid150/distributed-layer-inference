from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from dli.common.feature_flags import FeatureFlags


class ActivationPayloadPrecisionModule:
    KEY = "activation_precision"

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.activation_precision != "fp32"

    @staticmethod
    def prepare_tensor(
        tensor: torch.Tensor,
        flags: FeatureFlags,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mode = flags.activation_precision
        base_metadata: Dict[str, Any] = {
            "mode": mode,
            "source_dtype": str(tensor.dtype),
        }

        if not tensor.dtype.is_floating_point or mode == "fp32":
            base_metadata["applied"] = False
            return tensor, base_metadata

        if mode == "fp16":
            base_metadata["applied"] = True
            return tensor.to(torch.float16), base_metadata

        if mode == "bf16":
            base_metadata["applied"] = True
            return tensor.to(torch.bfloat16), base_metadata

        if mode == "int8":
            tensor_fp32 = tensor.to(torch.float32)
            max_abs = float(tensor_fp32.abs().max().item())
            scale = max(max_abs / 127.0, 1e-8)
            quantized = torch.clamp(torch.round(tensor_fp32 / scale), -127, 127).to(
                torch.int8
            )
            base_metadata.update(
                {
                    "applied": True,
                    "scale": scale,
                    "restored_dtype": str(tensor.dtype),
                }
            )
            return quantized, base_metadata

        base_metadata["applied"] = False
        return tensor, base_metadata

    @staticmethod
    def restore_tensor(tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        mode = str((metadata or {}).get("mode") or "fp32")
        applied = bool((metadata or {}).get("applied", False))
        if not applied:
            return tensor

        if mode == "int8":
            scale = float((metadata or {}).get("scale", 1.0))
            return tensor.to(torch.float32) * scale

        if mode == "fp16":
            return tensor.to(
                ActivationPayloadPrecisionModule._dtype_from_name(
                    str((metadata or {}).get("source_dtype") or "torch.float32")
                )
            )

        if mode == "bf16":
            return tensor.to(
                ActivationPayloadPrecisionModule._dtype_from_name(
                    str((metadata or {}).get("source_dtype") or "torch.float32")
                )
            )

        return tensor

    @staticmethod
    def _dtype_from_name(dtype_name: str) -> torch.dtype:
        normalized = dtype_name.replace("torch.", "").lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float64", "double"}:
            return torch.float64
        return torch.float32
