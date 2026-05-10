from __future__ import annotations

import io
import os
import zlib
from typing import Any, Dict
from urllib.parse import urlparse

import torch

from dli.common.feature_flags import FeatureFlags
from dli.common.timing import elapsed_ms, now_ms


class BinaryTransportModule:
    KEY = "binary_transport"
    WIRE_VERSION = 1

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.transport_mode == "binary_octet_stream"

    @staticmethod
    def compression_mode(flags: FeatureFlags) -> str:
        if not BinaryTransportModule.is_enabled(flags):
            return "none"
        return str(getattr(flags, "payload_compression", "none") or "none")

    @staticmethod
    def resolve_forward_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        path = parsed.path or "/forward"
        if path.endswith("/forward-binary"):
            next_path = path
        elif path.endswith("/forward"):
            next_path = f"{path}-binary"
        else:
            next_path = "/forward-binary"
        return parsed._replace(path=next_path).geturl()

    @staticmethod
    def pack_request(*, request_dict: Dict[str, Any], tensor_blob: bytes) -> bytes:
        envelope = {
            "wire_version": BinaryTransportModule.WIRE_VERSION,
            "request": request_dict,
            "tensor_blob": tensor_blob,
        }
        buffer = io.BytesIO()
        torch.save(envelope, buffer)
        return buffer.getvalue()

    @staticmethod
    def unpack_request(raw: bytes) -> Dict[str, Any]:
        buffer = io.BytesIO(raw)
        envelope = torch.load(buffer, map_location="cpu")
        if not isinstance(envelope, dict):
            raise TypeError("Binary request envelope must be a dict.")
        if envelope.get("wire_version") != BinaryTransportModule.WIRE_VERSION:
            raise ValueError(
                f"Unsupported binary wire version: {envelope.get('wire_version')}"
            )
        return envelope

    @staticmethod
    def prepare_tensor_blob(
        *,
        tensor_blob: bytes,
        flags: FeatureFlags,
    ) -> tuple[bytes, Dict[str, Any]]:
        mode = BinaryTransportModule.compression_mode(flags)
        metadata: Dict[str, Any] = {
            "mode": mode,
            "applied": False,
            "original_bytes": len(tensor_blob),
            "wire_bytes": len(tensor_blob),
            "compression_time_ms": 0.0,
            "compression_ratio": 1.0,
        }

        if mode == "none":
            return tensor_blob, metadata

        if mode != "zlib":
            metadata["skipped_reason"] = f"unsupported_mode:{mode}"
            return tensor_blob, metadata

        compression_start = now_ms()
        compressed = zlib.compress(tensor_blob, level=1)
        compression_time_ms = elapsed_ms(compression_start)
        metadata["compression_time_ms"] = compression_time_ms

        saved_bytes = len(tensor_blob) - len(compressed)
        min_saved_bytes = int(os.getenv("DLI_PAYLOAD_COMPRESSION_MIN_SAVED_BYTES", "4096"))

        if len(compressed) >= len(tensor_blob):
            metadata["skipped_reason"] = "compressed_payload_not_smaller"
            return tensor_blob, metadata

        if saved_bytes < min_saved_bytes:
            metadata["skipped_reason"] = "saved_bytes_below_threshold"
            metadata["saved_bytes"] = saved_bytes
            return tensor_blob, metadata

        metadata.update(
            {
                "applied": True,
                "wire_bytes": len(compressed),
                "saved_bytes": saved_bytes,
                "compression_ratio": len(compressed) / max(1, len(tensor_blob)),
            }
        )
        return compressed, metadata

    @staticmethod
    def restore_tensor_blob(*, tensor_blob: bytes, metadata: Dict[str, Any]) -> bytes:
        mode = str((metadata or {}).get("mode") or "none")
        applied = bool((metadata or {}).get("applied", False))

        if not applied:
            return tensor_blob

        if mode == "zlib":
            return zlib.decompress(tensor_blob)

        raise ValueError(f"Unsupported binary payload compression mode: {mode}")

    @staticmethod
    def pack_response(response_dict: Dict[str, Any]) -> bytes:
        envelope = {
            "wire_version": BinaryTransportModule.WIRE_VERSION,
            "response": response_dict,
        }
        buffer = io.BytesIO()
        torch.save(envelope, buffer)
        return buffer.getvalue()

    @staticmethod
    def unpack_response(raw: bytes) -> Dict[str, Any]:
        buffer = io.BytesIO(raw)
        envelope = torch.load(buffer, map_location="cpu")
        if not isinstance(envelope, dict):
            raise TypeError("Binary response envelope must be a dict.")
        if envelope.get("wire_version") != BinaryTransportModule.WIRE_VERSION:
            raise ValueError(
                f"Unsupported binary wire version: {envelope.get('wire_version')}"
            )
        response = envelope.get("response")
        if not isinstance(response, dict):
            raise TypeError("Binary response envelope is missing 'response' dict.")
        return response
