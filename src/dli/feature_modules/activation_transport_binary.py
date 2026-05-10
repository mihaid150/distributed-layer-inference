from __future__ import annotations

import io
import json
import os
import struct
import sys
import zlib
from typing import Any, Dict, List
from urllib.parse import urlparse

import torch

from dli.common.feature_flags import FeatureFlags
from dli.common.timing import elapsed_ms, now_ms


class BinaryTransportModule:
    KEY = "binary_transport"
    WIRE_VERSION = 2
    LEGACY_WIRE_VERSION = 1
    MAGIC = b"DLI2"
    HEADER_LENGTH = struct.Struct(">I")
    _DTYPE_BY_NAME = {
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "torch.float64": torch.float64,
        "float64": torch.float64,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "int64": torch.int64,
        "torch.int32": torch.int32,
        "int32": torch.int32,
        "torch.int16": torch.int16,
        "int16": torch.int16,
        "torch.int8": torch.int8,
        "int8": torch.int8,
        "torch.uint8": torch.uint8,
        "uint8": torch.uint8,
        "torch.bool": torch.bool,
        "bool": torch.bool,
    }

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
    def encode_tensor_blob(tensor: torch.Tensor) -> tuple[bytes, Dict[str, Any]]:
        tensor = tensor.detach().cpu().contiguous()
        raw = BinaryTransportModule._tensor_to_bytes(tensor)
        metadata: Dict[str, Any] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "byte_order": sys.byteorder,
            "tensor_bytes": len(raw),
        }
        return raw, metadata

    @staticmethod
    def decode_tensor_blob(raw: bytes, metadata: Dict[str, Any]) -> torch.Tensor:
        dtype = BinaryTransportModule._dtype_from_name(str(metadata.get("dtype", "")))
        shape = BinaryTransportModule._shape_from_metadata(metadata.get("shape"))

        if dtype == torch.bfloat16:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone()
            return tensor.view(torch.bfloat16).reshape(shape)

        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        return tensor.reshape(shape)

    @staticmethod
    def pack_request(
        *,
        request_dict: Dict[str, Any],
        tensor_blob: bytes,
        tensor_metadata: Dict[str, Any] | None = None,
    ) -> bytes:
        metadata = dict(tensor_metadata or {})
        metadata.setdefault("dtype", request_dict.get("tensor_dtype"))
        metadata.setdefault("shape", request_dict.get("tensor_shape"))
        metadata.setdefault("byte_order", sys.byteorder)
        metadata.setdefault("tensor_bytes", len(tensor_blob))
        header = {
            "wire_version": BinaryTransportModule.WIRE_VERSION,
            "kind": "request",
            "request": request_dict,
            "tensor": metadata,
        }
        return BinaryTransportModule._pack_v2(header=header, body=tensor_blob)

    @staticmethod
    def unpack_request(raw: bytes) -> Dict[str, Any]:
        if raw.startswith(BinaryTransportModule.MAGIC):
            header, body = BinaryTransportModule._unpack_v2(raw)
            if header.get("kind") != "request":
                raise TypeError("Binary v2 envelope is not a request.")
            request = header.get("request")
            if not isinstance(request, dict):
                raise TypeError("Binary v2 request envelope is missing request payload.")
            return {
                "wire_version": BinaryTransportModule.WIRE_VERSION,
                "request": request,
                "tensor_blob": body,
                "tensor": header.get("tensor") or {},
            }

        buffer = io.BytesIO(raw)
        envelope = torch.load(buffer, map_location="cpu")
        if not isinstance(envelope, dict):
            raise TypeError("Binary request envelope must be a dict.")
        if envelope.get("wire_version") != BinaryTransportModule.LEGACY_WIRE_VERSION:
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
        header = {
            "wire_version": BinaryTransportModule.WIRE_VERSION,
            "kind": "response",
            "response": response_dict,
        }
        return BinaryTransportModule._pack_v2(header=header, body=b"")

    @staticmethod
    def unpack_response(raw: bytes) -> Dict[str, Any]:
        if raw.startswith(BinaryTransportModule.MAGIC):
            header, body = BinaryTransportModule._unpack_v2(raw)
            if body:
                raise TypeError("Binary v2 response envelope unexpectedly contains a body.")
            if header.get("kind") != "response":
                raise TypeError("Binary v2 envelope is not a response.")
            response = header.get("response")
            if not isinstance(response, dict):
                raise TypeError("Binary v2 response envelope is missing response payload.")
            return response

        buffer = io.BytesIO(raw)
        envelope = torch.load(buffer, map_location="cpu")
        if not isinstance(envelope, dict):
            raise TypeError("Binary response envelope must be a dict.")
        if envelope.get("wire_version") != BinaryTransportModule.LEGACY_WIRE_VERSION:
            raise ValueError(
                f"Unsupported binary wire version: {envelope.get('wire_version')}"
            )
        response = envelope.get("response")
        if not isinstance(response, dict):
            raise TypeError("Binary response envelope is missing 'response' dict.")
        return response

    @staticmethod
    def _pack_v2(*, header: Dict[str, Any], body: bytes) -> bytes:
        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return (
            BinaryTransportModule.MAGIC
            + BinaryTransportModule.HEADER_LENGTH.pack(len(header_bytes))
            + header_bytes
            + body
        )

    @staticmethod
    def _unpack_v2(raw: bytes) -> tuple[Dict[str, Any], bytes]:
        offset = len(BinaryTransportModule.MAGIC)
        end_length = offset + BinaryTransportModule.HEADER_LENGTH.size
        if len(raw) < end_length:
            raise ValueError("Binary v2 envelope is truncated before header length.")
        header_length = BinaryTransportModule.HEADER_LENGTH.unpack(raw[offset:end_length])[0]
        header_start = end_length
        header_end = header_start + header_length
        if len(raw) < header_end:
            raise ValueError("Binary v2 envelope is truncated before header body.")
        header = json.loads(raw[header_start:header_end].decode("utf-8"))
        if not isinstance(header, dict):
            raise TypeError("Binary v2 envelope header must be a dict.")
        if header.get("wire_version") != BinaryTransportModule.WIRE_VERSION:
            raise ValueError(
                f"Unsupported binary wire version: {header.get('wire_version')}"
            )
        return header, raw[header_end:]

    @staticmethod
    def _dtype_from_name(dtype_name: str) -> torch.dtype:
        try:
            return BinaryTransportModule._DTYPE_BY_NAME[dtype_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported tensor dtype for binary transport: {dtype_name}") from exc

    @staticmethod
    def _shape_from_metadata(raw_shape: Any) -> List[int]:
        if not isinstance(raw_shape, list):
            raise ValueError("Binary tensor metadata is missing shape list.")
        shape = [int(item) for item in raw_shape]
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape in binary metadata: {shape}")
        return shape

    @staticmethod
    def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy().tobytes()
        return tensor.numpy().tobytes()
