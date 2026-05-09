from __future__ import annotations

import io
from typing import Any, Dict
from urllib.parse import urlparse

import torch

from dli.common.feature_flags import FeatureFlags


class BinaryTransportModule:
    KEY = "binary_transport"
    WIRE_VERSION = 1

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.transport_mode == "binary_octet_stream"

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

