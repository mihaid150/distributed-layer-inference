from __future__ import annotations

import base64
import io
from typing import List, Tuple

import torch


class ActivationCodec:
    """
    Encodes and decodes intermediate activations between Inference Stages.

    For the first prototype, torch.save + base64 is simple and robust.
    Later, this can be replaced with gRPC binary payloads, Apache Arrow,
    msgpack, or safetensors-style binary transport.
    """

    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> Tuple[str, str, List[int]]:
        tensor = tensor.detach().cpu()
        raw = ActivationCodec.encode_tensor_raw(tensor)
        encoded = base64.b64encode(raw).decode("utf-8")
        return encoded, str(tensor.dtype), list(tensor.shape)

    @staticmethod
    def decode_tensor(tensor_b64: str) -> torch.Tensor:
        raw = base64.b64decode(tensor_b64.encode("utf-8"))
        return ActivationCodec.decode_tensor_raw(raw)

    @staticmethod
    def encode_tensor_raw(tensor: torch.Tensor) -> bytes:
        tensor = tensor.detach().cpu()
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    @staticmethod
    def decode_tensor_raw(raw: bytes) -> torch.Tensor:
        buffer = io.BytesIO(raw)
        tensor = torch.load(buffer, map_location="cpu")

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Decoded object is not a torch.Tensor.")

        return tensor