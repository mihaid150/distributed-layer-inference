from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests
import torch

from dli.common.feature_flags import FeatureFlags
from dli.common.schemas import StageForwardRequest, StageForwardResponse
from dli.common.timing import elapsed_ms, now_ms
from dli.feature_modules.activation_payload_precision import ActivationPayloadPrecisionModule
from dli.feature_modules.activation_transport_binary import BinaryTransportModule
from dli.feature_modules.persistent_backpressure import PersistentSessionPool
from dli.inference_stage.activation_codec import ActivationCodec


@dataclass
class StageForwardResult:
    response: StageForwardResponse
    transport: dict


class StageClient:
    def __init__(self, first_stage_url: str, timeout_seconds: Optional[float] = None) -> None:
        self.first_stage_url = first_stage_url
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("STAGE_REQUEST_TIMEOUT_SECONDS", "120")
        )
        self.codec = ActivationCodec()

    def forward(self, request: StageForwardRequest) -> StageForwardResponse:
        return self.forward_with_transport(request).response

    def _post(
        self,
        *,
        url: str,
        flags: FeatureFlags,
        json_payload: Optional[dict] = None,
        binary_payload: Optional[bytes] = None,
    ) -> requests.Response:
        kwargs = {"timeout": self.timeout_seconds}
        if json_payload is not None:
            kwargs["json"] = json_payload
        if binary_payload is not None:
            kwargs["data"] = binary_payload
            kwargs["headers"] = {"Content-Type": "application/octet-stream"}

        if PersistentSessionPool.use_persistent_session(flags):
            session = PersistentSessionPool.get_or_create(self.first_stage_url)
            return session.post(url, **kwargs)
        return requests.post(url, **kwargs)

    def forward_with_transport(
        self,
        request: StageForwardRequest,
        *,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> StageForwardResult:
        flags = request.feature_flags
        effective_input_tensor = input_tensor
        if effective_input_tensor is None:
            effective_input_tensor = self.codec.decode_tensor(request.tensor_b64)

        precision_start_ms = now_ms()
        encoded_tensor, compression_meta = ActivationPayloadPrecisionModule.prepare_tensor(
            effective_input_tensor,
            flags,
        )
        precision_cast_ms = elapsed_ms(precision_start_ms)

        transport_encoding = "json_base64"
        target_url = self.first_stage_url
        encode_ms = 0.0
        pack_ms = 0.0
        http_roundtrip_ms = 0.0
        unpack_ms = 0.0
        decode_ms = 0.0
        compression_time_ms = 0.0
        decompression_time_ms = 0.0
        tensor_wire_bytes = 0
        payload_compression_meta: dict = {"mode": "none", "applied": False}

        if BinaryTransportModule.is_enabled(flags):
            transport_encoding = "binary_octet_stream"
            target_url = BinaryTransportModule.resolve_forward_url(self.first_stage_url)
            encode_start_ms = now_ms()
            tensor_blob, tensor_metadata = BinaryTransportModule.encode_tensor_blob(encoded_tensor)
            encode_ms = elapsed_ms(encode_start_ms)
            tensor_blob, payload_compression_meta = BinaryTransportModule.prepare_tensor_blob(
                tensor_blob=tensor_blob,
                flags=flags,
            )
            compression_time_ms = float(payload_compression_meta.get("compression_time_ms", 0.0))
            tensor_wire_bytes = len(tensor_blob)
            request_payload = request.model_dump()
            request_payload["tensor_b64"] = ""
            request_payload["tensor_dtype"] = str(encoded_tensor.dtype)
            request_payload["tensor_shape"] = list(encoded_tensor.shape)
            request_payload["transport"] = {
                "encoding": transport_encoding,
                "compression": compression_meta,
                "payload_compression": payload_compression_meta,
            }

            pack_start_ms = now_ms()
            packed = BinaryTransportModule.pack_request(
                request_dict=request_payload,
                tensor_blob=tensor_blob,
                tensor_metadata=tensor_metadata,
            )
            pack_ms = elapsed_ms(pack_start_ms)
            rpc_start_ms = now_ms()
            response = self._post(
                url=target_url,
                flags=flags,
                binary_payload=packed,
            )
            http_roundtrip_ms = elapsed_ms(rpc_start_ms)
            request_payload_bytes = len(packed)
        else:
            encode_start_ms = now_ms()
            tensor_b64, tensor_dtype, tensor_shape = self.codec.encode_tensor(encoded_tensor)
            encode_ms = elapsed_ms(encode_start_ms)
            tensor_wire_bytes = len(tensor_b64.encode("utf-8"))
            request_payload = request.model_dump()
            request_payload["tensor_b64"] = tensor_b64
            request_payload["tensor_dtype"] = tensor_dtype
            request_payload["tensor_shape"] = tensor_shape
            request_payload["transport"] = {
                "encoding": transport_encoding,
                "compression": compression_meta,
                "payload_compression": {"mode": "none", "applied": False},
            }
            pack_start_ms = now_ms()
            packed_json = json.dumps(
                request_payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            pack_ms = elapsed_ms(pack_start_ms)
            request_payload_bytes = len(packed_json)
            rpc_start_ms = now_ms()
            response = self._post(
                url=target_url,
                flags=flags,
                json_payload=request_payload,
            )
            http_roundtrip_ms = elapsed_ms(rpc_start_ms)

        response.raise_for_status()
        response_payload_bytes = len(response.content)

        unpack_start_ms = now_ms()
        if transport_encoding == "binary_octet_stream":
            stage_response = StageForwardResponse(
                **BinaryTransportModule.unpack_response(response.content)
            )
        else:
            stage_response = StageForwardResponse(**response.json())
        unpack_ms = elapsed_ms(unpack_start_ms)

        remote_server_wall_ms = float(getattr(stage_response, "server_wall_ms", 0.0) or 0.0)
        rpc_residual_ms = max(0.0, http_roundtrip_ms - remote_server_wall_ms)
        true_comm_ms = (
            precision_cast_ms
            + encode_ms
            + pack_ms
            + compression_time_ms
            + rpc_residual_ms
            + unpack_ms
            + decode_ms
            + decompression_time_ms
        )

        transport = {
            "url": target_url,
            "http_status": response.status_code,
            "encoding": transport_encoding,
            "request_payload_bytes": request_payload_bytes,
            "response_payload_bytes": response_payload_bytes,
            "request_wire_bytes": request_payload_bytes,
            "response_wire_bytes": response_payload_bytes,
            "tensor_wire_bytes": tensor_wire_bytes,
            "transfer_time_ms": http_roundtrip_ms,
            "rpc_wall_time_ms": http_roundtrip_ms,
            "http_roundtrip_ms": http_roundtrip_ms,
            "remote_server_wall_ms": remote_server_wall_ms,
            "rpc_residual_ms": rpc_residual_ms,
            "true_comm_ms": true_comm_ms,
            "precision_cast_ms": precision_cast_ms,
            "encode_ms": encode_ms,
            "pack_ms": pack_ms,
            "compression_ms": compression_time_ms,
            "unpack_ms": unpack_ms,
            "decode_ms": decode_ms,
            "decompression_ms": decompression_time_ms,
            "estimated_link_mbps": (
                (
                    (request_payload_bytes + response_payload_bytes)
                    * 8.0
                    / (http_roundtrip_ms / 1000.0)
                    / 1_000_000.0
                )
                if http_roundtrip_ms > 0.0
                else 0.0
            ),
            "payload_compression": request_payload.get("transport", {}).get(
                "payload_compression",
                {"mode": "none", "applied": False},
            ),
            "persistent_session_pool": PersistentSessionPool.snapshot(),
        }

        return StageForwardResult(response=stage_response, transport=transport)
