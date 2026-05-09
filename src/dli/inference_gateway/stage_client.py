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

        encoded_tensor, compression_meta = ActivationPayloadPrecisionModule.prepare_tensor(
            effective_input_tensor,
            flags,
        )

        transfer_start_ms = now_ms()
        transport_encoding = "json_base64"
        target_url = self.first_stage_url

        if BinaryTransportModule.is_enabled(flags):
            transport_encoding = "binary_octet_stream"
            target_url = BinaryTransportModule.resolve_forward_url(self.first_stage_url)
            tensor_blob = self.codec.encode_tensor_raw(encoded_tensor)
            request_payload = request.model_dump()
            request_payload["tensor_b64"] = ""
            request_payload["tensor_dtype"] = str(encoded_tensor.dtype)
            request_payload["tensor_shape"] = list(encoded_tensor.shape)
            request_payload["transport"] = {
                "encoding": transport_encoding,
                "compression": compression_meta,
            }

            packed = BinaryTransportModule.pack_request(
                request_dict=request_payload,
                tensor_blob=tensor_blob,
            )
            response = self._post(
                url=target_url,
                flags=flags,
                binary_payload=packed,
            )
            request_payload_bytes = len(packed)
        else:
            tensor_b64, tensor_dtype, tensor_shape = self.codec.encode_tensor(encoded_tensor)
            request_payload = request.model_dump()
            request_payload["tensor_b64"] = tensor_b64
            request_payload["tensor_dtype"] = tensor_dtype
            request_payload["tensor_shape"] = tensor_shape
            request_payload["transport"] = {
                "encoding": transport_encoding,
                "compression": compression_meta,
            }
            request_payload_bytes = len(
                json.dumps(
                    request_payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            response = self._post(
                url=target_url,
                flags=flags,
                json_payload=request_payload,
            )

        response.raise_for_status()
        transfer_time_ms = elapsed_ms(transfer_start_ms)
        response_payload_bytes = len(response.content)

        if transport_encoding == "binary_octet_stream":
            stage_response = StageForwardResponse(
                **BinaryTransportModule.unpack_response(response.content)
            )
        else:
            stage_response = StageForwardResponse(**response.json())

        transport = {
            "url": target_url,
            "http_status": response.status_code,
            "encoding": transport_encoding,
            "request_payload_bytes": request_payload_bytes,
            "response_payload_bytes": response_payload_bytes,
            "transfer_time_ms": transfer_time_ms,
            "estimated_link_mbps": (
                ((request_payload_bytes + response_payload_bytes) * 8.0 / (transfer_time_ms / 1000.0) / 1_000_000.0)
                if transfer_time_ms > 0.0
                else 0.0
            ),
        }

        return StageForwardResult(response=stage_response, transport=transport)
