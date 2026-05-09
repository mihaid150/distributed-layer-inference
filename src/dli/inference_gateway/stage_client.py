from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests

from dli.common.schemas import StageForwardRequest, StageForwardResponse
from dli.common.timing import elapsed_ms, now_ms


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

    def forward(self, request: StageForwardRequest) -> StageForwardResponse:
        return self.forward_with_transport(request).response

    def forward_with_transport(self, request: StageForwardRequest) -> StageForwardResult:
        request_payload = request.model_dump()
        request_payload_bytes = len(
            json.dumps(request_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        transfer_start_ms = now_ms()
        response = requests.post(
            self.first_stage_url,
            json=request_payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        transfer_time_ms = elapsed_ms(transfer_start_ms)
        response_payload_bytes = len(response.content)

        stage_response = StageForwardResponse(**response.json())
        transport = {
            "url": self.first_stage_url,
            "http_status": response.status_code,
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
