from __future__ import annotations

import os
from typing import Any, Dict

import requests
import torch
from fastapi import FastAPI, HTTPException

from dli.common.logging_config import configure_logging
from dli.common.metrics import make_stage_metric
from dli.common.schemas import HealthResponse, StageForwardRequest, StageForwardResponse
from dli.common.timing import elapsed_ms, now_ms
from dli.inference_stage.activation_codec import ActivationCodec
from dli.inference_stage.model_partition_loader import ModelPartitionLoader
from dli.inference_stage.stage_config import StageConfig, load_stage_config
from dli.inference_stage.stage_executor import StageExecutor


DEVICE = os.getenv("DEVICE", "cpu")

stage_config: StageConfig = load_stage_config()
logger = configure_logging(stage_config.service_name)

loader = ModelPartitionLoader(
    partition_file=stage_config.partition_file,
    device=DEVICE,
)

partition = loader.load()
executor = StageExecutor(partition=partition, device=DEVICE)
codec = ActivationCodec()

app = FastAPI(
    title=f"Inference Stage {stage_config.stage_id}",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service_name=stage_config.service_name,
        stage_id=stage_config.stage_id,
    )


@app.post("/forward", response_model=StageForwardResponse)
def forward(request: StageForwardRequest) -> StageForwardResponse:
    logger.info(
        "Received forward request_id=%s token_index=%s",
        request.request_id,
        request.token_index,
    )

    inbound_payload_b64_bytes = len(request.tensor_b64.encode("utf-8"))

    try:
        input_tensor = codec.decode_tensor(request.tensor_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode activation tensor: {exc}",
        ) from exc

    compute_start = now_ms()

    try:
        with torch.no_grad():
            output_tensor = executor.forward(input_tensor)
    except Exception as exc:
        logger.exception("Stage execution failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Stage execution failed: {exc}",
        ) from exc

    compute_time_ms = elapsed_ms(compute_start)
    input_tensor_bytes = int(input_tensor.numel() * input_tensor.element_size())
    output_tensor_bytes = int(output_tensor.numel() * output_tensor.element_size())

    local_metric = make_stage_metric(
        service_name=stage_config.service_name,
        stage_id=stage_config.stage_id,
        token_index=request.token_index,
        compute_time_ms=compute_time_ms,
        transfer_time_ms=0.0,
        extra={
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output_tensor.shape),
            "is_final_stage": stage_config.next_stage_url is None,
            "input_tensor_bytes": input_tensor_bytes,
            "output_tensor_bytes": output_tensor_bytes,
            "inbound_payload_b64_bytes": inbound_payload_b64_bytes,
        },
    )

    accumulated_metrics = list(request.metrics)
    accumulated_metrics.append(local_metric)

    if stage_config.next_stage_url:
        return _forward_to_next_stage(
            request=request,
            output_tensor=output_tensor,
            accumulated_metrics=accumulated_metrics,
        )

    return _finalize_response(
        request=request,
        hidden_states=output_tensor,
        accumulated_metrics=accumulated_metrics,
    )


def _forward_to_next_stage(
    *,
    request: StageForwardRequest,
    output_tensor: torch.Tensor,
    accumulated_metrics: list[Dict[str, Any]],
) -> StageForwardResponse:
    tensor_b64, tensor_dtype, tensor_shape = codec.encode_tensor(output_tensor)
    outbound_payload_b64_bytes = len(tensor_b64.encode("utf-8"))

    next_request = StageForwardRequest(
        request_id=request.request_id,
        token_index=request.token_index,
        tensor_b64=tensor_b64,
        tensor_dtype=tensor_dtype,
        tensor_shape=tensor_shape,
        metadata=request.metadata,
        metrics=accumulated_metrics,
    )

    transfer_start = now_ms()

    try:
        response = requests.post(
            stage_config.next_stage_url,
            json=next_request.model_dump(),
            timeout=float(os.getenv("STAGE_REQUEST_TIMEOUT_SECONDS", "120")),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to forward to next stage.")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to forward to next stage {stage_config.next_stage_url}: {exc}",
        ) from exc

    transfer_time_ms = elapsed_ms(transfer_start)

    response_payload = response.json()

    # Keep downstream metrics as canonical and only patch current-stage transfer time.
    response_metrics = response_payload.get("metrics", [])
    if accumulated_metrics and response_metrics:
        local_metric_index = len(accumulated_metrics) - 1
        if 0 <= local_metric_index < len(response_metrics):
            response_metrics[local_metric_index]["transfer_time_ms"] = transfer_time_ms
            response_metrics[local_metric_index]["outbound_payload_b64_bytes"] = outbound_payload_b64_bytes
            response_metrics[local_metric_index]["outbound_payload_mebibytes"] = (
                outbound_payload_b64_bytes / (1024.0 * 1024.0)
            )
            response_metrics[local_metric_index]["estimated_link_mbps"] = (
                (outbound_payload_b64_bytes * 8.0 / (transfer_time_ms / 1000.0) / 1_000_000.0)
                if transfer_time_ms > 0.0
                else 0.0
            )
            response_metrics[local_metric_index]["next_stage_url"] = stage_config.next_stage_url
    elif accumulated_metrics:
        accumulated_metrics[-1]["transfer_time_ms"] = transfer_time_ms
        accumulated_metrics[-1]["outbound_payload_b64_bytes"] = outbound_payload_b64_bytes
        accumulated_metrics[-1]["outbound_payload_mebibytes"] = (
            outbound_payload_b64_bytes / (1024.0 * 1024.0)
        )
        accumulated_metrics[-1]["estimated_link_mbps"] = (
            (outbound_payload_b64_bytes * 8.0 / (transfer_time_ms / 1000.0) / 1_000_000.0)
            if transfer_time_ms > 0.0
            else 0.0
        )
        accumulated_metrics[-1]["next_stage_url"] = stage_config.next_stage_url
        response_metrics = accumulated_metrics

    response_payload["metrics"] = response_metrics

    return StageForwardResponse(**response_payload)


def _finalize_response(
    *,
    request: StageForwardRequest,
    hidden_states: torch.Tensor,
    accumulated_metrics: list[Dict[str, Any]],
) -> StageForwardResponse:
    finalize_start = now_ms()

    try:
        logits = executor.finalize_logits(hidden_states)

        temperature = float(request.metadata.get("temperature", 0.0))
        top_k = request.metadata.get("top_k")
        top_k = int(top_k) if top_k is not None else None
        top_p = request.metadata.get("top_p")
        top_p = float(top_p) if top_p is not None else None
        min_new_tokens = int(request.metadata.get("min_new_tokens", 1))
        eos_token_id = request.metadata.get("eos_token_id")
        forbidden_token_ids = []
        # Avoid special tokens (including EOS/BOS/PAD) before minimum generation length.
        if (request.token_index + 1) < min_new_tokens:
            raw_forbidden = request.metadata.get("forbidden_token_ids_before_min") or []
            for token_id in raw_forbidden:
                try:
                    forbidden_token_ids.append(int(token_id))
                except Exception:
                    continue
            if eos_token_id is not None:
                forbidden_token_ids.append(int(eos_token_id))
            forbidden_token_ids = sorted(set(forbidden_token_ids))

        next_token_id = executor.select_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )

    except Exception as exc:
        logger.exception("Final stage failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Final stage failed: {exc}",
        ) from exc

    finalize_time_ms = elapsed_ms(finalize_start)

    final_metric = make_stage_metric(
        service_name=stage_config.service_name,
        stage_id=stage_config.stage_id,
        token_index=request.token_index,
        compute_time_ms=finalize_time_ms,
        transfer_time_ms=0.0,
        extra={
            "operation": "finalize_logits",
            "logits_shape": list(logits.shape),
            "next_token_id": next_token_id,
        },
    )

    accumulated_metrics.append(final_metric)

    return StageForwardResponse(
        request_id=request.request_id,
        token_index=request.token_index,
        next_token_id=next_token_id,
        next_token_text=None,
        metrics=accumulated_metrics,
    )
