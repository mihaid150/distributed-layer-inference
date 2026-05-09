from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests
import torch
from fastapi import FastAPI, HTTPException, Request, Response

from dli.common.logging_config import configure_logging
from dli.common.metrics import make_stage_metric
from dli.common.schemas import HealthResponse, StageForwardRequest, StageForwardResponse
from dli.common.timing import elapsed_ms, now_ms
from dli.feature_modules.activation_payload_precision import ActivationPayloadPrecisionModule
from dli.feature_modules.activation_transport_binary import BinaryTransportModule
from dli.feature_modules.kv_cache_stage import StageForwardCache
from dli.feature_modules.partition_rebalance import PartitionRebalanceModule
from dli.feature_modules.persistent_backpressure import PersistentSessionPool
from dli.feature_modules.topology_aware_routing import TopologyAwareRoutingModule
from dli.inference_stage.activation_codec import ActivationCodec
from dli.inference_stage.model_partition_loader import ModelPartitionLoader
from dli.inference_stage.stage_config import StageConfig, load_stage_config
from dli.inference_stage.stage_executor import StageExecutor


DEVICE = os.getenv("DEVICE", "cpu")
STAGE_REQUEST_TIMEOUT_SECONDS = float(os.getenv("STAGE_REQUEST_TIMEOUT_SECONDS", "120"))
STAGE_CACHE_ENTRIES = int(os.getenv("STAGE_KV_CACHE_MAX_ENTRIES", "64"))

stage_config: StageConfig = load_stage_config()
logger = configure_logging(stage_config.service_name)

loader = ModelPartitionLoader(
    partition_file=stage_config.partition_file,
    device=DEVICE,
)

partition = loader.load()
executor = StageExecutor(partition=partition, device=DEVICE)
codec = ActivationCodec()
stage_forward_cache = StageForwardCache(max_entries=STAGE_CACHE_ENTRIES)

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
        "Received forward request_id=%s token_index=%s encoding=%s",
        request.request_id,
        request.token_index,
        request.transport.get("encoding", "json_base64"),
    )

    inbound_payload_b64_bytes = len(request.tensor_b64.encode("utf-8"))
    try:
        input_tensor = codec.decode_tensor(request.tensor_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode activation tensor: {exc}",
        ) from exc

    compression_meta = request.transport.get("compression", {})
    input_tensor = ActivationPayloadPrecisionModule.restore_tensor(
        input_tensor,
        compression_meta,
    )

    return _run_forward_flow(
        request=request,
        input_tensor=input_tensor,
        inbound_payload_bytes=inbound_payload_b64_bytes,
    )


@app.post("/forward-binary")
async def forward_binary(raw_request: Request) -> Response:
    raw_body = await raw_request.body()
    inbound_payload_bytes = len(raw_body)

    try:
        envelope = BinaryTransportModule.unpack_request(raw_body)
        request_dict = envelope.get("request")
        tensor_blob = envelope.get("tensor_blob")
        if not isinstance(request_dict, dict):
            raise TypeError("Binary envelope is missing request payload.")
        if not isinstance(tensor_blob, (bytes, bytearray)):
            raise TypeError("Binary envelope is missing tensor blob.")

        request = StageForwardRequest(**request_dict)
        input_tensor = codec.decode_tensor_raw(bytes(tensor_blob))
        compression_meta = request.transport.get("compression", {})
        input_tensor = ActivationPayloadPrecisionModule.restore_tensor(
            input_tensor,
            compression_meta,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse binary forward request: {exc}",
        ) from exc

    response_model = _run_forward_flow(
        request=request,
        input_tensor=input_tensor,
        inbound_payload_bytes=inbound_payload_bytes,
    )
    packed_response = BinaryTransportModule.pack_response(response_model.model_dump())
    return Response(content=packed_response, media_type="application/octet-stream")


def _run_forward_flow(
    *,
    request: StageForwardRequest,
    input_tensor: torch.Tensor,
    inbound_payload_bytes: int,
) -> StageForwardResponse:
    feature_flags = request.feature_flags

    compute_start = now_ms()
    cache_hit = False
    cache_key = None

    if StageForwardCache.is_enabled(feature_flags):
        cache_key = StageForwardCache.build_key(
            request_id=request.request_id,
            token_index=request.token_index,
            tensor=input_tensor,
        )
        cached = stage_forward_cache.get(cache_key)
        if cached is not None:
            cache_hit = True
            output_tensor = cached
        else:
            output_tensor = _execute_stage_forward(input_tensor)
            stage_forward_cache.put(cache_key, output_tensor)
    else:
        output_tensor = _execute_stage_forward(input_tensor)

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
            "inbound_payload_b64_bytes": inbound_payload_bytes,
            "inbound_payload_bytes": inbound_payload_bytes,
            "kv_cache_enabled": feature_flags.kv_cache_enabled,
            "kv_cache_hit": cache_hit,
            "kv_cache_entries": stage_forward_cache.size() if feature_flags.kv_cache_enabled else 0,
            "rebalance_profile": PartitionRebalanceModule.profile(feature_flags),
            "transport_encoding": request.transport.get("encoding", "json_base64"),
            "feature_modules": feature_flags.enabled_module_keys(),
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


def _execute_stage_forward(input_tensor: torch.Tensor) -> torch.Tensor:
    try:
        with torch.no_grad():
            return executor.forward(input_tensor)
    except Exception as exc:
        logger.exception("Stage execution failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Stage execution failed: {exc}",
        ) from exc


def _forward_to_next_stage(
    *,
    request: StageForwardRequest,
    output_tensor: torch.Tensor,
    accumulated_metrics: list[Dict[str, Any]],
) -> StageForwardResponse:
    feature_flags = request.feature_flags
    next_stage_url = TopologyAwareRoutingModule.resolve_next_stage_url(
        stage_id=stage_config.stage_id,
        default_next_stage_url=stage_config.next_stage_url,
        metadata=request.metadata,
    )
    if not next_stage_url:
        raise HTTPException(
            status_code=500,
            detail="Next stage URL is not configured for non-final stage.",
        )

    encoded_tensor, compression_meta = ActivationPayloadPrecisionModule.prepare_tensor(
        output_tensor,
        feature_flags,
    )
    transfer_start = now_ms()
    if PersistentSessionPool.use_persistent_session(feature_flags):
        stage_http_client = PersistentSessionPool.get_or_create(next_stage_url)
    else:
        stage_http_client = requests

    try:
        if BinaryTransportModule.is_enabled(feature_flags):
            tensor_blob = codec.encode_tensor_raw(encoded_tensor)
            next_request = StageForwardRequest(
                request_id=request.request_id,
                token_index=request.token_index,
                tensor_b64="",
                tensor_dtype=str(encoded_tensor.dtype),
                tensor_shape=list(encoded_tensor.shape),
                metadata=request.metadata,
                metrics=accumulated_metrics,
                transport={
                    "encoding": "binary_octet_stream",
                    "compression": compression_meta,
                },
                feature_flags=feature_flags,
            )
            packed = BinaryTransportModule.pack_request(
                request_dict=next_request.model_dump(),
                tensor_blob=tensor_blob,
            )
            response = stage_http_client.post(
                BinaryTransportModule.resolve_forward_url(next_stage_url),
                data=packed,
                headers={"Content-Type": "application/octet-stream"},
                timeout=STAGE_REQUEST_TIMEOUT_SECONDS,
            )
            request_payload_bytes = len(packed)
        else:
            tensor_b64, tensor_dtype, tensor_shape = codec.encode_tensor(encoded_tensor)
            next_request = StageForwardRequest(
                request_id=request.request_id,
                token_index=request.token_index,
                tensor_b64=tensor_b64,
                tensor_dtype=tensor_dtype,
                tensor_shape=tensor_shape,
                metadata=request.metadata,
                metrics=accumulated_metrics,
                transport={
                    "encoding": "json_base64",
                    "compression": compression_meta,
                },
                feature_flags=feature_flags,
            )
            payload = next_request.model_dump()
            response = stage_http_client.post(
                next_stage_url,
                json=payload,
                timeout=STAGE_REQUEST_TIMEOUT_SECONDS,
            )
            request_payload_bytes = len(
                json.dumps(
                    payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )

        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to forward to next stage.")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to forward to next stage {next_stage_url}: {exc}",
        ) from exc

    transfer_time_ms = elapsed_ms(transfer_start)
    response_payload_bytes = len(response.content)

    if BinaryTransportModule.is_enabled(feature_flags):
        response_payload = BinaryTransportModule.unpack_response(response.content)
    else:
        response_payload = response.json()

    response_metrics = response_payload.get("metrics", [])
    if accumulated_metrics and response_metrics:
        local_metric_index = len(accumulated_metrics) - 1
        if 0 <= local_metric_index < len(response_metrics):
            response_metrics[local_metric_index]["transfer_time_ms"] = transfer_time_ms
            response_metrics[local_metric_index]["outbound_payload_b64_bytes"] = request_payload_bytes
            response_metrics[local_metric_index]["outbound_payload_b64_mebibytes"] = (
                request_payload_bytes / (1024.0 * 1024.0)
            )
            response_metrics[local_metric_index]["outbound_payload_bytes"] = request_payload_bytes
            response_metrics[local_metric_index]["outbound_payload_mebibytes"] = (
                request_payload_bytes / (1024.0 * 1024.0)
            )
            response_metrics[local_metric_index]["transport_encoding"] = (
                "binary_octet_stream" if BinaryTransportModule.is_enabled(feature_flags) else "json_base64"
            )
            response_metrics[local_metric_index]["estimated_link_mbps"] = (
                (request_payload_bytes * 8.0 / (transfer_time_ms / 1000.0) / 1_000_000.0)
                if transfer_time_ms > 0.0
                else 0.0
            )
            response_metrics[local_metric_index]["next_stage_url"] = next_stage_url
            response_metrics[local_metric_index]["next_stage_response_payload_bytes"] = response_payload_bytes
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
            "transport_encoding": request.transport.get("encoding", "json_base64"),
            "feature_modules": request.feature_flags.enabled_module_keys(),
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
