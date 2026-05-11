from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
import torch
from fastapi import FastAPI, HTTPException, Request, Response

from dli.common.logging_config import configure_logging
from dli.common.metrics import make_stage_metric
from dli.common.schemas import HealthResponse, StageForwardRequest, StageForwardResponse
from dli.common.timing import elapsed_ms, now_ms
from dli.feature_modules.activation_payload_precision import ActivationPayloadPrecisionModule
from dli.feature_modules.activation_transport_binary import BinaryTransportModule
from dli.feature_modules.kv_cache_stage import StageForwardCache, StageKVCacheManager
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


class StageExecutorRegistry:
    def __init__(self, config: StageConfig, device: str) -> None:
        self.config = config
        self.device = device
        self._executors: Dict[str, tuple[StageExecutor, str, bool]] = {}

    def get(self, profile: str) -> tuple[StageExecutor, str, bool]:
        partition_file, runtime_applied = PartitionRebalanceModule.resolve_partition_file(
            profile=profile,
            partition_profiles=self.config.partition_profiles,
            baseline_partition_file=self.config.partition_file,
        )
        selected_profile = profile if runtime_applied or profile == "baseline" else "baseline"

        if not os.path.exists(partition_file):
            logger.warning(
                "Partition profile %s points to missing file %s; falling back to baseline.",
                profile,
                partition_file,
            )
            selected_profile = "baseline"
            partition_file = self.config.partition_file
            runtime_applied = False

        cached = self._executors.get(selected_profile)
        if cached is not None and cached[1] == partition_file:
            return cached

        loader = ModelPartitionLoader(partition_file=partition_file, device=self.device)
        partition = loader.load()
        executor = StageExecutor(
            partition=partition,
            device=self.device,
            stage_id=self.config.stage_id,
        )
        record = (executor, partition_file, runtime_applied)
        self._executors[selected_profile] = record
        logger.info(
            "Loaded partition profile=%s file=%s layers=%s",
            selected_profile,
            partition_file,
            partition.metadata.get("layer_indices"),
        )
        return record


executor_registry = StageExecutorRegistry(stage_config, DEVICE)
codec = ActivationCodec()
stage_forward_cache = StageForwardCache(max_entries=STAGE_CACHE_ENTRIES)
stage_kv_cache = StageKVCacheManager(max_entries=STAGE_CACHE_ENTRIES)

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
    server_start_ms = now_ms()
    logger.info(
        "Received forward request_id=%s token_index=%s encoding=%s",
        request.request_id,
        request.token_index,
        request.transport.get("encoding", "json_base64"),
    )

    inbound_payload_b64_bytes = len(request.tensor_b64.encode("utf-8"))
    try:
        decode_start_ms = now_ms()
        input_tensor = codec.decode_tensor(request.tensor_b64)
        inbound_decode_ms = elapsed_ms(decode_start_ms)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode activation tensor: {exc}",
        ) from exc

    compression_meta = request.transport.get("compression", {})
    restore_start_ms = now_ms()
    input_tensor = ActivationPayloadPrecisionModule.restore_tensor(
        input_tensor,
        compression_meta,
    )
    inbound_precision_restore_ms = elapsed_ms(restore_start_ms)

    return _run_forward_flow(
        request=request,
        input_tensor=input_tensor,
        inbound_payload_bytes=inbound_payload_b64_bytes,
        server_start_ms=server_start_ms,
        inbound_transport_metrics={
            "inbound_unpack_ms": 0.0,
            "inbound_decode_ms": inbound_decode_ms,
            "inbound_decompression_ms": 0.0,
            "inbound_precision_restore_ms": inbound_precision_restore_ms,
        },
    )


@app.post("/forward-binary")
async def forward_binary(raw_request: Request) -> Response:
    server_start_ms = now_ms()
    raw_body = await raw_request.body()
    inbound_payload_bytes = len(raw_body)

    try:
        unpack_start_ms = now_ms()
        envelope = BinaryTransportModule.unpack_request(raw_body)
        inbound_unpack_ms = elapsed_ms(unpack_start_ms)
        request_dict = envelope.get("request")
        tensor_blob = envelope.get("tensor_blob")
        if not isinstance(request_dict, dict):
            raise TypeError("Binary envelope is missing request payload.")
        if not isinstance(tensor_blob, (bytes, bytearray)):
            raise TypeError("Binary envelope is missing tensor blob.")

        request = StageForwardRequest(**request_dict)
        decompress_start_ms = now_ms()
        tensor_blob = BinaryTransportModule.restore_tensor_blob(
            tensor_blob=bytes(tensor_blob),
            metadata=request.transport.get("payload_compression", {}),
        )
        inbound_decompression_ms = elapsed_ms(decompress_start_ms)
        decode_start_ms = now_ms()
        if envelope.get("wire_version") == BinaryTransportModule.WIRE_VERSION:
            input_tensor = BinaryTransportModule.decode_tensor_blob(
                bytes(tensor_blob),
                envelope.get("tensor") or {},
            )
        else:
            input_tensor = codec.decode_tensor_raw(tensor_blob)
        inbound_decode_ms = elapsed_ms(decode_start_ms)
        compression_meta = request.transport.get("compression", {})
        restore_start_ms = now_ms()
        input_tensor = ActivationPayloadPrecisionModule.restore_tensor(
            input_tensor,
            compression_meta,
        )
        inbound_precision_restore_ms = elapsed_ms(restore_start_ms)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse binary forward request: {exc}",
        ) from exc

    response_model = _run_forward_flow(
        request=request,
        input_tensor=input_tensor,
        inbound_payload_bytes=inbound_payload_bytes,
        server_start_ms=server_start_ms,
        inbound_transport_metrics={
            "inbound_unpack_ms": inbound_unpack_ms,
            "inbound_decode_ms": inbound_decode_ms,
            "inbound_decompression_ms": inbound_decompression_ms,
            "inbound_precision_restore_ms": inbound_precision_restore_ms,
        },
    )
    packed_response = BinaryTransportModule.pack_response(response_model.model_dump())
    return Response(content=packed_response, media_type="application/octet-stream")


def _run_forward_flow(
    *,
    request: StageForwardRequest,
    input_tensor: torch.Tensor,
    inbound_payload_bytes: int,
    server_start_ms: Optional[float] = None,
    inbound_transport_metrics: Optional[Dict[str, Any]] = None,
) -> StageForwardResponse:
    feature_flags = request.feature_flags
    rebalance_profile = PartitionRebalanceModule.profile(feature_flags)
    executor, partition_file, rebalance_runtime_applied = executor_registry.get(
        rebalance_profile
    )
    generation_mode = str(request.metadata.get("generation_mode") or "legacy")
    cache_position_start = _optional_int(request.metadata.get("cache_position_start"))
    use_transformer_kv_cache = (
        StageKVCacheManager.is_enabled(feature_flags)
        and generation_mode in {"prefill", "decode"}
    )

    compute_start = now_ms()
    forward_dedupe_hit = False
    cache_key = None
    forward_dedupe_hash_ms = 0.0
    forward_dedupe_store_ms = 0.0
    kv_cache_created = False
    kv_cache_present = False
    kv_cache_seq_before = 0
    kv_cache_seq_after = 0

    if StageForwardCache.is_enabled(feature_flags):
        hash_start_ms = now_ms()
        cache_key = StageForwardCache.build_key(
            request_id=request.request_id,
            token_index=request.token_index,
            tensor=input_tensor,
            generation_mode=generation_mode,
            cache_position_start=cache_position_start,
        )
        forward_dedupe_hash_ms = elapsed_ms(hash_start_ms)
        cached = stage_forward_cache.get(cache_key)
        if cached is not None:
            forward_dedupe_hit = True
            output_tensor = cached
        else:
            (
                output_tensor,
                kv_cache_created,
                kv_cache_present,
                kv_cache_seq_before,
                kv_cache_seq_after,
            ) = _execute_stage_forward_with_cache_policy(
                input_tensor=input_tensor,
                executor=executor,
                request=request,
                use_transformer_kv_cache=use_transformer_kv_cache,
                generation_mode=generation_mode,
                cache_position_start=cache_position_start,
            )
            store_start_ms = now_ms()
            stage_forward_cache.put(cache_key, output_tensor)
            forward_dedupe_store_ms = elapsed_ms(store_start_ms)
    else:
        (
            output_tensor,
            kv_cache_created,
            kv_cache_present,
            kv_cache_seq_before,
            kv_cache_seq_after,
        ) = _execute_stage_forward_with_cache_policy(
            input_tensor=input_tensor,
            executor=executor,
            request=request,
            use_transformer_kv_cache=use_transformer_kv_cache,
            generation_mode=generation_mode,
            cache_position_start=cache_position_start,
        )

    if forward_dedupe_hit and use_transformer_kv_cache:
        kv_cache_present = stage_kv_cache.get(request.request_id) is not None
        kv_cache_seq_after = stage_kv_cache.seq_length(
            request_id=request.request_id,
            layer_idx=executor.first_cache_layer_idx,
        )
        kv_cache_seq_before = kv_cache_seq_after

    compute_time_ms = elapsed_ms(compute_start)
    input_tensor_bytes = int(input_tensor.numel() * input_tensor.element_size())
    output_tensor_bytes = int(output_tensor.numel() * output_tensor.element_size())
    inbound_metrics = inbound_transport_metrics or {}
    inbound_true_comm_ms = (
        float(inbound_metrics.get("inbound_unpack_ms", 0.0))
        + float(inbound_metrics.get("inbound_decode_ms", 0.0))
        + float(inbound_metrics.get("inbound_decompression_ms", 0.0))
        + float(inbound_metrics.get("inbound_precision_restore_ms", 0.0))
    )
    input_token_count = _tensor_sequence_length(input_tensor)
    output_token_count = _tensor_sequence_length(output_tensor)
    kv_cache_step_valid = _validate_kv_cache_step(
        executor=executor,
        use_transformer_kv_cache=use_transformer_kv_cache,
        generation_mode=generation_mode,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        kv_cache_seq_before=kv_cache_seq_before,
        kv_cache_seq_after=kv_cache_seq_after,
    )
    if not kv_cache_step_valid:
        raise HTTPException(
            status_code=500,
            detail=(
                "KV cache decode invariant failed: "
                f"mode={generation_mode} input_tokens={input_token_count} "
                f"output_tokens={output_token_count} before={kv_cache_seq_before} "
                f"after={kv_cache_seq_after} cache_arg={executor.cache_arg_name}"
            ),
        )

    local_metric = make_stage_metric(
        service_name=stage_config.service_name,
        stage_id=stage_config.stage_id,
        token_index=request.token_index,
        compute_time_ms=compute_time_ms,
        transfer_time_ms=0.0,
        true_comm_ms=inbound_true_comm_ms,
        extra={
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output_tensor.shape),
            "is_final_stage": stage_config.next_stage_url is None,
            "input_tensor_bytes": input_tensor_bytes,
            "output_tensor_bytes": output_tensor_bytes,
            "stage_input_token_count": input_token_count,
            "stage_output_token_count": output_token_count,
            "inbound_payload_b64_bytes": inbound_payload_bytes,
            "inbound_payload_bytes": inbound_payload_bytes,
            "generation_mode": generation_mode,
            "kv_cache_enabled": feature_flags.kv_cache_enabled,
            "kv_cache_mode": generation_mode,
            "kv_cache_present": kv_cache_present,
            "kv_cache_created": kv_cache_created,
            "kv_cache_seq_before": kv_cache_seq_before,
            "kv_cache_seq_after": kv_cache_seq_after,
            "kv_cache_step_valid": kv_cache_step_valid,
            "cache_arg_name": executor.cache_arg_name,
            "supports_cache_position": executor.supports_cache_position,
            "kv_cache_entries": stage_kv_cache.size() if feature_flags.kv_cache_enabled else 0,
            "forward_dedupe_enabled": feature_flags.forward_dedupe_enabled,
            "kv_forward_dedupe_hit": forward_dedupe_hit,
            "kv_forward_dedupe_entries": stage_forward_cache.size()
            if feature_flags.forward_dedupe_enabled
            else 0,
            "forward_dedupe_hash_ms": forward_dedupe_hash_ms,
            "forward_dedupe_store_ms": forward_dedupe_store_ms,
            "rebalance_profile": rebalance_profile,
            "rebalance_runtime_applied": rebalance_runtime_applied,
            "partition_file": partition_file,
            "transport_encoding": request.transport.get("encoding", "json_base64"),
            "payload_compression": request.transport.get(
                "payload_compression",
                {"mode": "none", "applied": False},
            ),
            "torch_threads": executor.thread_config,
            "lm_head_quantization": executor.lm_head_quantization,
            "feature_modules": feature_flags.enabled_module_keys(),
            "inbound_true_comm_ms": inbound_true_comm_ms,
            **inbound_metrics,
        },
    )

    accumulated_metrics = list(request.metrics)
    accumulated_metrics.append(local_metric)

    if stage_config.next_stage_url:
        response_model = _forward_to_next_stage(
            request=request,
            output_tensor=output_tensor,
            accumulated_metrics=accumulated_metrics,
        )
    else:
        response_model = _finalize_response(
            request=request,
            executor=executor,
            hidden_states=output_tensor,
            accumulated_metrics=accumulated_metrics,
        )

    if server_start_ms is not None:
        response_model.server_wall_ms = elapsed_ms(server_start_ms)
    return response_model


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _tensor_sequence_length(tensor: torch.Tensor) -> int:
    if tensor.ndim >= 3:
        return int(tensor.shape[1])
    if tensor.ndim >= 2:
        return int(tensor.shape[-1])
    if tensor.ndim == 1:
        return int(tensor.shape[0])
    return 0


def _validate_kv_cache_step(
    *,
    executor: StageExecutor,
    use_transformer_kv_cache: bool,
    generation_mode: str,
    input_token_count: int,
    output_token_count: int,
    kv_cache_seq_before: int,
    kv_cache_seq_after: int,
) -> bool:
    if not use_transformer_kv_cache or not executor.has_transformer_layers:
        return True

    if generation_mode == "decode":
        return (
            input_token_count == 1
            and output_token_count == 1
            and kv_cache_seq_after == kv_cache_seq_before + 1
        )

    if generation_mode == "prefill":
        return (
            input_token_count > 0
            and output_token_count == input_token_count
            and kv_cache_seq_after >= input_token_count
        )

    return True


def _execute_stage_forward_with_cache_policy(
    *,
    input_tensor: torch.Tensor,
    executor: StageExecutor,
    request: StageForwardRequest,
    use_transformer_kv_cache: bool,
    generation_mode: str,
    cache_position_start: Optional[int],
) -> tuple[torch.Tensor, bool, bool, int, int]:
    if not use_transformer_kv_cache:
        output_tensor = _execute_stage_forward(input_tensor=input_tensor, executor=executor)
        return output_tensor, False, False, 0, 0

    try:
        executor.ensure_kv_cache_supported()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if executor.cache_config is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "KV cache mode is enabled, but this stage partition does not "
                "expose a cache-compatible config."
            ),
        )

    if generation_mode == "prefill":
        stage_kv_cache.reset(request.request_id)
    elif generation_mode == "decode" and stage_kv_cache.get(request.request_id) is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Decode request arrived before a prefill KV cache existed for "
                f"request_id={request.request_id}."
            ),
        )

    kv_entry, kv_cache_created = stage_kv_cache.get_or_create(
        request_id=request.request_id,
        config=executor.cache_config,
    )

    with kv_entry.lock:
        kv_cache_seq_before = executor.get_cache_seq_length(kv_entry.cache)
        output_tensor = _execute_stage_forward(
            input_tensor=input_tensor,
            executor=executor,
            past_key_values=kv_entry.cache,
            use_cache=True,
            cache_position_start=cache_position_start,
        )
        kv_cache_seq_after = executor.get_cache_seq_length(kv_entry.cache)

    return output_tensor, kv_cache_created, True, kv_cache_seq_before, kv_cache_seq_after


def _execute_stage_forward(
    *,
    input_tensor: torch.Tensor,
    executor: StageExecutor,
    past_key_values: Optional[Any] = None,
    use_cache: bool = False,
    cache_position_start: Optional[int] = None,
) -> torch.Tensor:
    try:
        with torch.no_grad():
            return executor.forward(
                input_tensor,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position_start=cache_position_start,
            )
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

    precision_start_ms = now_ms()
    encoded_tensor, compression_meta = ActivationPayloadPrecisionModule.prepare_tensor(
        output_tensor,
        feature_flags,
    )
    precision_cast_ms = elapsed_ms(precision_start_ms)
    encode_ms = 0.0
    pack_ms = 0.0
    http_roundtrip_ms = 0.0
    unpack_ms = 0.0
    decode_ms = 0.0
    compression_time_ms = 0.0
    decompression_time_ms = 0.0
    tensor_wire_bytes = 0

    if PersistentSessionPool.use_persistent_session(feature_flags):
        stage_http_client = PersistentSessionPool.get_or_create(next_stage_url)
    else:
        stage_http_client = requests

    try:
        if BinaryTransportModule.is_enabled(feature_flags):
            encode_start_ms = now_ms()
            tensor_blob, tensor_metadata = BinaryTransportModule.encode_tensor_blob(encoded_tensor)
            encode_ms = elapsed_ms(encode_start_ms)
            tensor_blob, payload_compression_meta = BinaryTransportModule.prepare_tensor_blob(
                tensor_blob=tensor_blob,
                flags=feature_flags,
            )
            compression_time_ms = float(payload_compression_meta.get("compression_time_ms", 0.0))
            tensor_wire_bytes = len(tensor_blob)
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
                    "payload_compression": payload_compression_meta,
                },
                feature_flags=feature_flags,
            )
            pack_start_ms = now_ms()
            packed = BinaryTransportModule.pack_request(
                request_dict=next_request.model_dump(),
                tensor_blob=tensor_blob,
                tensor_metadata=tensor_metadata,
            )
            pack_ms = elapsed_ms(pack_start_ms)
            rpc_start_ms = now_ms()
            response = stage_http_client.post(
                BinaryTransportModule.resolve_forward_url(next_stage_url),
                data=packed,
                headers={"Content-Type": "application/octet-stream"},
                timeout=STAGE_REQUEST_TIMEOUT_SECONDS,
            )
            http_roundtrip_ms = elapsed_ms(rpc_start_ms)
            request_payload_bytes = len(packed)
        else:
            encode_start_ms = now_ms()
            tensor_b64, tensor_dtype, tensor_shape = codec.encode_tensor(encoded_tensor)
            encode_ms = elapsed_ms(encode_start_ms)
            tensor_wire_bytes = len(tensor_b64.encode("utf-8"))
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
                    "payload_compression": {"mode": "none", "applied": False},
                },
                feature_flags=feature_flags,
            )
            payload = next_request.model_dump()
            pack_start_ms = now_ms()
            packed_json = json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            pack_ms = elapsed_ms(pack_start_ms)
            rpc_start_ms = now_ms()
            response = stage_http_client.post(
                next_stage_url,
                json=payload,
                timeout=STAGE_REQUEST_TIMEOUT_SECONDS,
            )
            http_roundtrip_ms = elapsed_ms(rpc_start_ms)
            request_payload_bytes = len(packed_json)

        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Failed to forward to next stage.")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to forward to next stage {next_stage_url}: {exc}",
        ) from exc

    response_payload_bytes = len(response.content)

    unpack_start_ms = now_ms()
    if BinaryTransportModule.is_enabled(feature_flags):
        response_payload = BinaryTransportModule.unpack_response(response.content)
    else:
        response_payload = response.json()
    unpack_ms = elapsed_ms(unpack_start_ms)
    remote_server_wall_ms = float(response_payload.get("server_wall_ms", 0.0) or 0.0)
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

    response_metrics = response_payload.get("metrics", [])
    if accumulated_metrics and response_metrics:
        local_metric_index = len(accumulated_metrics) - 1
        if 0 <= local_metric_index < len(response_metrics):
            response_metrics[local_metric_index]["transfer_time_ms"] = http_roundtrip_ms
            response_metrics[local_metric_index]["rpc_wall_time_ms"] = http_roundtrip_ms
            response_metrics[local_metric_index]["http_roundtrip_ms"] = http_roundtrip_ms
            response_metrics[local_metric_index]["remote_server_wall_ms"] = remote_server_wall_ms
            response_metrics[local_metric_index]["rpc_residual_ms"] = rpc_residual_ms
            response_metrics[local_metric_index]["true_comm_ms"] = true_comm_ms
            response_metrics[local_metric_index]["precision_cast_ms"] = precision_cast_ms
            response_metrics[local_metric_index]["encode_ms"] = encode_ms
            response_metrics[local_metric_index]["pack_ms"] = pack_ms
            response_metrics[local_metric_index]["compression_ms"] = compression_time_ms
            response_metrics[local_metric_index]["unpack_ms"] = unpack_ms
            response_metrics[local_metric_index]["decode_ms"] = decode_ms
            response_metrics[local_metric_index]["decompression_ms"] = decompression_time_ms
            response_metrics[local_metric_index]["outbound_payload_b64_bytes"] = request_payload_bytes
            response_metrics[local_metric_index]["outbound_payload_b64_mebibytes"] = (
                request_payload_bytes / (1024.0 * 1024.0)
            )
            response_metrics[local_metric_index]["outbound_payload_bytes"] = request_payload_bytes
            response_metrics[local_metric_index]["outbound_payload_mebibytes"] = (
                request_payload_bytes / (1024.0 * 1024.0)
            )
            response_metrics[local_metric_index]["request_wire_bytes"] = request_payload_bytes
            response_metrics[local_metric_index]["response_wire_bytes"] = response_payload_bytes
            response_metrics[local_metric_index]["tensor_wire_bytes"] = tensor_wire_bytes
            response_metrics[local_metric_index]["transport_encoding"] = (
                "binary_octet_stream" if BinaryTransportModule.is_enabled(feature_flags) else "json_base64"
            )
            response_metrics[local_metric_index]["estimated_link_mbps"] = (
                (request_payload_bytes * 8.0 / (http_roundtrip_ms / 1000.0) / 1_000_000.0)
                if http_roundtrip_ms > 0.0
                else 0.0
            )
            response_metrics[local_metric_index]["next_stage_url"] = next_stage_url
            response_metrics[local_metric_index]["next_stage_response_payload_bytes"] = response_payload_bytes
            response_metrics[local_metric_index]["payload_compression"] = (
                next_request.transport.get("payload_compression", {})
            )
            response_metrics[local_metric_index]["persistent_session_pool"] = (
                PersistentSessionPool.snapshot()
            )
    response_payload["metrics"] = response_metrics

    return StageForwardResponse(**response_payload)


def _finalize_response(
    *,
    request: StageForwardRequest,
    executor: StageExecutor,
    hidden_states: torch.Tensor,
    accumulated_metrics: list[Dict[str, Any]],
) -> StageForwardResponse:
    finalize_start = now_ms()

    try:
        finalize_result = executor.finalize_logits_profiled(hidden_states)
        logits = finalize_result.logits

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

        select_start = now_ms()
        next_token_id = executor.select_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )
        select_time_ms = elapsed_ms(select_start)

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
            "hidden_states_shape": list(hidden_states.shape),
            "stage4_norm_ms": finalize_result.profile.norm_time_ms,
            "stage4_lm_head_ms": finalize_result.profile.lm_head_time_ms,
            "stage4_select_ms": select_time_ms,
            "stage4_last_token_only": finalize_result.profile.last_token_only,
            "next_token_id": next_token_id,
            "transport_encoding": request.transport.get("encoding", "json_base64"),
            "torch_threads": executor.thread_config,
            "lm_head_quantization": executor.lm_head_quantization,
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