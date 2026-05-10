from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from dli.common.metrics import compute_tokens_per_second, make_stage_metric
from dli.common.schemas import (
    ChatMessage,
    GenerateRequest,
    GenerateResponse,
    StageForwardRequest,
    TokenStepMetric,
)
from dli.common.timing import elapsed_ms, now_ms
from dli.feature_modules.partition_rebalance import PartitionRebalanceModule
from dli.feature_modules.topology_aware_routing import TopologyAwareRoutingModule
from dli.inference_stage.activation_codec import ActivationCodec
from dli.inference_gateway.stage_client import StageClient


class GenerationLoop:
    """
    Coordinates autoregressive generation over the distributed inference pipeline.

    The gateway keeps the generation state:
        input_ids -> Stage 1 -> Stage 2 -> ... -> final stage -> next_token_id

    Each iteration appends one generated token.
    """

    def __init__(
        self,
        *,
        tokenizer: AutoTokenizer,
        stage_client: StageClient,
        topology_route_candidates: Optional[Dict[str, List[str]]] = None,
        topology_probe_timeout_seconds: float = 0.25,
    ) -> None:
        self.tokenizer = tokenizer
        self.stage_client = stage_client
        self.codec = ActivationCodec()
        self.topology_route_candidates = topology_route_candidates or {}
        self.topology_probe_timeout_seconds = topology_probe_timeout_seconds

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        request_id = str(uuid.uuid4())
        feature_flags = request.feature_flags

        total_start = now_ms()

        prompt_for_response = self._prompt_for_response(request)
        input_ids: torch.Tensor = self._encode_input_ids(
            prompt=prompt_for_response,
            messages=request.messages,
        )
        prompt_token_count = int(input_ids.shape[-1])

        generated_token_ids: List[int] = []
        token_metrics: List[TokenStepMetric] = []
        termination_reason = "max_new_tokens"
        eos_token_id: Optional[int] = self.tokenizer.eos_token_id
        special_token_ids = sorted(
            {
                int(token_id)
                for token_id in (self.tokenizer.all_special_ids or [])
                if token_id is not None
            }
        )
        topology_route_overrides = TopologyAwareRoutingModule.build_route_overrides(
            flags=feature_flags,
            route_candidates=self.topology_route_candidates,
            probe_timeout_seconds=self.topology_probe_timeout_seconds,
        )

        for token_index in range(request.max_new_tokens):
            step_start = now_ms()
            use_prefill_decode = bool(feature_flags.kv_cache_enabled)
            current_sequence_length = int(input_ids.shape[-1])

            if use_prefill_decode and token_index > 0:
                stage_input_tensor = input_ids[:, -1:].contiguous()
                generation_mode = "decode"
                cache_position_start: Optional[int] = current_sequence_length - 1
            else:
                stage_input_tensor = input_ids
                generation_mode = "prefill" if use_prefill_decode else "legacy"
                cache_position_start = 0 if use_prefill_decode else None

            input_tensor_bytes = int(
                stage_input_tensor.numel() * stage_input_tensor.element_size()
            )

            stage_request = StageForwardRequest(
                request_id=request_id,
                token_index=token_index,
                tensor_b64="",
                tensor_dtype=str(stage_input_tensor.dtype),
                tensor_shape=list(stage_input_tensor.shape),
                metadata={
                    "feature_flags": feature_flags.model_dump(),
                    "rebalance": PartitionRebalanceModule.describe(feature_flags),
                    "temperature": request.temperature,
                    "min_new_tokens": request.min_new_tokens,
                    "eos_token_id": eos_token_id,
                    "forbidden_token_ids_before_min": special_token_ids,
                    "top_k": request.top_k,
                    "top_p": request.top_p,
                    "current_sequence_length": current_sequence_length,
                    "prompt_token_count": prompt_token_count,
                    "generation_mode": generation_mode,
                    "use_transformer_kv_cache": use_prefill_decode,
                    "cache_position_start": cache_position_start,
                    "stage_input_token_count": int(stage_input_tensor.shape[-1]),
                    "topology_route_overrides": topology_route_overrides,
                },
                metrics=[],
                transport={"encoding": "json_base64"},
                feature_flags=feature_flags,
            )

            stage_result = self.stage_client.forward_with_transport(
                stage_request,
                input_tensor=stage_input_tensor,
            )
            stage_response = stage_result.response

            if stage_response.next_token_id is None:
                raise RuntimeError("Final Inference Stage did not return next_token_id.")

            next_token_id = int(stage_response.next_token_id)
            generated_token_ids.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

            token_text = self.tokenizer.decode([next_token_id])
            request_payload_bytes = int(stage_result.transport.get("request_payload_bytes", 0))
            response_payload_bytes = int(stage_result.transport.get("response_payload_bytes", 0))
            rpc_wall_time_ms = float(stage_result.transport.get("rpc_wall_time_ms", 0.0))
            true_comm_ms = float(stage_result.transport.get("true_comm_ms", 0.0))
            gateway_metric = make_stage_metric(
                service_name="inference-gateway",
                stage_id=0,
                token_index=token_index,
                compute_time_ms=0.0,
                transfer_time_ms=rpc_wall_time_ms,
                rpc_wall_time_ms=rpc_wall_time_ms,
                true_comm_ms=true_comm_ms,
                extra={
                    "operation": "gateway_to_stage1",
                    "request_id": request_id,
                    "input_tensor_bytes": input_tensor_bytes,
                    "outbound_payload_b64_bytes": request_payload_bytes,
                    "outbound_payload_mebibytes": request_payload_bytes / (1024.0 * 1024.0),
                    "outbound_payload_bytes": request_payload_bytes,
                    "request_wire_bytes": request_payload_bytes,
                    "response_wire_bytes": response_payload_bytes,
                    "tensor_wire_bytes": int(stage_result.transport.get("tensor_wire_bytes", 0)),
                    "rpc_wall_time_ms": rpc_wall_time_ms,
                    "http_roundtrip_ms": float(stage_result.transport.get("http_roundtrip_ms", 0.0)),
                    "remote_server_wall_ms": float(stage_result.transport.get("remote_server_wall_ms", 0.0)),
                    "rpc_residual_ms": float(stage_result.transport.get("rpc_residual_ms", 0.0)),
                    "true_comm_ms": true_comm_ms,
                    "precision_cast_ms": float(stage_result.transport.get("precision_cast_ms", 0.0)),
                    "encode_ms": float(stage_result.transport.get("encode_ms", 0.0)),
                    "pack_ms": float(stage_result.transport.get("pack_ms", 0.0)),
                    "compression_ms": float(stage_result.transport.get("compression_ms", 0.0)),
                    "unpack_ms": float(stage_result.transport.get("unpack_ms", 0.0)),
                    "decode_ms": float(stage_result.transport.get("decode_ms", 0.0)),
                    "decompression_ms": float(stage_result.transport.get("decompression_ms", 0.0)),
                    "generation_mode": generation_mode,
                    "current_sequence_length": current_sequence_length,
                    "stage_input_shape": list(stage_input_tensor.shape),
                    "stage_input_token_count": int(stage_input_tensor.shape[-1]),
                    "cache_position_start": cache_position_start,
                    "stage1_request_payload_bytes": request_payload_bytes,
                    "stage1_response_payload_bytes": response_payload_bytes,
                    "estimated_link_mbps": float(
                        stage_result.transport.get("estimated_link_mbps", 0.0)
                    ),
                    "stage1_url": stage_result.transport.get("url"),
                    "transport_encoding": stage_result.transport.get("encoding"),
                    "payload_compression": stage_result.transport.get("payload_compression"),
                    "persistent_session_pool": stage_result.transport.get(
                        "persistent_session_pool"
                    ),
                    "topology_route_overrides": topology_route_overrides,
                    "feature_modules": feature_flags.enabled_module_keys(),
                },
            )
            stage_metrics = list(stage_response.metrics)
            stage_metrics.append(gateway_metric)
            TopologyAwareRoutingModule.observe_stage_metrics(stage_metrics)

            token_metrics.append(
                TokenStepMetric(
                    token_index=token_index,
                    token_id=next_token_id,
                    token_text=token_text,
                    latency_ms=elapsed_ms(step_start),
                    stage_metrics=stage_metrics,
                )
            )

            if self._should_stop(next_token_id) and (token_index + 1) >= request.min_new_tokens:
                termination_reason = "eos"
                break

        total_latency_ms = elapsed_ms(total_start)

        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        return GenerateResponse(
            prompt=prompt_for_response,
            assistant_message=generated_text,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            total_latency_ms=total_latency_ms,
            tokens_per_second=compute_tokens_per_second(
                num_tokens=len(generated_token_ids),
                total_latency_ms=total_latency_ms,
            ),
            termination_reason=termination_reason,
            prompt_token_count=prompt_token_count,
            summary_metrics=self._build_summary_metrics(
                token_metrics=token_metrics,
                total_latency_ms=total_latency_ms,
                prompt_token_count=prompt_token_count,
                feature_flags=feature_flags.model_dump(),
            ),
            token_metrics=token_metrics,
        )

    def _should_stop(self, token_id: int) -> bool:
        eos_token_id: Optional[int] = self.tokenizer.eos_token_id

        if eos_token_id is None:
            return False

        return token_id == eos_token_id

    def _encode_input_ids(
        self,
        *,
        prompt: str,
        messages: Optional[List[ChatMessage]],
    ) -> torch.Tensor:
        """
        Prefer chat-template formatting when available for instruct/chat models.
        Fallback to plain tokenizer call for non-chat models.
        """
        if messages:
            normalized_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
        else:
            normalized_messages = [{"role": "user", "content": prompt}]

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            try:
                chat_ids = self.tokenizer.apply_chat_template(
                    normalized_messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(chat_ids, torch.Tensor):
                    return chat_ids
            except Exception:
                pass

        if messages:
            prompt = self._messages_to_plain_prompt(messages)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"]

    def _messages_to_plain_prompt(self, messages: List[ChatMessage]) -> str:
        lines: List[str] = []
        for message in messages:
            role = (message.role or "user").strip().lower()
            content = (message.content or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _prompt_for_response(self, request: GenerateRequest) -> str:
        prompt = (request.prompt or "").strip()
        if prompt:
            return prompt
        messages = request.messages or []
        for message in reversed(messages):
            if (message.role or "").strip().lower() == "user":
                return (message.content or "").strip()
        if messages:
            return (messages[-1].content or "").strip()
        return ""

    def _build_summary_metrics(
        self,
        *,
        token_metrics: List[TokenStepMetric],
        total_latency_ms: float,
        prompt_token_count: int,
        feature_flags: Dict[str, Any],
    ) -> Dict[str, Any]:
        per_stage: Dict[str, Dict[str, Any]] = {}
        total_compute_ms = 0.0
        total_rpc_wall_ms = 0.0
        total_true_comm_ms = 0.0
        total_payload_b64_bytes = 0
        total_payload_bytes = 0
        total_network_delta_bytes = 0
        max_process_memory_mb = 0.0
        token_latencies_ms = [float(item.latency_ms) for item in token_metrics]

        for token_metric in token_metrics:
            for metric in token_metric.stage_metrics:
                stage_id = metric.get("stage_id")
                service_name = metric.get("service_name", "unknown")
                stage_key = (
                    f"stage_{stage_id}" if stage_id not in (None, 0) else service_name
                )

                bucket = per_stage.setdefault(
                    stage_key,
                    {
                        "service_name": service_name,
                        "stage_id": stage_id,
                        "samples": 0,
                        "compute_time_ms_sum": 0.0,
                        "rpc_wall_time_ms_sum": 0.0,
                        "true_comm_ms_sum": 0.0,
                        "transfer_time_ms_sum": 0.0,
                        "payload_b64_bytes_sum": 0,
                        "payload_bytes_sum": 0,
                        "network_delta_bytes_sum": 0,
                        "max_process_memory_mb": 0.0,
                        "max_cpu_percent": 0.0,
                        "stage4_norm_ms_sum": 0.0,
                        "stage4_lm_head_ms_sum": 0.0,
                        "stage4_select_ms_sum": 0.0,
                    },
                )

                compute_time = float(metric.get("compute_time_ms", 0.0))
                transfer_time = float(metric.get("transfer_time_ms", 0.0))
                rpc_wall_time = float(metric.get("rpc_wall_time_ms", transfer_time))
                true_comm_time = float(metric.get("true_comm_ms", 0.0))
                payload_b64_bytes = int(metric.get("outbound_payload_b64_bytes", 0))
                payload_bytes = int(
                    metric.get("outbound_payload_bytes", payload_b64_bytes)
                )
                network_delta = metric.get("network_delta", {}) or {}
                network_delta_bytes = int(
                    network_delta.get(
                        "bytes_total",
                        int(network_delta.get("bytes_sent", 0))
                        + int(network_delta.get("bytes_recv", 0)),
                    )
                )
                process_memory_mb = float(metric.get("process_memory_mb", 0.0))
                cpu_percent = float(metric.get("process_cpu_percent", metric.get("cpu_percent", 0.0)))

                bucket["samples"] += 1
                bucket["compute_time_ms_sum"] += compute_time
                bucket["rpc_wall_time_ms_sum"] += rpc_wall_time
                bucket["true_comm_ms_sum"] += true_comm_time
                bucket["transfer_time_ms_sum"] += rpc_wall_time
                bucket["payload_b64_bytes_sum"] += payload_b64_bytes
                bucket["payload_bytes_sum"] += payload_bytes
                bucket["network_delta_bytes_sum"] += network_delta_bytes
                bucket["max_process_memory_mb"] = max(
                    float(bucket["max_process_memory_mb"]), process_memory_mb
                )
                bucket["max_cpu_percent"] = max(float(bucket["max_cpu_percent"]), cpu_percent)
                bucket["stage4_norm_ms_sum"] += float(metric.get("stage4_norm_ms", 0.0))
                bucket["stage4_lm_head_ms_sum"] += float(metric.get("stage4_lm_head_ms", 0.0))
                bucket["stage4_select_ms_sum"] += float(metric.get("stage4_select_ms", 0.0))

                total_compute_ms += compute_time
                total_rpc_wall_ms += rpc_wall_time
                total_true_comm_ms += true_comm_time
                total_payload_b64_bytes += payload_b64_bytes
                total_payload_bytes += payload_bytes
                total_network_delta_bytes += network_delta_bytes
                max_process_memory_mb = max(max_process_memory_mb, process_memory_mb)

        stage4_bucket = per_stage.get("stage_4") or {}
        generated_token_count = len(token_metrics)
        stage4_compute_sum = float(stage4_bucket.get("compute_time_ms_sum", 0.0))
        rpc_compute_ratio = total_rpc_wall_ms / total_compute_ms if total_compute_ms > 0.0 else 0.0
        true_comm_compute_ratio = (
            total_true_comm_ms / total_compute_ms if total_compute_ms > 0.0 else 0.0
        )

        return {
            "prompt_token_count": prompt_token_count,
            "generated_token_count": generated_token_count,
            "total_latency_ms": total_latency_ms,
            "feature_flags": feature_flags,
            "enabled_modules": [
                name
                for name, enabled in {
                    "binary_transport": feature_flags.get("transport_mode") == "binary_octet_stream",
                    "activation_precision": feature_flags.get("activation_precision") != "fp32",
                    "payload_compression": feature_flags.get("payload_compression") != "none",
                    "kv_cache": bool(feature_flags.get("kv_cache_enabled")),
                    "rebalance": feature_flags.get("rebalance_profile") != "baseline",
                    "topology_aware": bool(feature_flags.get("topology_aware_routing")),
                    "persistent_sessions_backpressure": bool(
                        feature_flags.get("persistent_sessions_enabled")
                        or feature_flags.get("backpressure_enabled")
                    ),
                }.items()
                if enabled
            ],
            "aggregate": {
                "compute_time_ms_sum": total_compute_ms,
                "rpc_wall_time_ms_sum": total_rpc_wall_ms,
                "true_comm_ms_sum": total_true_comm_ms,
                "transfer_time_ms_sum": total_rpc_wall_ms,
                "rpc_compute_ratio": rpc_compute_ratio,
                "true_comm_compute_ratio": true_comm_compute_ratio,
                "comm_compute_ratio": true_comm_compute_ratio,
                "payload_b64_bytes_sum": total_payload_b64_bytes,
                "payload_b64_mebibytes_sum": total_payload_b64_bytes / (1024.0 * 1024.0),
                "payload_bytes_sum": total_payload_bytes,
                "payload_mebibytes_sum": total_payload_bytes / (1024.0 * 1024.0),
                "network_delta_bytes_sum": total_network_delta_bytes,
                "network_delta_mebibytes_sum": total_network_delta_bytes / (1024.0 * 1024.0),
                "max_process_memory_mb": max_process_memory_mb,
                "stage4_compute_time_ms_sum": stage4_compute_sum,
                "stage4_compute_time_ms_per_token": (
                    stage4_compute_sum / generated_token_count
                    if generated_token_count > 0
                    else 0.0
                ),
                "stage4_norm_ms_sum": float(stage4_bucket.get("stage4_norm_ms_sum", 0.0)),
                "stage4_lm_head_ms_sum": float(
                    stage4_bucket.get("stage4_lm_head_ms_sum", 0.0)
                ),
                "stage4_select_ms_sum": float(stage4_bucket.get("stage4_select_ms_sum", 0.0)),
                "token_latency_ms_p50": self._percentile(token_latencies_ms, 50.0),
                "token_latency_ms_p95": self._percentile(token_latencies_ms, 95.0),
                "token_latency_ms_p99": self._percentile(token_latencies_ms, 99.0),
            },
            "per_stage": per_stage,
        }

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (percentile / 100.0) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)
