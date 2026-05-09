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
    ) -> None:
        self.tokenizer = tokenizer
        self.stage_client = stage_client
        self.codec = ActivationCodec()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        request_id = str(uuid.uuid4())

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

        for token_index in range(request.max_new_tokens):
            step_start = now_ms()

            encode_start = now_ms()
            tensor_b64, tensor_dtype, tensor_shape = self.codec.encode_tensor(input_ids)
            encode_time_ms = elapsed_ms(encode_start)
            input_tensor_bytes = int(input_ids.numel() * input_ids.element_size())
            outbound_payload_b64_bytes = len(tensor_b64.encode("utf-8"))

            stage_request = StageForwardRequest(
                request_id=request_id,
                token_index=token_index,
                tensor_b64=tensor_b64,
                tensor_dtype=tensor_dtype,
                tensor_shape=tensor_shape,
                metadata={
                    "temperature": request.temperature,
                    "min_new_tokens": request.min_new_tokens,
                    "eos_token_id": eos_token_id,
                    "forbidden_token_ids_before_min": special_token_ids,
                    "top_k": request.top_k,
                    "top_p": request.top_p,
                    "current_sequence_length": int(input_ids.shape[-1]),
                },
                metrics=[],
            )

            stage_result = self.stage_client.forward_with_transport(stage_request)
            stage_response = stage_result.response

            if stage_response.next_token_id is None:
                raise RuntimeError("Final Inference Stage did not return next_token_id.")

            next_token_id = int(stage_response.next_token_id)
            generated_token_ids.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

            token_text = self.tokenizer.decode([next_token_id])
            gateway_metric = make_stage_metric(
                service_name="inference-gateway",
                stage_id=0,
                token_index=token_index,
                compute_time_ms=encode_time_ms,
                transfer_time_ms=float(stage_result.transport.get("transfer_time_ms", 0.0)),
                extra={
                    "operation": "gateway_to_stage1",
                    "request_id": request_id,
                    "input_tensor_bytes": input_tensor_bytes,
                    "outbound_payload_b64_bytes": outbound_payload_b64_bytes,
                    "outbound_payload_mebibytes": outbound_payload_b64_bytes / (1024.0 * 1024.0),
                    "stage1_request_payload_bytes": int(
                        stage_result.transport.get("request_payload_bytes", 0)
                    ),
                    "stage1_response_payload_bytes": int(
                        stage_result.transport.get("response_payload_bytes", 0)
                    ),
                    "estimated_link_mbps": float(
                        stage_result.transport.get("estimated_link_mbps", 0.0)
                    ),
                    "stage1_url": stage_result.transport.get("url"),
                },
            )
            stage_metrics = list(stage_response.metrics)
            stage_metrics.append(gateway_metric)

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
    ) -> Dict[str, Any]:
        per_stage: Dict[str, Dict[str, Any]] = {}
        total_compute_ms = 0.0
        total_transfer_ms = 0.0
        total_payload_b64_bytes = 0
        total_network_delta_bytes = 0
        max_process_memory_mb = 0.0

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
                        "transfer_time_ms_sum": 0.0,
                        "payload_b64_bytes_sum": 0,
                        "network_delta_bytes_sum": 0,
                        "max_process_memory_mb": 0.0,
                        "max_cpu_percent": 0.0,
                    },
                )

                compute_time = float(metric.get("compute_time_ms", 0.0))
                transfer_time = float(metric.get("transfer_time_ms", 0.0))
                payload_b64_bytes = int(metric.get("outbound_payload_b64_bytes", 0))
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
                bucket["transfer_time_ms_sum"] += transfer_time
                bucket["payload_b64_bytes_sum"] += payload_b64_bytes
                bucket["network_delta_bytes_sum"] += network_delta_bytes
                bucket["max_process_memory_mb"] = max(
                    float(bucket["max_process_memory_mb"]), process_memory_mb
                )
                bucket["max_cpu_percent"] = max(float(bucket["max_cpu_percent"]), cpu_percent)

                total_compute_ms += compute_time
                total_transfer_ms += transfer_time
                total_payload_b64_bytes += payload_b64_bytes
                total_network_delta_bytes += network_delta_bytes
                max_process_memory_mb = max(max_process_memory_mb, process_memory_mb)

        return {
            "prompt_token_count": prompt_token_count,
            "generated_token_count": len(token_metrics),
            "total_latency_ms": total_latency_ms,
            "aggregate": {
                "compute_time_ms_sum": total_compute_ms,
                "transfer_time_ms_sum": total_transfer_ms,
                "payload_b64_bytes_sum": total_payload_b64_bytes,
                "payload_b64_mebibytes_sum": total_payload_b64_bytes / (1024.0 * 1024.0),
                "network_delta_bytes_sum": total_network_delta_bytes,
                "network_delta_mebibytes_sum": total_network_delta_bytes / (1024.0 * 1024.0),
                "max_process_memory_mb": max_process_memory_mb,
            },
            "per_stage": per_stage,
        }
