from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dli.common.metrics import compute_tokens_per_second, make_stage_metric
from dli.common.schemas import GenerateRequest, GenerateResponse, TokenStepMetric
from dli.common.timing import elapsed_ms, now_ms


class BaselineExecutor:
    """
    Executes full-model inference on one Raspberry Pi.

    This is the Baseline Single-Node Inference configuration used as the
    controlled comparison against Distributed Layer Inference.
    """

    def __init__(
        self,
        *,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        service_name: str = "inference-baseline",
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.service_name = service_name
        self.device = torch.device(device)

    @torch.no_grad()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        total_start = now_ms()

        prompt_text = self._resolve_prompt_text(request)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids: torch.Tensor = encoded["input_ids"].to(self.device)
        attention_mask: Optional[torch.Tensor] = encoded.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generated_token_ids: List[int] = []
        token_metrics: List[TokenStepMetric] = []

        past_key_values = None

        for token_index in range(request.max_new_tokens):
            step_start = now_ms()
            compute_start = now_ms()

            if past_key_values is None:
                model_input_ids = input_ids
            else:
                model_input_ids = input_ids[:, -1:]

            outputs = self.model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_id = self._select_next_token(
                logits=logits,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                suppress_eos=(token_index + 1) < request.min_new_tokens,
            )

            compute_time_ms = elapsed_ms(compute_start)

            generated_token_ids.append(next_token_id)

            next_token_tensor = torch.tensor(
                [[next_token_id]],
                dtype=input_ids.dtype,
                device=self.device,
            )

            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

            if attention_mask is not None:
                new_attention = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=self.device,
                )
                attention_mask = torch.cat([attention_mask, new_attention], dim=-1)

            token_text = self.tokenizer.decode([next_token_id])

            metric = make_stage_metric(
                service_name=self.service_name,
                stage_id=None,
                token_index=token_index,
                compute_time_ms=compute_time_ms,
                transfer_time_ms=0.0,
                extra={
                    "execution_mode": "single_node_baseline",
                    "sequence_length": int(input_ids.shape[-1]),
                    "next_token_id": next_token_id,
                },
            )

            token_metrics.append(
                TokenStepMetric(
                    token_index=token_index,
                    token_id=next_token_id,
                    token_text=token_text,
                    latency_ms=elapsed_ms(step_start),
                    stage_metrics=[metric],
                )
            )

            if self._should_stop(next_token_id):
                break

        total_latency_ms = elapsed_ms(total_start)

        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        return GenerateResponse(
            prompt=prompt_text,
            assistant_message=generated_text,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            total_latency_ms=total_latency_ms,
            tokens_per_second=compute_tokens_per_second(
                num_tokens=len(generated_token_ids),
                total_latency_ms=total_latency_ms,
            ),
            termination_reason="eos" if generated_token_ids and self._should_stop(generated_token_ids[-1]) else "max_new_tokens",
            prompt_token_count=int(encoded["input_ids"].shape[-1]),
            token_metrics=token_metrics,
        )

    def _should_stop(self, token_id: int) -> bool:
        eos_token_id = self.tokenizer.eos_token_id
        return eos_token_id is not None and token_id == eos_token_id

    @staticmethod
    def _select_next_token(
        *,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        suppress_eos: bool = False,
    ) -> int:
        next_token_logits = logits[:, -1, :].clone()

        if suppress_eos and eos_token_id is not None and 0 <= int(eos_token_id) < int(next_token_logits.shape[-1]):
            next_token_logits[..., int(eos_token_id)] = float("-inf")

        if temperature <= 0.0:
            return int(torch.argmax(next_token_logits, dim=-1).item())

        scaled_logits = next_token_logits / temperature

        if top_k is not None and top_k > 0:
            values, indices = torch.topk(scaled_logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            selected = torch.multinomial(probabilities, num_samples=1)
            return int(indices.gather(-1, selected).item())

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits,
                descending=True,
            )
            sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

            sorted_indices_to_remove = cumulative_probabilities > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(
                sorted_indices_to_remove,
                float("-inf"),
            )

            probabilities = torch.softmax(sorted_logits, dim=-1)
            selected = torch.multinomial(probabilities, num_samples=1)
            return int(sorted_indices.gather(-1, selected).item())

        probabilities = torch.softmax(scaled_logits, dim=-1)
        selected = torch.multinomial(probabilities, num_samples=1)

        return int(selected.item())

    @staticmethod
    def _resolve_prompt_text(request: GenerateRequest) -> str:
        prompt_text = (request.prompt or "").strip()
        if prompt_text:
            return prompt_text

        if request.messages:
            for message in reversed(request.messages):
                if (message.role or "").strip().lower() == "user":
                    text = (message.content or "").strip()
                    if text:
                        return text
            return (request.messages[-1].content or "").strip()

        return ""
