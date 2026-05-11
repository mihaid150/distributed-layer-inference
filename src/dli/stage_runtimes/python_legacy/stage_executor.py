from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, create_causal_mask

from dli.common.timing import elapsed_ms, now_ms
from dli.stage_runtimes.python_legacy.model_partition_loader import ModelPartition


@dataclass(frozen=True)
class FinalizeLogitsProfile:
    norm_time_ms: float
    lm_head_time_ms: float
    last_token_only: bool


@dataclass(frozen=True)
class FinalizeLogitsResult:
    logits: torch.Tensor
    profile: FinalizeLogitsProfile


class StageExecutor:
    """
    Executes one static Inference Stage.

    Stage 1 usually receives input_ids and applies embedding + first layers.
    Intermediate stages receive hidden_states and apply transformer layers.
    Final stage applies remaining layers, norm, lm_head, and produces logits.
    """

    def __init__(
        self,
        partition: ModelPartition,
        device: str = "cpu",
        *,
        stage_id: Optional[int] = None,
    ) -> None:
        self.partition = partition
        self.device = torch.device(device)
        self.stage_id = stage_id
        self._layer_forward_signatures: dict[int, tuple[set[str], bool]] = {}
        self._cache_layer_indices: list[int] = []
        self._original_cache_layer_indices: list[int] = []
        self._llama_config = None
        self._rotary_embedding = None
        self._cache_arg_name: Optional[str] = None
        self._supports_cache_position = False
        self.thread_config = self._configure_torch_threads()
        self.lm_head_quantization = self._configure_lm_head_quantization()

        if self.partition.layers:
            first_layer = self.partition.layers[0]
            first_self_attn = getattr(first_layer, "self_attn", None)
            self._llama_config = getattr(first_self_attn, "config", None)

            if self._llama_config is not None:
                self._rotary_embedding = LlamaRotaryEmbedding(config=self._llama_config).to(
                    self.device
                )
                self._rotary_embedding.eval()

        for local_layer_idx, layer in enumerate(self.partition.layers):
            self_attn = getattr(layer, "self_attn", None)
            layer_idx = getattr(self_attn, "layer_idx", None)
            if layer_idx is not None:
                self._original_cache_layer_indices.append(int(layer_idx))
                setattr(self_attn, "layer_idx", local_layer_idx)
                self._cache_layer_indices.append(local_layer_idx)

            signature = inspect.signature(layer.forward)
            parameter_names = set(signature.parameters.keys())
            has_var_keyword = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            self._layer_forward_signatures[id(layer)] = (parameter_names, has_var_keyword)
            if "past_key_values" in parameter_names and self._cache_arg_name is None:
                self._cache_arg_name = "past_key_values"
            elif "past_key_value" in parameter_names and self._cache_arg_name is None:
                self._cache_arg_name = "past_key_value"
            if "cache_position" in parameter_names:
                self._supports_cache_position = True

    @torch.no_grad()
    def forward(
        self,
        input_tensor: torch.Tensor,
        *,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        cache_position_start: Optional[int] = None,
    ) -> torch.Tensor:
        x = input_tensor.to(self.device)

        if self.partition.embedding is not None:
            x = x.long()
            x = self.partition.embedding(x)

        layer_kwargs = self._build_llama_layer_kwargs(
            x,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position_start=cache_position_start,
        )

        for layer in self.partition.layers:
            x = self._run_transformer_layer(layer, x, layer_kwargs)

        return x

    @torch.no_grad()
    def finalize_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.finalize_logits_profiled(hidden_states).logits

    @torch.no_grad()
    def finalize_logits_profiled(self, hidden_states: torch.Tensor) -> FinalizeLogitsResult:
        x = hidden_states.to(self.device)

        if self.partition.norm is None:
            raise RuntimeError("Final stage requires norm, but norm is missing.")

        if self.partition.lm_head is None:
            raise RuntimeError("Final stage requires lm_head, but lm_head is missing.")

        if x.ndim == 3 and x.shape[1] > 0:
            x = x[:, -1:, :].contiguous()

        norm_start = now_ms()
        x = self.partition.norm(x)
        norm_time_ms = elapsed_ms(norm_start)

        lm_head_start = now_ms()
        logits = self.partition.lm_head(x)
        lm_head_time_ms = elapsed_ms(lm_head_start)

        return FinalizeLogitsResult(
            logits=logits,
            profile=FinalizeLogitsProfile(
                norm_time_ms=norm_time_ms,
                lm_head_time_ms=lm_head_time_ms,
                last_token_only=True,
            ),
        )

    @staticmethod
    def select_next_token(
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        forbidden_token_ids: Optional[Iterable[int]] = None,
    ) -> int:
        last_token_logits = logits[:, -1, :].clone()

        if forbidden_token_ids:
            vocab_size = int(last_token_logits.shape[-1])
            for token_id in forbidden_token_ids:
                if 0 <= int(token_id) < vocab_size:
                    last_token_logits[..., int(token_id)] = float("-inf")

        if temperature <= 0.0:
            return int(torch.argmax(last_token_logits, dim=-1).item())

        scaled_logits = last_token_logits / temperature

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

    def _build_llama_layer_kwargs(
        self,
        hidden_states: torch.Tensor,
        *,
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position_start: Optional[int],
    ) -> dict[str, object]:
        if self._rotary_embedding is None or self._llama_config is None:
            return {}

        if hidden_states.ndim != 3:
            return {}

        batch_size, sequence_length, _ = hidden_states.shape
        if cache_position_start is None:
            cache_position_start = self.get_cache_seq_length(past_key_values)

        position_ids = (
            torch.arange(
                cache_position_start,
                cache_position_start + sequence_length,
                device=hidden_states.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        attention_mask = create_causal_mask(
            config=self._llama_config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=past_key_values if use_cache else None,
            position_ids=position_ids,
        )
        position_embeddings = self._rotary_embedding(hidden_states, position_ids=position_ids)

        kwargs: dict[str, object] = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": use_cache,
        }
        if use_cache and past_key_values is not None:
            if self._cache_arg_name:
                kwargs[self._cache_arg_name] = past_key_values
            if self._supports_cache_position:
                kwargs["cache_position"] = torch.arange(
                    cache_position_start,
                    cache_position_start + sequence_length,
                    device=hidden_states.device,
                    dtype=torch.long,
                )

        return kwargs

    def _filter_layer_kwargs(
        self,
        layer: nn.Module,
        candidate_kwargs: dict[str, object],
    ) -> dict[str, object]:
        signature_data = self._layer_forward_signatures.get(id(layer))
        if signature_data is None:
            return {}

        parameter_names, has_var_keyword = signature_data
        if has_var_keyword:
            return candidate_kwargs

        return {name: value for name, value in candidate_kwargs.items() if name in parameter_names}

    def _run_transformer_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        layer_kwargs: dict[str, object],
    ) -> torch.Tensor:
        """
        Handles the common Hugging Face transformer block return format.

        Many decoder blocks return a tuple:
            (hidden_states, attention_weights, present_key_value, ...)
        """
        filtered_kwargs = self._filter_layer_kwargs(layer, layer_kwargs)
        output = layer(hidden_states, **filtered_kwargs)

        if isinstance(output, tuple):
            return output[0]

        return output

    def get_cache_seq_length(self, past_key_values: Optional[Any]) -> int:
        if past_key_values is None:
            return 0

        layer_idx = self._cache_layer_indices[0] if self._cache_layer_indices else 0
        try:
            return int(past_key_values.get_seq_length(layer_idx))
        except Exception:
            return 0

    @property
    def cache_config(self) -> Optional[Any]:
        return self._llama_config

    @property
    def has_transformer_layers(self) -> bool:
        return bool(self.partition.layers)

    @property
    def cache_arg_name(self) -> Optional[str]:
        return self._cache_arg_name

    @property
    def supports_cache_position(self) -> bool:
        return self._supports_cache_position

    def ensure_kv_cache_supported(self) -> None:
        if self.partition.layers and self._cache_arg_name is None:
            raise RuntimeError(
                "KV cache mode is enabled, but transformer layers do not expose "
                "past_key_values or past_key_value in their forward signatures."
            )

    @property
    def first_cache_layer_idx(self) -> int:
        return self._cache_layer_indices[0] if self._cache_layer_indices else 0

    def _configure_torch_threads(self) -> dict[str, Optional[int]]:
        requested_num_threads = self._read_thread_setting("TORCH_NUM_THREADS")
        if requested_num_threads is None:
            requested_num_threads = self._read_thread_setting("OMP_NUM_THREADS")
        requested_interop_threads = self._read_thread_setting("TORCH_INTEROP_NUM_THREADS")

        if requested_num_threads is not None and requested_num_threads > 0:
            torch.set_num_threads(requested_num_threads)

        if requested_interop_threads is not None and requested_interop_threads > 0:
            try:
                torch.set_num_interop_threads(requested_interop_threads)
            except RuntimeError:
                pass

        return {
            "requested_num_threads": requested_num_threads,
            "actual_num_threads": int(torch.get_num_threads()),
            "requested_interop_threads": requested_interop_threads,
            "actual_interop_threads": int(torch.get_num_interop_threads()),
            "omp_num_threads_env": self._read_thread_setting("OMP_NUM_THREADS"),
        }

    def _read_thread_setting(self, suffix: str) -> Optional[int]:
        keys = []
        if self.stage_id is not None:
            keys.append(f"STAGE_{self.stage_id}_{suffix}")
            keys.append(f"STAGE{self.stage_id}_{suffix}")
        keys.extend([f"STAGE_{suffix}", f"DLI_{suffix}", suffix])

        for key in keys:
            raw_value = os.getenv(key)
            if raw_value is None or raw_value.strip() == "":
                continue
            try:
                return int(raw_value)
            except ValueError:
                continue
        return None

    def _configure_lm_head_quantization(self) -> dict[str, object]:
        mode = self._read_string_setting("LM_HEAD_QUANTIZATION") or "none"
        result: dict[str, object] = {
            "mode": mode,
            "applied": False,
        }

        if mode in {"", "none", "off", "false"}:
            result["mode"] = "none"
            return result

        if self.partition.lm_head is None:
            result["skipped_reason"] = "lm_head_missing"
            return result

        if self.device.type != "cpu":
            result["skipped_reason"] = "dynamic_quantization_cpu_only"
            return result

        if mode not in {"dynamic_int8", "int8"}:
            result["skipped_reason"] = f"unsupported_mode:{mode}"
            return result

        try:
            self.partition.lm_head = torch.quantization.quantize_dynamic(
                self.partition.lm_head,
                {nn.Linear},
                dtype=torch.qint8,
            )
            self.partition.lm_head.eval()
            result.update({"mode": "dynamic_int8", "applied": True})
        except Exception as exc:
            result["skipped_reason"] = str(exc)

        return result

    def _read_string_setting(self, suffix: str) -> Optional[str]:
        keys = []
        if self.stage_id is not None:
            keys.append(f"STAGE_{self.stage_id}_{suffix}")
            keys.append(f"STAGE{self.stage_id}_{suffix}")
        keys.extend([f"STAGE_{suffix}", f"DLI_{suffix}", suffix])

        for key in keys:
            raw_value = os.getenv(key)
            if raw_value is None:
                continue
            value = raw_value.strip().lower()
            if value:
                return value
        return None
