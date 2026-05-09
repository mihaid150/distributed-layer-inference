from __future__ import annotations

import inspect
from typing import Iterable, Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, create_causal_mask

from dli.inference_stage.model_partition_loader import ModelPartition


class StageExecutor:
    """
    Executes one static Inference Stage.

    Stage 1 usually receives input_ids and applies embedding + first layers.
    Intermediate stages receive hidden_states and apply transformer layers.
    Final stage applies remaining layers, norm, lm_head, and produces logits.
    """

    def __init__(self, partition: ModelPartition, device: str = "cpu") -> None:
        self.partition = partition
        self.device = torch.device(device)
        self._layer_forward_signatures: dict[int, tuple[set[str], bool]] = {}
        self._llama_config = None
        self._rotary_embedding = None

        if self.partition.layers:
            first_layer = self.partition.layers[0]
            first_self_attn = getattr(first_layer, "self_attn", None)
            self._llama_config = getattr(first_self_attn, "config", None)

            if self._llama_config is not None:
                self._rotary_embedding = LlamaRotaryEmbedding(config=self._llama_config).to(
                    self.device
                )
                self._rotary_embedding.eval()

        for layer in self.partition.layers:
            signature = inspect.signature(layer.forward)
            parameter_names = set(signature.parameters.keys())
            has_var_keyword = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            self._layer_forward_signatures[id(layer)] = (parameter_names, has_var_keyword)

    @torch.no_grad()
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor.to(self.device)

        if self.partition.embedding is not None:
            x = x.long()
            x = self.partition.embedding(x)

        layer_kwargs = self._build_llama_layer_kwargs(x)

        for layer in self.partition.layers:
            x = self._run_transformer_layer(layer, x, layer_kwargs)

        return x

    @torch.no_grad()
    def finalize_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(self.device)

        if self.partition.norm is None:
            raise RuntimeError("Final stage requires norm, but norm is missing.")

        if self.partition.lm_head is None:
            raise RuntimeError("Final stage requires lm_head, but lm_head is missing.")

        x = self.partition.norm(x)
        logits = self.partition.lm_head(x)

        return logits

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

    def _build_llama_layer_kwargs(self, hidden_states: torch.Tensor) -> dict[str, object]:
        if self._rotary_embedding is None or self._llama_config is None:
            return {}

        if hidden_states.ndim != 3:
            return {}

        batch_size, sequence_length, _ = hidden_states.shape
        position_ids = (
            torch.arange(sequence_length, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        attention_mask = create_causal_mask(
            config=self._llama_config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self._rotary_embedding(hidden_states, position_ids=position_ids)

        return {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": False,
        }

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
