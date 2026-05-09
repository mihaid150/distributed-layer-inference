from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().strip()

    if normalized in {"float32", "fp32"}:
        return torch.float32

    if normalized in {"float16", "fp16"}:
        return torch.float16

    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16

    raise ValueError(f"Unsupported TORCH_DTYPE={dtype_name}")


class BaselineModelLoader:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = resolve_torch_dtype(torch_dtype)

    def load(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )

        model.to(self.device)
        model.eval()

        return tokenizer, model