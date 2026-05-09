from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


def sanitize_model_id(model_id: str) -> str:
    """Convert a Hugging Face model id into a safe local folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id.strip())


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().strip()

    if normalized in {"auto"}:
        # The caller handles "auto" by passing torch_dtype="auto" to HF.
        raise ValueError("auto must be handled by the caller")
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16

    raise ValueError(
        f"Unsupported dtype '{dtype_name}'. Use auto, float32, float16, or bfloat16."
    )


def resolve_model_source(
    *,
    model_id: str,
    models_root: Path,
    local_model_dir: Optional[Path] = None,
) -> Path | str:
    """
    Return a local model path if one exists in the project, otherwise return the
    Hugging Face model id so transformers can download it.

    Search order:
      1. explicit --local-model-dir
      2. models/<sanitized_model_id>
      3. model_id from Hugging Face Hub
    """
    if local_model_dir is not None:
        if not local_model_dir.exists():
            raise FileNotFoundError(f"Local model directory does not exist: {local_model_dir}")
        return local_model_dir

    project_model_dir = models_root / sanitize_model_id(model_id)
    if project_model_dir.exists():
        return project_model_dir

    return model_id


def download_snapshot_if_needed(
    *,
    model_id: str,
    models_root: Path,
    local_model_dir: Optional[Path],
    force_download: bool = False,
    allow_patterns: Optional[list[str]] = None,
) -> Path | str:
    """
    Download the model snapshot into models/<model_id> when it is not already
    present. If huggingface_hub is not available, return the HF id and let
    transformers download into its cache.
    """
    if local_model_dir is not None:
        if not local_model_dir.exists():
            raise FileNotFoundError(f"Local model directory does not exist: {local_model_dir}")
        return local_model_dir

    target_dir = models_root / sanitize_model_id(model_id)

    if target_dir.exists() and not force_download:
        return target_dir

    models_root.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_id

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
        allow_patterns=allow_patterns,
    )

    return target_dir


def load_causal_lm_and_tokenizer(
    *,
    model_source: Path | str,
    dtype_name: str = "float32",
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> Tuple[Any, nn.Module]:
    """Load tokenizer and model for splitting."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to split Hugging Face models. "
            "Install requirements-stage.txt or run: pip install transformers accelerate huggingface_hub"
        ) from exc
    torch_dtype = "auto" if dtype_name.lower() == "auto" else resolve_torch_dtype(dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    model.to(torch.device(device))
    model.eval()

    return tokenizer, model


def get_decoder_stack(model: nn.Module) -> tuple[nn.Module, nn.ModuleList | list[nn.Module], nn.Module, nn.Module]:
    """
    Extract embedding, decoder layers, final norm, and lm_head from common
    Hugging Face causal LM architectures.

    Supported directly:
      - LLaMA / TinyLlama / Mistral / Qwen-like decoder-only models:
        model.model.embed_tokens, model.model.layers, model.model.norm, model.lm_head
      - GPT-2-like models:
        model.transformer.wte, model.transformer.h, model.transformer.ln_f, model.lm_head
    """
    # LLaMA / TinyLlama / Mistral / Qwen decoder-only style.
    if hasattr(model, "model"):
        inner = model.model
        if all(hasattr(inner, attr) for attr in ["embed_tokens", "layers", "norm"]):
            if not hasattr(model, "lm_head"):
                raise AttributeError("Model has decoder stack but no lm_head attribute.")
            return inner.embed_tokens, inner.layers, inner.norm, model.lm_head

    # GPT-2 style.
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if all(hasattr(transformer, attr) for attr in ["wte", "h", "ln_f"]):
            if not hasattr(model, "lm_head"):
                raise AttributeError("GPT-like model has no lm_head attribute.")
            return transformer.wte, transformer.h, transformer.ln_f, model.lm_head

    raise TypeError(
        "Unsupported model architecture for automatic splitting. "
        "Expected LLaMA/TinyLlama/Mistral/Qwen-like or GPT-2-like causal LM."
    )


def read_num_hidden_layers(model_source: Path | str, trust_remote_code: bool = False) -> int | None:
    """Read num_hidden_layers from config without loading full weights."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_source, trust_remote_code=trust_remote_code)
    except Exception:
        return None

    for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)

    return None
