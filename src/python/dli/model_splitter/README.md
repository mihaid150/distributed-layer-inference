# Model Splitter

This folder contains the tooling that prepares a Hugging Face causal language model for the distributed-layer inference runtime.

It creates checkpoints with the same structure expected by `dli.inference_stage.model_partition_loader.ModelPartitionLoader`:

```text
models/partitions/<model>/<split>/
├── stage_1.pt
├── stage_2.pt
├── stage_3.pt
├── stage_4.pt
├── split_manifest.json
├── tokenizer/
└── config/
```

Each `stage_N.pt` contains:

```python
{
    "embedding": optional nn.Module,
    "layers": list[nn.Module],
    "norm": optional nn.Module,
    "lm_head": optional nn.Module,
    "metadata": dict,
}
```

## Basic usage

From the project root:

```bash
python -m dli.model_splitter.cli \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-stages 4 \
  --output-dir models/partitions/tinyllama-1.1b-chat/4-stage \
  --stage-map-file configs/stage_map.yaml \
  --physical-nodes dli-worker-1 dli-worker-2 dli-worker-3 dli-worker-4 \
  --dtype float32 \
  --device cpu \
  --force
```

Equivalent wrapper:

```bash
PYTHONPATH=src python scripts/split_model_into_stages.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-stages 4 \
  --output-dir models/partitions/tinyllama-1.1b-chat/4-stage \
  --stage-map-file configs/stage_map.yaml \
  --force
```

## Explicit layer ranges

For TinyLlama with 22 decoder layers, a balanced 4-stage split is generated automatically. You can override it:

```bash
PYTHONPATH=src python scripts/split_model_into_stages.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-stages 4 \
  --layer-ranges 0-5 6-10 11-15 16-21 \
  --output-dir models/partitions/tinyllama-1.1b-chat/custom-4-stage \
  --force
```

## Download behavior

If the model is not found in `models/hf/<sanitized-model-id>`, the splitter downloads it from Hugging Face into that folder. Use `--local-model-dir` to force loading from an existing directory.

## Supported architectures

The automatic splitter supports common decoder-only Hugging Face models:

- LLaMA / TinyLlama / Mistral / Qwen-like models: `model.model.embed_tokens`, `model.model.layers`, `model.model.norm`, `model.lm_head`
- GPT-2-like models: `model.transformer.wte`, `model.transformer.h`, `model.transformer.ln_f`, `model.lm_head`

