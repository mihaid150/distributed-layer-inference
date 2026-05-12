# DLI GGUF Stage Shard Specification

Status: initial native prototype contract.

This document defines the `.dli.gguf` stage-shard format consumed by the native distributed-layer-inference runtime.

A DLI GGUF stage shard is a valid GGUF file containing:

1. The normal model/global/tokenizer GGUF metadata needed by llama.cpp.
2. A strict subset of model tensors owned by exactly one DLI partition.
3. DLI-specific metadata under the `dli.*` namespace.

The shard writer must not rename tensors. Tensor names inside a shard must match the original full GGUF tensor names exactly.

---

## File naming

Recommended output naming:

```text
partition-1.dli.gguf
partition-2.dli.gguf
...
```

The filename is not authoritative. The authoritative partition identity is stored in metadata:

```text
dli.partition_id
dli.stage_id
```

---

## Required DLI metadata

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `dli.format` | string | yes | Literal `dli.gguf.stage_shard`. |
| `dli.format_version` | int32 | yes | Initial version is `1`. |
| `dli.partition_id` | string | yes | Static partition id, for example `partition-1`. |
| `dli.stage_id` | int32 | yes | Numeric stage id from `stage_map.yaml`. |
| `dli.layers` | array<int32> | yes | Transformer layer ids owned by this shard. |
| `dli.owns_embedding` | bool | yes | Whether this shard owns `token_embd.weight`. |
| `dli.owns_norm` | bool | yes | Whether this shard owns `output_norm.weight`. |
| `dli.owns_lm_head` | bool | yes | Whether this shard owns `output.weight`. |
| `dli.next_partition_id` | string | yes | Next static partition id if known, otherwise empty string. |
| `dli.next_stage_url` | string | yes | Next stage forward URL from `stage_map.yaml`, or empty string for terminal partition. |
| `dli.hidden_size` | int32 | yes | Hidden size, usually copied from `llama.embedding_length`; `-1` if unavailable. |
| `dli.source_model` | string | yes | Source full GGUF path or logical source model id used by the writer. |

---

## Tensor ownership contract

A tensor belongs to a partition using the same rules as the manifest generator:

- `token_embd.weight` belongs to the partition with `owns_embedding=true`.
- Every tensor with prefix `blk.<layer_id>.` belongs to the partition that owns that layer id.
- `output_norm.weight` belongs to the partition with `owns_norm=true`.
- `output.weight` belongs to the partition with `owns_lm_head=true`.

A valid shard must contain:

- all tensors listed in `partition-N.manifest.json`;
- no tensors owned by another partition;
- no extra tensors that are not matched by its component allocation.

The sum of all partition tensor counts must equal the source GGUF tensor count for the static allocation to be complete.

---

## Metadata preservation

The shard writer must preserve the source GGUF global and tokenizer metadata, then add/override only the `dli.*` keys.

At minimum, a validator should check that the shard has:

- `general.architecture`;
- at least one tokenizer metadata key with prefix `tokenizer.ggml.`;
- all required `dli.*` keys.

The first native writer may copy all source metadata into every shard. Later, metadata can be minimized if the runtime no longer needs tokenizer/global metadata on every partition.

---

## Stage runtime loading rule

When a stage is configured with:

```yaml
backend: llama
native_partition_file: /app/models/partition-1.dli.gguf
```

the native C++ stage must prefer `native_partition_file` over the legacy Python `.pt` `partition_file`.

The legacy field remains valid:

```yaml
partition_file: /app/models/stage_1.pt
```

This preserves the Python/PyTorch path while allowing native llama.cpp shards to be introduced incrementally.

---

## Terminal partition

A terminal partition is discovered generically by either:

```text
dli.owns_lm_head=true
```

or:

```text
dli.next_stage_url=""
```

The implementation must not assume that the terminal partition is stage 4.
