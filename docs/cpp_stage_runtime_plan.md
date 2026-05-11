# Native C++ Stage Runtime Plan

## Direction

Keep the Python gateway and replace stage containers incrementally. The gateway
continues to own tokenization, generation orchestration, experiment profiles,
history, and benchmarks. Stage services become interchangeable backends behind
the existing `/forward-binary` protocol.

The immediate repo structure is:

- `src/dli/stage_runtimes/python_legacy`: current Python/PyTorch stage runtime.
- `src/dli/inference_stage`: compatibility shims for the old import path.
- `stage-runtimes/cpp`: planned native C++/ggml stage workspace.
- `docker/Dockerfile.stage`: legacy Python/PyTorch stage image.
- `docker/Dockerfile.stage-cpp`: native stage image skeleton.

## Architecture Target

Runtime chain:

```text
gateway-python
  -> dli-stage-cpp-1
  -> dli-stage-cpp-2
  -> dli-stage-cpp-3
  -> dli-stage-cpp-4
```

The native stage server must preserve:

- `GET /health`
- `GET /config`
- `POST /forward-binary`

The C++ stage is a custom DLI server using ggml/llama.cpp internals. It is not a
stock `llama-server` deployment, because DLI needs partial-layer execution and
hidden-state input/output between stages.

## Native Binary Protocol

Use a stable protocol independent of Python/Pydantic:

```cpp
struct DliRequestHeader {
    char magic[4];          // "DLI2"
    uint32_t version;       // 2
    uint32_t header_len;    // JSON metadata length
    uint64_t tensor_len;    // raw tensor bytes length
};
```

Frame:

```text
[DliRequestHeader][JSON metadata][raw tensor bytes]
```

Required metadata:

```json
{
  "request_id": "uuid",
  "token_index": 7,
  "generation_mode": "decode",
  "tensor": {
    "dtype": "float16",
    "shape": [1, 1, 2048],
    "byte_order": "little"
  },
  "feature_flags": {
    "kv_cache_enabled": true
  },
  "sampling": {
    "temperature": 0.0,
    "top_k": null,
    "top_p": null
  }
}
```

Stage 1-3 responses carry hidden-state tensors. Stage 4 responses carry
`next_token_id`.

## Milestones

### M1: C++ Stage Shell

Implement only server/protocol plumbing:

- `/health`
- `/config`
- `/forward-binary`
- DLI2 binary decode/encode
- identity/dummy backend

Acceptance:

- Python gateway can call all four C++ shell stages.
- Metrics appear in the UI.
- Kubernetes deployment shape is compatible.

### M2: C++ Stage 4 First

Implement final-stage execution:

```text
hidden_states -> local layers -> final norm -> lm_head -> next_token_id
```

Start with F16 DLI GGUF shards, then Q8_0.

Acceptance:

- Hybrid pipeline works: Python stage 1-3, C++ stage 4.
- Temperature 0 token choices are comparable against PyTorch stage 4.
- Stage 4 memory and compute/token improve.

### M3: C++ Intermediate Stage

Implement stage 3:

```text
hidden_states -> local layers -> hidden_states
```

Acceptance:

- Hidden-state shape and dtype match the gateway/stage contract.
- Generated text remains readable.
- Divergence against PyTorch is measured explicitly.

### M4: Full C++ Pipeline

Implement stage 1 embedding and stage 2:

```text
token_ids -> embedding -> local layers -> hidden_states
```

Acceptance:

- Python gateway drives C++ stages 1-4.
- KV decode invariants remain valid.
- End-to-end latency improves against the Python/PyTorch stage profile.

## Model Artifacts

Keep two artifact families:

- Legacy: `stage_N.pt` PyTorch partitions.
- Native: `stage_N.<quant>.dli.gguf` shards.

Initial quantization order:

1. F16
2. Q8_0
3. Q6_K
4. Q5_K_M
5. Q4_K_M only after quality validation

## `/config` Contract

Native stages should return:

```json
{
  "service_name": "inference-stage-2",
  "stage_id": 2,
  "backend": "ggml-cpp",
  "model_format": "dli-gguf",
  "quantization": "Q8_0",
  "layers": [6, 7, 8, 9, 10, 11],
  "supports_kv_cache": true,
  "hidden_dtype": "float16"
}
```

The legacy runtime now returns `backend=python-pytorch-legacy` from `/config`.

## Metrics To Preserve

- `compute_time_ms`
- `rpc_wall_time_ms`
- `true_comm_ms`
- `request_wire_bytes`
- `response_wire_bytes`
- `tensor_wire_bytes`
- `stage_input_token_count`
- `stage_output_token_count`
- `kv_cache_seq_before`
- `kv_cache_seq_after`
- `kv_cache_step_valid`
- `backend`
- `quantization`
- `n_threads`
- `weight_bytes_loaded`
- `effective_weight_bandwidth_mb_s`

Native-only metrics:

- `ggml_graph_build_ms`
- `ggml_graph_compute_ms`
- `ggml_alloc_ms`
- `kv_cache_bytes`
- `model_mmap`

## Main Risk

The hard part is not HTTP. The hard part is partial Llama layer execution with
hidden-state input/output. The practical path is to fork or extract the relevant
llama.cpp graph construction logic and wrap it in a DLI stage API.
