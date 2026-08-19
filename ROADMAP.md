# DLI Optimization and Restructuring Roadmap

Tracked development backlog for Distributed Layer Inference. This document is
the working plan for the `optimization-roadmap` branch; it is deliberately kept
off `main` so that `README.md` stays a description of the system as built.

It supersedes the local, git-ignored `docs/future-optimizations.md`. Every item
is grounded either in a measurement recorded in `docs/ARCHITECTURE.md` §8/§11 or
in a specific code path, cited as `file:line`.

**Baseline at the time of writing.** TinyLlama-1.1B Q4_K_M, 4-stage 5/5/6/6 on
4x Raspberry Pi 5 (4 GB), greedy, fp32 activations, `performance` governor,
orchestrated routing, `DLI_GGML_THREADS=4`, `DLI_GGML_ATTENTION=1`:

**4096 tokens = 938.6 s / 4.36 tok/s.**

---

## 1. The constraint that orders this document

The system is **memory-bandwidth bound, not compute bound**. Every priority
below follows from that fact.

| Quantity | Value |
| --- | --- |
| Per-node peak fp32 throughput (4x Cortex-A76 @ 2.4 GHz, 2x 128-bit NEON FMA) | ~150 GFLOP/s theoretical |
| Per-node DRAM read bandwidth (LPDDR4X-4267, 32-bit) | ~17 GB/s theoretical, ~10 GB/s achieved |
| Roofline ridge point | ~15 FLOP/byte |
| Q4_K_M GEMV at batch=1 (2 FLOP per ~0.6-byte weight) | ~3.3 FLOP/byte |
| Measured matmul throughput (674 MB of shards read per token in 66.2 ms) | ~10.2 GB/s, ~33 GFLOP/s |
| Utilisation of the **active** node's FLOPs | **~22%** |
| Utilisation of **cluster** FLOPs (serial pipeline: 1 of 4 nodes computes) | **~5%** |
| Measured attention throughput (~103 GB of KV read per node per 4096-token run) | ~3.3 GB/s |

Note that `docs/ARCHITECTURE.md:320` and `:427` state the model is "201 MB";
the deployed shards actually total **674 MB** (171.8 + 135.0 + 156.7 + 210.4 MB)
and the source GGUF is 669 MB. Any bandwidth reasoning derived from 201 MB is
wrong by a factor of ~3.3. See Tier 0.

### 1.1 Three facts that decide what is worth building

**(a) There are two separate pools of idle CPU, with different fixes.**

- The *intra-node* pool (~78% of the active node's FLOPs) is stalled on DRAM.
  Only recoverable by raising arithmetic intensity: speculative decoding,
  batching, heavier quantisation.
- The *inter-node* pool (~75%, three idle nodes per token) is a **scheduling**
  problem, not a bandwidth problem. Each idle node has its own unused ~10 GB/s.
  Recoverable by continuous batching (fills the pipeline) or tensor parallelism
  (splits one token across all four buses).

**(b) Transport cost is CPU, not bytes.** §11.5 shows `true_comm/compute` stays
in a 0.55-0.75 band independent of payload; §11.6 shows fp16 activations halve
the payload with **no wall-time change**; links peak at ~9 Mbps of a gigabit.
Any transport work must remove **CPU work**, not wire bytes. Compressing
activations would be strictly the wrong direction.

**(c) The gateway residual becomes the dominant term.** The ~58.8 ms/token of
serial gateway bookkeeping is 26% of wall time today. Under the optimised
scenarios in §5 it rises to ~49-51%. It is the most under-prioritised item in
this document and is promoted to Tier 0 accordingly.

### 1.2 Optimisation taxonomy

Useful for the paper: not everything here is the same kind of work.

| Class | Items |
| --- | --- |
| Hardware configuration (no code) | thread count, CPU governor, pod network path |
| Hardware-aware software | KV fp16 (FEAT_FP16 + bandwidth), fused attention (L1/L2 blocking), batching and speculation (arithmetic intensity), tensor parallelism (memory-controller aggregation) |
| Pure software / waste removal | HTTP -> DLI2, hot-path allocation and libm churn, unified ggml graph |
| Algorithmic | sliding window (changes the work done, not how hardware is used) |

Already done and worth crediting as hardware-aware work: ISA targeting with
`-mcpu=armv8.2-a+fp16+dotprod` and the deliberate exclusion of `i8mm`
(`docker/Dockerfile.native:33-42`), the ggml/NEON attention kernel (~6x over
scalar, §11.2), the `performance` governor (~11-12%), and the `TCP_NODELAY`
diagnosis of the Nagle/delayed-ACK interaction (§8.1).

---

## 2. Tier 0 — instrumentation and measurement hygiene

**Do this first.** Without it, gains cannot be attributed and several items
below are being estimated rather than measured.

### 2.1 Add `overhead_ms` and `elementwise_ms` counters

`matmul_ms_` is incremented only around `compute_single_node_graph`
(`llama_cpu_executor.cpp:485-489`), so every `ggml_init`/`ggml_free`, the norm
weight reads, RoPE, SiLU, and `std::vector` churn fall into an unmeasured
remainder.

That remainder is **~58 s at 4096 tokens — 13% of compute** (compute 445.7 −
matmul 271.1 − attention 116.2) and is currently unattributed.

Consequence for the paper: the reported "matmul = 61% of compute" is a **lower
bound** on the true cost of the matvec path. A reviewer can reasonably ask what
is in "other". Fix the instrumentation before the numbers are published.

### 2.2 Profile and trim the gateway per-token residual

**Promoted from Tier 3.** ~58.8 ms/token of serial gateway bookkeeping —
incremental tokenisation, sampling, append, frame build — sitting on the
critical path.

It is 26% of wall time now, but §5 shows it becoming **~49% under tensor
parallelism and ~51% under TP plus speculative decoding**. Every other
optimisation makes it relatively more important. Nothing else in this document
addresses it except speculative decoding, which amortises it over K tokens.

Start with a flamegraph on the gateway during a 1024-token run.

### 2.3 Fix the model-size figure in the docs

`docs/ARCHITECTURE.md:320` and `:427`: "Q4_K_M 201 MB" -> 674 MB total across
shards. See §1.

### 2.4 Log thermal state during runs

Nothing in `documentation/` records cooling or throttling. Add
`vcgencmd get_throttled` and core temperature to the per-run metrics, otherwise
long 4096-token runs cannot be proven clean of thermal throttling.

---

## 3. Tier 1 — waste in the hot path

Cheap, low-risk, and all **bit-identical** unless noted. These live inside the
unmeasured remainder from §2.1.

### 3.1 64 MiB ggml context allocated per norm-weight read, per layer, per token

**The largest single find.** `run_layer` (`llama_cpu_executor.cpp:898-902`)
calls `tensor_vector_to_float` twice per layer:

```cpp
const std::vector<float> attn_norm_weight =
    tensor_vector_to_float(prefix + "attn_norm.weight", hp_.hidden_size);
const std::vector<float> ffn_norm_weight =
    tensor_vector_to_float(prefix + "ffn_norm.weight", hp_.hidden_size);
```

That path (`:262`, `:289`) allocates a **64 MiB** ggml context, builds a
`get_rows` + `ggml_cpy` graph, takes the backend mutex, dispatches the
threadpool, and frees — to read 2048 constants that never change.

At decode `seq_len == 1`, so this is **49,152 allocations of 64 MiB per node per
4096-token run** (2 x 6 layers x 4096).

Worse: 64 MiB exceeds glibc's `DEFAULT_MMAP_THRESHOLD_MAX` (32 MiB), so the
allocator never recycles it — it is a real `mmap`/`munmap` pair with page faults
and TLB invalidation every time. The 4 MiB matvec context escapes this after the
first free; this one does not.

*Fix: resolve norm weights once at load into a per-layer cache. ~10 lines.*

### 3.2 RoPE recomputes the same 32 cos/sin pairs ~200 times per token

`apply_rope` (`:545`) computes `std::cos`/`std::sin` inside the per-head loop
(`:570-573`), but the angle depends only on `position` and `i` — not on head and
not on layer.

- 36 heads (32 Q + 4 KV) x 32 pairs x 2 calls = 2304 libm calls per layer per token
- x 6 layers x 4096 tokens = **~56M calls per node per run**
- Distinct values actually needed: **32 per token**

`rope_inv_freq_` is already precomputed, which is the right instinct — the
cos/sin table is the remaining half.

*Fix: build one 32-entry cos/sin table per token, share it across layers.
~15 lines. Bit-identical (same values, computed once).*

### 3.3 Tensor lookup by string in the hot loop

`matvec(prefix + "attn_q.weight", ...)` allocates a string via
`layer_prefix()` + concatenation and performs a name lookup, for each of 9
tensors per layer per token — **~220k string allocations per node per run**.

*Fix: resolve `ggml_tensor*` pointers into a per-layer struct at load time.*

### 3.4 `matvec` copies its output twice

`ggml_mul_mat` already produces F32, but the path adds a `ggml_cpy` into an
output tensor and then `read_f32_tensor_flat` copies again into a
`std::vector`. On the lm_head (32000 logits) that is ~256 KB of pointless
copying per token on stage 4.

### 3.5 Reuse the ggml scratch context

`matvec` (`:414-418`) and `compute_attention_ggml_views` (`:779-783`) still
`ggml_init`/`ggml_free` per call, with 4 MiB and 2 MiB cushions respectively.

This is the same pattern already fixed twice — 128 MiB -> right-sized matvec
(§8, "-11% matmul"), 16 MB -> 1 MB attention. The remaining step is a
persistent, reused arena rather than a smaller per-call allocation.

*Fix: one scratch arena per executor, `no_alloc` + offset reset per call.*

### 3.6 Sampling sorts 32000 candidates per token

`sample_from_logits` (`:1279-1286`) fully sorts every vocabulary entry before
truncating to `top_k` — ~480k comparisons per token for typically 40 kept
candidates.

The greedy path is a clean linear argmax and is unaffected, so this does not
show up in current performance runs. But it is exactly the path needed for the
long-context quality work in §7.

*Fix: `std::nth_element` + `partial_sort`. 3 lines.*

---

## 4. Tier 2 — configuration only

### 4.1 `DLI_GGML_THREADS` 4 -> 2 or 3

§11.2 measures 76-77 s compute at 2-3 threads vs 81.6 s at 4; four ggml workers
oversubscribe against the network/IO/main threads on a 4-core Pi. The 1->2 step
is a genuine 1.60x on compute; 2->3 is flat within 1%, which is the cleanest
empirical evidence of the bandwidth roof in the whole dataset.

The knob is live via `configMapKeyRef` (`k8s/native/configmap.yaml:16`,
`k8s/native/deployments.yaml:106`). ~3% for one `kubectl patch`.

**Verify with `printenv` in the pod after patching** — per the §11.2 rule, only
runs where `printenv` matches the patch are valid.

### 4.2 Test the pod network path

`k8s/native/deployments.yaml` sets no `hostNetwork`, so stage pods run on the
default k3s pod network — **flannel with VXLAN**. Every gateway-to-stage message
is encapsulated (veth -> bridge -> VXLAN encap -> NIC -> decap -> veth), adding
latency and CPU on a path that already pays 16,384 round trips per 4096-token
run.

The cluster is a single switched L2 subnet (README, Cluster Topology), so the
encapsulation buys nothing.

Two experiments:

- `hostNetwork: true` plus `dnsPolicy: ClusterFirstWithHostNet` on the stage
  pods — targeted and reversible.
- `--flannel-backend=host-gw` on k3s — removes encapsulation entirely via direct
  routes. Correct on a single L2 segment; requires a k3s restart.

Worth measuring before the DLI2 framing work in §5.4, since it targets the same
cost for one line of YAML.

### 4.3 Quantisation ablation

Weight bytes are the dominant term, so bits per weight maps almost linearly to
matmul time. `IQ4_XS` (~4.25 bpw) is ~11% fewer bytes at small quality cost;
`Q3_K_M` (~3.9 bpw) is ~19% but degrades visibly on a 1.1B model.

Caveat: I-quants cost more dequantisation compute per byte than K-quants, and
K-quants are better optimised for ARM `dotprod`. The win is smaller than the
byte ratio suggests. Config-only: regenerate shards, no code change.

### 4.4 Check the lm_head tensor type

`Q4_K_M` normally keeps `output.weight` at Q6_K, and stage 4 reads it on every
token. `dli-gguf-info` does not report tensor types — use llama.cpp's
`gguf_dump`. If it is Q6_K, a Q4_K output is worth an ablation, but the output
layer is quality-sensitive; treat it as an experiment, not a free win.

---

## 5. Tier 3 — structural work

### 5.1 Speculative decoding — highest priority

A cheap proposer suggests K tokens; the model verifies all K in **one forward
pass**, so weights are read once per K accepted tokens.

**Why it fits this system unusually well:**

1. **Works at batch=1, single stream.** It does not require concurrent users and
   does not abandon the latency benchmark that all of §8/§11 is built on.
2. **Amortises the entire per-token cost**, not just weight reads: the 5 network
   hops, the framing, and the gateway residual — together over 50% of wall time
   (`true_comm` 251 s + ~240 s residual of 938 s). On a single-node GPU,
   speculation only attacks bandwidth. Here it attacks bandwidth **and**
   transport **and** the residual from §2.2.
3. **Exact.** With the standard acceptance rule, greedy output is identical to
   normal greedy decoding, so the bit-identical validation methodology from §8
   survives — unlike KV fp16, sliding window, or vectorised SiLU.
4. **The multi-token path already exists and is exercised by prefill.**
   `run_layer` loops `for (int t = 0; t < seq_len; ++t)` with
   `position = kv_seq_before + t`, and `llama_partial_runtime.cpp:1055-1059`
   handles `seq_len += input_seq`. Only decode is hardcoded to `+= 1`
   (`:1061-1065`). The DLI2 frame already carries `shape`, so there is no ABI
   change.
5. **Rollback on rejection is trivial.** The KV cache is a linear buffer indexed
   by position (`llama_cpu_executor.cpp:617-618`), so truncation is a decrement
   of `seq_len`.

**Start with prompt-lookup / n-gram speculation.** No draft model, no extra
memory, ~50 lines, and it lives entirely in the gateway. §8 notes greedy output
degenerates into repetition past ~30 sections at 4096, which gives n-gram
matching a high acceptance rate on exactly the benchmark workload.

**Rough estimate ~1.5-2x end-to-end**, but acceptance rate is workload
dependent. The degenerate 4096-token run is an unusually favourable case.
**Report acceptance rate as a measured variable and do not present the best case
as the general result** — on novel text expect 1.2-1.3x.

To batch the verification GEMVs into a GEMM, the batch dimension is literally
the `1` in `ggml_new_tensor_2d(ctx, F32, input_dim, 1)` (`:427`).

### 5.2 One ggml graph per partition forward

`run_layer` is a hand-written scalar forward pass that calls ggml as a
per-operation library. `compute_single_node_graph` (`:148`) builds a fresh
`ggml_cgraph`, takes the backend mutex and wakes the threadpool **~60 times per
token per node** (10 per layer x 6 layers).

ggml has native ops for everything done by hand here: `ggml_rms_norm`,
`ggml_rope`, `ggml_mul_mat`, `ggml_silu`, `ggml_mul`, `ggml_add`,
`ggml_soft_max`.

Building one graph per partition forward:

- subsumes §3.1-3.5 rather than fixing them one at a time
- gives NEON-vectorised elementwise kernels for free (including §6.4)
- reduces ~60 dispatch/barrier pairs per token to one
- turns batching (§5.3) and flash attention (§6.3) from surgery into ordinary
  graph edits

~300-400 lines, but lower risk than it sounds: the scalar path
(`compute_attention_scalar`, `:689`) remains as a reference oracle, and the
project already has the discipline of bit-identical validation.

### 5.3 Continuous batching

The only lever that harvests the **inter-node** idle pool: with N requests in
flight, all four stages compute simultaneously on different requests, so the
cluster's aggregate ~40 GB/s is actually used, and per-token GEMVs become GEMMs
(weights read once for N requests).

**Enablers already in place:**

- Weights are shared through a single `ggml_context`
  (`llama_cpu_executor.hpp:70,150`) — N sessions do not mean N copies of weights
- `LlamaRequestSession` holds only KV plus an executor
  (`llama_partial_runtime.hpp:98-101`)
- The batch dimension exists (`:427`)
- DLI2 metadata already carries `shape`, so `[N, hidden]` needs no ABI change

**Blockers:**

- The gateway serialises `/generate` behind `generation_mutex`
  (`dli_gateway_cpp/src/server.cpp:421`) — needs a scheduler with a queue
- Attention cannot be batched into one graph across sessions (different KV,
  different lengths) — needs a per-session loop. Acceptable: attention is
  bandwidth-bound per sequence anyway, and the win is on matmul.

**Implement iteration-level (continuous) scheduling, not static batches.** A
static batch of N runs at the speed of its longest request and leaves finished
slots idle; that loses most of the gain on variable-length generation. This is
the part of vLLM worth adopting — see §8.2.

**Useful batch range is roughly N = 4-8.** Per node, weights are ~168 MB while
attention is ~25 MB per sequence at 2048 context, so beyond N ~ 7 attention
traffic dominates again and further batching gives diminishing returns.

Note this changes the project from a single-stream latency benchmark into a
multi-user throughput system. That is a scope decision, not just an
implementation one.

### 5.4 Replace HTTP with raw DLI2 frames between gateway and stages

The frame format is **already self-delimiting** — a 20-byte header with magic,
version, `metadata_len` (u32) and `tensor_len` (u64) (`protocol.hpp:10-20`).
It is currently wrapped in an HTTP POST whose text header is built and parsed on
every token, every hop (`http_client.cpp:446-450`, `stage_client.cpp`).

So this is removing an envelope, not designing a protocol.

- **Client:** a `Dli2Client` that sends `encode_frame(...)` and reads 20 bytes,
  then the declared lengths. Reuses `send_all` (`http_client.cpp:66`) and the
  existing persistent `TCP_NODELAY` socket.
- **Server:** peek the first 4 bytes of an accepted connection
  (`dli_stage_cpp/src/server.cpp:686-706`) — `"DLI2"` takes the binary path,
  `"POST"`/`"GET"` keeps the existing HTTP parser so `/health` and the Ops UI
  are untouched.

**Also strip the per-message JSON.** `build_stub_metadata_json` and
`request_metadata_json` (`stage_client.cpp`) build metadata with
`ostringstream` on every call — text formatting per token per hop. A fixed
binary header removes it.

Target is the ~240 s serial residual at 4096. **Profile first** (§2.2) — part of
that residual is gateway tokenisation and sampling, not HTTP.

---

## 6. Tier 4 — approximate, gate behind flags

These change output, so each needs a default-off flag to preserve the
bit-identical baseline used throughout §8.

### 6.1 Sliding-window attention (`DLI_ATTN_WINDOW`)

Turns attention from O(n^2) into O(n) with **constant per-token cost**: at 4096
tokens, ~116 s becomes ~29 s with a 1024 window.

The cache layout already supports it. K is read in place with a fixed stride and
V is transposed with positions contiguous (`:817-830`), so a window is an offset
plus a length on both views:

```
start = max(0, required_seq - window)
K: data += start * kv_dim,  ne[1] = required_seq - start
V: data += start,           ne[0] = required_seq - start
```

RoPE stays on absolute positions, as llama.cpp does for Mistral/Gemma SWA.
~20-30 lines.

**Quality note that may invert the expected trade-off:**
`max_position_embeddings` is **2048** while the benchmark generates 4096 tokens
— the model is being run beyond its trained context. A 1024-2048 window keeps it
inside that range, so long-context quality may **improve**, not degrade. Run
this as an ablation with a quality axis (window ∈ {512, 1024, 2048, ∞} x
length), not as a pure latency change.

This is the only item that attacks *complexity* rather than constants, which
makes it the strongest single contribution for the paper.

### 6.2 fp16 KV cache (`DLI_KV_FP16`)

Halves the bytes read by the bandwidth-bound attention kernel (~103 GB -> ~51 GB
per node per 4096-token run).

`LayerKvCache` (`llama_cpu_executor.hpp:54-65`) is three `std::vector<float>`;
`ggml_mul_mat` accepts F16 `src0` with F32 `src1`, so the graph is unchanged.
Cortex-A76 implements ARMv8.2 FEAT_FP16 natively, so conversion is cheap.
Introduces rounding, so output is no longer bit-identical.

### 6.3 Fused / flash attention

Removes score-materialisation traffic. The current graph is
`mul_mat -> scale -> soft_max -> mul_mat` (`:833-837`) — four separate passes
over a scores array that is 512 KB per layer at n=4096, larger than the 512 KB
per-core L2.

The headroom is visible in the measurements: attention achieves **~3.3 GB/s**
against the **~10.2 GB/s** the same hardware reaches on matmul.

Two constraints:

- `ggml_flash_attn_ext` wants **F16 K/V** — do §6.2 first. These are coupled,
  not independent.
- It wants V **untransposed**, in `(head_dim, n_kv, n_head_kv)` layout, which
  conflicts with the persistent `v_t` buffer that produced the -74-75% attention
  win. Either keep both layouts or drop `v_t`.

Do this **after** §5.2, where it becomes a graph node rather than a separate
graph with incompatible layouts.

Expected ~4-7% end-to-end at 4096, ~0 at 1024. Correctly ranked last among the
compute items.

### 6.4 Vectorised SiLU

`silu()` (`:140-142`) calls `std::exp` in a scalar loop over `ffn_size = 5632`
(`:961-965`) — **~138M libm calls per node per 4096-token run**, which the
compiler cannot vectorise because of the libm call.

`ggml_silu` or a NEON `expf` approximation fixes it. Not bit-identical. Becomes
free if §5.2 is done.

---

## 7. Correctness and stability (not performance)

### 7.1 Session TTL / LRU eviction

§11.9 records `Max Session Count` climbing to 8 (~75 MiB KV) with the warning
"Native stage retained 8 sessions — check request cleanup/session pruning".

This is the bound on long unattended runs — **not memory pressure**, which peaks
at 477 MB of a 2000 MB limit. A stage restart resets it.

~30 lines in `LlamaPartialRuntime`. Needed regardless of any batching work, and
a prerequisite for multi-user operation.

### 7.2 Repetition / presence / frequency penalties in the sampler

`LlamaSamplingParams` currently exposes only temperature/top_k/top_p. This is
the real fix for the degenerate looping in long greedy generations (§8), which
is a sampling problem, not a numeric fault. No effect on greedy runs unless set.

Pairs with §3.6 — the penalty path runs through the sampling code that currently
sorts 32000 candidates per token.

### 7.3 Sampling presets in the Ops UI

Offer temp 0.7 / top_p 0.9, and stop forcing high `min_new_tokens`. Forcing
thousands of tokens from a 35-token prompt guarantees degeneration regardless of
model or sampler.

---

## 8. Architectural restructuring

Items in this section change what the system *is*, not how fast it runs. They
belong to a different decision than Tiers 0-4.

### 8.1 Tensor parallelism — conditionally viable

**Why it is attractive.** TP is the only scheme that gangs all four memory
controllers on a *single* token: 674 MB spread across 4 buses in parallel,
~168 MB per node. It is also the only architecture in which **adding nodes
reduces per-token latency**. Pipeline parallelism adds capacity (bigger models);
TP adds speed.

**Why it does not work today.** A Megatron-style column-then-row split leaves
each node holding a *partial sum*, so every transformer block needs two
all-reduces — one after the attention output projection, one after `ffn_down`.
That is **44 collectives per token instead of 5 hops**, and the 44:5 ratio is
structural to TP. A 2x2 hybrid changes the number of *participants* (halving
per-collective cost, since 2 nodes need 1 round instead of 2) but not the
number of collectives.

**The arithmetic**, per token at 4096:

| | pipeline (now) | TP 4-way |
| --- | ---: | ---: |
| matmul | 66.2 ms | 16.6 ms |
| attention (split by head) | 28.4 ms | 7.1 ms |
| other | 14.3 ms | 3.6 ms |
| **compute** | **108.8 ms** | **~27 ms** |
| communication | 61.4 ms (5 hops) | 44 collectives x C |
| gateway residual | 58.8 ms | 58.8 ms (unchanged) |

Break-even is therefore **C <= ~3.3 ms per collective**. Measured
point-to-point cost today is **~12.3 ms per hop** (251.4 s / 4096 tokens /
5 hops), so at a conservative 2x per collective the result is ~1100 ms per token
against a current total of 229 ms — about **5x slower**.

**The blocker is the transport stack, not the network.** §8.1 of ARCHITECTURE.md
establishes that `true_comm` is serialisation/framing CPU plus protocol stalls,
not wire time; links peak at ~9 Mbps of a gigabit. An 8 KB activation is
0.066 ms of gigabit wire time plus ~0.2-0.3 ms of LAN round-trip, so an
all-reduce near the hardware floor is ~0.6-0.9 ms — at which point TP would be
roughly **2x faster**, not 5x slower.

So TP is rejected **on the current transport**, with a ~8x gap in per-message
cost, rather than ruled out by the interconnect. The layers to strip are all
software and all listed elsewhere in this document: per-message JSON (§5.4),
HTTP headers (§5.4), VXLAN encapsulation (§4.2).

**What restructuring would actually require:**

1. **Sharding by tensor slice, not by layer.** Rewrites
   `dli-gguf-shard-writer`, the partition manifests,
   `partition_tensor_assignment`, the `stage_map` schema, and the contracts in
   `contracts/`.
   *Favourable accident:* TinyLlama has exactly **4 KV heads**, so 4-way TP maps
   to 1 KV head + 8 Q heads per node with no replication. This does not
   generalise past 4 nodes without replicating the KV cache.
2. **A collective primitive.** Use **hub-based reduce + broadcast (2 rounds)**,
   not ring all-reduce. Ring on 4 nodes is 2(N-1) = 6 sequential steps; at 8 KB
   messages latency dominates over bandwidth, so 6 steps is worse than 2. The
   network is a star through a switch and §8.1 confirms pinode->pinode is no
   cheaper than pinode->master, so a hub scheme fits the topology.
3. **Lockstep synchronisation.** All four nodes must run the same layer
   simultaneously — 44 barriers per token. Mitigations: CPU pinning
   (`isolcpus`/`taskset`), scheduling priority, `hostNetwork` to remove CNI
   jitter, and no other pods on stage nodes. Pipeline parallelism is
   jitter-tolerant by construction; TP is not.
4. **Not required:** persistent `TCP_NODELAY` sockets and a self-delimiting
   frame format already exist.

**Honest cost/benefit.** The transport work benefits *both* architectures:

| Scenario | compute | comm | residual | **per token** |
| --- | ---: | ---: | ---: | ---: |
| Now (pipeline) | 108.8 | 61.4 | 58.8 | **229 ms** |
| Pipeline + fast transport | 108.8 | ~2 | 58.8 | **~170 ms** |
| TP + fast transport | ~27 | ~35 | 58.8 | **~121 ms** |

TP wins by ~1.4x over an equally optimised pipeline — real, but not the 5x the
raw compute split suggests, and it costs a rewrite of the sharding tooling, the
ABI, and the scheduler. Speculative decoding (§5.1) delivers a comparable factor
for ~50 lines.

**When TP becomes the right call:** if the project's target moves to running a
7B-13B model at usable latency on edge nodes. At 13B Q4 (~7.5 GB) the pipeline
must stream ~750 ms of weights per token *regardless of transport quality*; only
TP divides that across buses. If the thesis stays "characterise distributed edge
inference with a small model", pipeline parallelism is correct and TP belongs in
the paper as the quantified rejected alternative.

### 8.2 Tensor parallelism combined with speculative decoding

**They compose, and nearly multiplicatively.** This is also the standard
combination in production serving stacks (vLLM, TensorRT-LLM), so it carries
little research risk.

Estimates below are built on the estimates above; treat them as orders of
magnitude.

| Scenario | compute | comm | residual | **per token** |
| --- | ---: | ---: | ---: | ---: |
| Now | 108.8 | 61.4 | 58.8 | **229 ms** |
| Pipeline + fast transport | 108.8 | ~2 | 58.8 | **~170 ms** |
| TP + fast transport | ~27 | ~35 | 58.8 | **~121 ms** |
| TP + transport + speculation (K=4) | ~35 | ~44 | ~85 | **~68 ms** (pass = 164 ms / 2.4 accepted) |

TP alone 1.9x; speculation alone ~2x; combined **~3.4x**.

**Why they reinforce each other:**

- **Speculation repairs TP's weak point.** The 44 collectives are per *pass*,
  not per token. At 2.4 accepted tokens per pass the effective cost falls to
  ~18 collectives per token — and barriers per token fall the same way, easing
  the jitter sensitivity from §8.1. In a real sense, speculation is what makes
  TP affordable.
- **TP does not consume the headroom speculation needs.** Under TP each node
  reads 1/4 of the weights *and* does 1/4 of the FLOPs, so arithmetic intensity
  stays ~3.3 FLOP/byte. The node remains equally memory-bound, so the idle-FLOP
  reserve that speculation spends is unchanged. They are orthogonal.
- **Larger messages do not hurt.** Activations grow from 8 KB to 32 KB (4
  positions), but cost is latency-dominated: 0.26 ms vs 0.066 ms of wire time,
  so per-collective cost moves ~0.8 -> ~1.0 ms. Negligible against amortisation
  over 2.4 tokens.

**Frictions to design for:**

- **Rollback becomes a synchronised operation.** On the pipeline each node owns
  whole layers, so KV truncation is local and independent. Under TP each node
  owns a slice of *every* layer's KV, so truncation must be consistent across
  all four. Divergence corrupts the model silently — this needs a consistency
  check, not just a decrement.
- **Proposer placement.** A draft *model* would force a choice between
  TP-splitting it (bad — it is small, collectives would dominate) and running it
  whole on the gateway. **Prompt-lookup sidesteps this entirely**: no model, runs
  in the gateway, neutral to stage topology. Another reason to start there.

**Note the punchline:** in the combined scenario the gateway residual is ~35 ms
of ~68 ms per accepted token — **over half**. See §2.2.

### 8.3 Order of architectural work

**Build speculative decoding first, regardless of the TP decision.** It is
~50 lines against a system rewrite; it works on the current architecture today;
it is the only one of the two that amortises the gateway residual that becomes
dominant under TP; and it **transfers unchanged**, because it lives in the
gateway while TP changes the stage layer.

This is not a choice between them. It is: speculation now, TP later if the
project's direction moves to larger models — at which point the speculation work
already done is what makes TP viable.

---

## 9. Missing experiments

### 9.1 1-stage and 2-stage baselines (§11.3, still pending)

The 674 MB model fits on one 4 GB Pi, so this is the control that measures what
distribution actually costs: 2 hops vs 5, and no inter-stage serialisation.

**Without it the paper has no baseline for its central claim.** Highest-value
missing measurement in the project.

### 9.2 Concurrency sweep (§11.4, still pending)

N = 1/2/4/8 simultaneous `/generate` requests at 256 tokens. Currently every run
reports "1 request currently in flight, drop-on-busy mode". This is the
empirical motivation for §5.3.

### 9.3 Shard imbalance is a future, not current, problem

Shards are 171.8 / 135.0 / 156.7 / 210.4 MB. Under the serial batch-1 pipeline
total latency is the **sum** of stages, so imbalance costs nothing today. The
moment continuous batching lands, the 210 MB stage becomes the throughput
limiter.

Worth stating explicitly in the paper: the `latency_balanced` partitioning is
optimal for the serial regime and suboptimal for the pipelined one.

---

## 10. Evaluated and rejected — do not re-attempt

### 10.1 KV cache or weights on SSD (FlexGen-style offloading)

Pi 5 NVMe over PCIe 2.0 x1 is ~450 MB/s against ~10 GB/s of LPDDR4X — a 20-25x
cliff, with ~50-100 us access latency that batch=1 decoding cannot hide.

And there is no capacity problem to solve: peak cgroup memory is **477 MB of
2000 MB** (§8) and KV at 4096 tokens is ~48 MB per node. Offloading trades free
RAM for 20x slower storage.

*Terminology note: "FlashAttention" is unrelated to flash storage. Its
IO-awareness concerns on-chip SRAM vs DRAM, which on a Pi maps to L1/L2/L3 vs
LPDDR — never to disk.*

### 10.2 PagedAttention

It is a memory-capacity optimisation, not a faster kernel. vLLM's gains come
from fitting more sequences in HBM, and its own paper acknowledges the paged
kernel is slower than a contiguous one.

Here the scarcity is inverted (bandwidth, not capacity): even at N = 8 and 2048
context, KV is ~201 MB per node against a 2000 MB limit. The current allocator
already grows on demand (`:617-618`, `:641-643`), so the over-reservation
PagedAttention targets does not exist in this codebase.

Worse, blocked storage breaks both the in-place K view and the transposed V
buffer (`:817-830`), which would either reintroduce the per-token re-gather that
was removed for a **74-75% attention win** (§8) or require the fused kernel
first.

**The useful part of vLLM here is continuous batching** (§5.3), which is
independent of paged memory.

### 10.3 GPU offload via the Vulkan backend (VideoCore VII)

Shares the *same* LPDDR4X, so the same ~10 GB/s ceiling, now contended with the
CPU, on a much weaker compute unit. A memory-bound kernel gains nothing by
moving to another processor on the same bus. Predictable from the roofline
without testing.

### 10.4 PCIe NPU (Hailo-8 / AI HAT)

Toolchains target INT8 CNNs, not generative transformers, and the accelerator
would be fed through the same ~450 MB/s PCIe 2.0 x1 link. It also occupies the
same slot as an SSD.

### 10.5 fp16 activations on the wire

Already supported and measured (§11.6): halves the payload exactly (25.6 ->
12.8 MiB) with **no wall-time change**, because the pipeline is round-trip
bound, not bandwidth bound.

### 10.6 Compressing activations on the wire

Would spend CPU to save bytes on the resource that is not scarce, and add to the
resource that is (§1.1b). Strictly the wrong direction.

### 10.7 Activation sparsity (Deja Vu / PowerInfer)

Requires ReLU-family sparsity; LLaMA/TinyLlama uses SiLU, whose activations are
not naturally sparse. "ReLUfication" needs fine-tuning, which is out of scope.

### 10.8 MoE architectures

Fewer bytes read per token for equal quality, but it changes the model and moves
cost onto the routing/communication path — the already-saturated side here. See
`sota-papers/` (`CacheMoE`, `WDMoE`, `Think_Fast_Infer_Smart`) for the
communication overheads involved.

### 10.9 Native stage chaining (`native_stage_chaining_enabled`)

Measured ~6% *slower* at 2048/4096 (§8.1): it relocates frame serialisation onto
the already CPU-saturated stage nodes, while orchestrated routing keeps that
work on the spare-CPU gateway. The star network means pinode->pinode is no
cheaper than pinode->master, so there are no hop savings to offset it. Keep it
off.

---

## 11. Where an SSD does help

Not on the inference path, but these are real and worth measuring separately:

- **Cold start.** Loading 135-210 MB shards and pulling container images from an
  SD card dominates time-to-first-token after a pod restart. Worth a distinct
  operational metric.
- **Swapping inactive sessions.** Parking the KV of *idle* sessions — never the
  active one — is vLLM's CPU swap space idea, and pairs with §7.1.
- **k3s/etcd and logs.** The SD card is the likely source of control-plane
  latency.

The distributed architecture **is** the alternative to offloading: the cluster
aggregates 16 GB of RAM and 16 cores. A 7B Q4 model (~4.5 GB) is ~1.1 GB per
node and a 13B Q4 (~7.5 GB) is ~1.9 GB per node — both fit in RAM. Adding nodes
beats adding storage.

---

## 12. Suggested order of work

1. **§2 Tier 0 instrumentation.** Without `overhead_ms`/`elementwise_ms` and a
   gateway flamegraph, gains cannot be attributed and §3 is being estimated
   rather than measured.
2. **§3.1, §3.2, §3.6** — norm hoisting, RoPE table, `nth_element`. Under a day,
   bit-identical, risk-free, and they audit the ~58 s remainder.
3. **§4.1, §4.2** — thread count and the pod network path. Config-level, and
   §4.2 informs whether §5.4 is worth its effort.
4. **§5.1 speculative decoding.** Best remaining ratio of payoff to disruption,
   and it transfers to any future architecture.
5. **§6.1 sliding window, with a quality axis.** The only item that attacks
   complexity rather than constants; strongest paper contribution.
6. **§9.1 the 1-stage baseline.** Needed for the paper regardless of any
   optimisation work.
7. **§5.2 unified ggml graph**, then §5.3 continuous batching and §6.3 flash
   attention on top of it.
8. **§8 architectural restructuring** — only if the project's target moves to
   larger models.

---

## 13. Open questions

- Does the ~58 s "other" bucket decompose the way §3 predicts? Estimates there
  (~2 s RoPE, ~1.7 s SiLU, ~2.5 s context churn per node) are **paper
  calculations, not measurements**. §2.1 settles it.
- What is the actual per-message floor after §5.4 and §4.2? This single number
  decides whether §8.1 is ever worth revisiting.
- What acceptance rate does prompt-lookup speculation achieve on non-degenerate
  text? Determines whether §5.1 is a 2x or a 1.2x.
- Does a sliding window improve or degrade quality at 4096, given
  `max_position_embeddings = 2048`? §6.1 predicts improvement; unverified.
