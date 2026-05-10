# Ablation Matrix Analysis (Gateway `/generate`)

Date: 2026-05-10  
Source: Ops UI history runs visible in dashboard screenshots (IDs 1-16)

## 1) Fixed test setup

- Target: `gateway /generate`
- Prompt tokens: `9`
- Prompt chars: `53`
- Generation: `max_new_tokens=48`, `min_new_tokens=48`
- Timeout: `300000 ms`
- Same topology hash family (same stage placement pattern)

Note: Run `#1` latency appeared in UI as `2046311`; based on `tok/s` it is interpreted as `~204631.1 ms` (formatting inconsistency).

## 2) Raw ablation runs (key metrics)

| ID | Profile/Modules | Variant summary | Latency ms | Tok/s | Comm/Comp | Net MiB |
|---:|---|---|---:|---:|---:|---:|
| 1 | baseline | json+base64, fp32 | 204631.1 | 0.235 | 2.806 | 108.325 |
| 2 | activation_precision | json+base64, fp16 | 201718.4 | 0.238 | 2.747 | 59.185 |
| 3 | activation_precision | json+base64, bf16 | 202872.5 | 0.237 | 2.757 | 58.274 |
| 4 | activation_precision | json+base64, int8 | 200723.4 | 0.239 | 2.722 | 32.909 |
| 5 | binary_transport | octet-stream, fp32 | 196651.8 | 0.244 | 2.861 | 121.697 |
| 6 | activation_precision + binary_transport | octet-stream, fp16 | 193548.2 | 0.248 | 2.715 | 66.031 |
| 7 | activation_precision + binary_transport | octet-stream, bf16 | 213826.0 | 0.224 | 2.623 | 65.443 |
| 8 | rebalance | json+base64, fp32, latency-balanced | 210204.9 | 0.228 | 2.635 | 109.008 |
| 9 | binary_transport + rebalance | octet-stream, fp32 | 219472.7 | 0.219 | 2.685 | 123.125 |
| 10 | activation_precision + binary_transport + rebalance | octet-stream, fp16 | 213074.0 | 0.225 | 2.612 | 66.044 |
| 11 | activation_precision + binary_transport + rebalance | octet-stream, bf16 | 211313.4 | 0.227 | 2.600 | 65.518 |
| 12 | binary_transport + kv_cache + rebalance | octet-stream, fp32 | 223028.5 | 0.215 | 2.701 | 122.356 |
| 13 | activation_precision + binary_transport + kv_cache + rebalance | octet-stream, fp16 | 224015.1 | 0.214 | 2.667 | 66.086 |
| 14 | activation_precision + binary_transport + kv_cache + rebalance + topology | octet-stream, fp16 | 224076.6 | 0.214 | 2.667 | 65.087 |
| 15 | full stack (sessions on) | octet-stream, fp16, kv on, rebalance on, topology on | 222035.8 | 0.216 | 2.638 | 63.852 |
| 16 | full stack + backpressure (queue=2) | same as #15 + backpressure | 221822.0 | 0.216 | 2.636 | 63.825 |

## 3) Grouped averages vs baseline

Baseline reference: run `#1`

| Group | n | Avg latency ms | Avg tok/s | Avg comm/comp | Avg net MiB | Latency Δ vs baseline | Tok/s Δ | Comm/Comp Δ | Net Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| activation_precision (fp16/bf16/int8) | 3 | 201771.4 | 0.238 | 2.742 | 50.123 | -1.40% | +1.28% | -2.28% | -53.73% |
| binary_transport only | 1 | 196651.8 | 0.244 | 2.861 | 121.697 | -3.90% | +3.83% | +1.96% | +12.34% |
| activation_precision + binary_transport | 2 | 203687.1 | 0.236 | 2.669 | 65.737 | -0.46% | +0.43% | -4.88% | -39.32% |
| activation_precision + binary_transport + rebalance | 2 | 212193.7 | 0.226 | 2.606 | 65.781 | +3.70% | -3.83% | -7.13% | -39.27% |
| full stack (sessions/topology/kv/backpressure variants) | 2 | 221928.9 | 0.216 | 2.637 | 63.839 | +8.45% | -8.09% | -6.02% | -41.07% |

## 4) Interpretation

1. **System remains communication-heavy.**  
   `Comm/Comp` stays roughly `2.6–2.8` across all configurations, so transfer overhead still dominates.

2. **Activation precision is the strongest traffic reducer.**  
   With precision changes on JSON transport (runs 2-4), network drops by ~54% average vs baseline.

3. **INT8 cuts traffic the most but quality regressed in this dataset.**  
   Run `#4` has lowest network (`32.9 MiB`) but generated text quality visibly degraded, so it is not a safe default.

4. **Binary transport alone did not reduce traffic here.**  
   Run `#5` improved latency but increased network volume. This suggests transport swap without precision reduction is not enough and may add overhead depending on payload path/serialization details.

5. **Best balanced point in this matrix is fp16 + binary transport (run #6).**  
   It delivered the best observed latency (`193548.2 ms`) and still large network reduction (`66.0 MiB`) vs baseline.

6. **Rebalance/KV/topology/session/backpressure stack did not improve latency in these runs.**  
   These combinations mostly reduced net traffic but increased wall latency compared to baseline and especially compared to run `#6`.

7. **Backpressure effect is minimal in single-request sequential tests.**  
   Run `#16` vs `#15` is almost identical; this is expected without concurrency pressure.

## 5) Practical conclusion for current objective

- If objective is **faster response while preserving output quality**, current best candidate is:
  - `transport=binary_octet_stream`
  - `precision=fp16`
  - keep `kv/topology/rebalance/backpressure` off until independently validated.

- If objective is **maximum network reduction**, `activation_precision=int8` is strongest but currently unsafe for quality.

## 6) Recommended next ablation phase

Run 5 repetitions per config (same prompt + same token settings), then compare mean/std/p95:

1. baseline (`json+fp32`)
2. precision only (`json+fp16`)
3. binary only (`binary+fp32`)
4. binary + precision (`binary+fp16`)
5. +rebalance
6. +kv_cache
7. +topology
8. +sessions
9. +backpressure under concurrent load (not single request)

This will separate true gains from run-to-run variance and produce publishable confidence bounds.

