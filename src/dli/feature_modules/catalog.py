from __future__ import annotations


FEATURE_MODULE_CATALOG = [
    {
        "key": "binary_transport",
        "title": "Binary Activation Transport",
        "summary": (
            "Replace JSON+base64 activations with binary octet-stream envelopes,"
            " with optional zlib compression for tensor blobs."
        ),
    },
    {
        "key": "activation_precision",
        "title": "Activation Precision Reduction",
        "summary": (
            "Send FP16/BF16 activations per hop and restore source dtype before compute;"
            " INT8 remains experimental."
        ),
    },
    {
        "key": "kv_cache",
        "title": "Stage Transformer KV Cache",
        "summary": (
            "Run prefill once, then decode one token at a time with per-stage"
            " past_key_values state."
        ),
    },
    {
        "key": "forward_dedupe_cache",
        "title": "Forward Dedupe Cache",
        "summary": "Optional retry/idempotency cache for duplicate forward requests.",
    },
    {
        "key": "rebalance",
        "title": "Partition Rebalance Profile",
        "summary": "Load alternate partition files for real rebalance profile experiments.",
    },
    {
        "key": "topology_aware",
        "title": "Topology-Aware Routing",
        "summary": "Auto-select next-hop route overrides from measured RTT/bandwidth observations.",
    },
    {
        "key": "payload_compression",
        "title": "Binary Payload Compression",
        "summary": "Optionally compress binary tensor blobs when compression wins on byte size.",
    },
    {
        "key": "persistent_sessions_backpressure",
        "title": "Persistent Sessions + Backpressure",
        "summary": (
            "Reuse HTTP sessions and expose pool reuse counters; backpressure is kept"
            " for concurrency experiments rather than single-request latency wins."
        ),
    },
]
