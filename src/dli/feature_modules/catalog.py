from __future__ import annotations


FEATURE_MODULE_CATALOG = [
    {
        "key": "binary_transport",
        "title": "Binary Activation Transport",
        "summary": "Replace JSON+base64 activations with binary octet-stream envelopes.",
    },
    {
        "key": "activation_precision",
        "title": "Activation Precision Reduction",
        "summary": "Send FP16/BF16/INT8 activations per hop to reduce payload size.",
    },
    {
        "key": "kv_cache",
        "title": "Stage KV/Forward Cache",
        "summary": (
            "Enable stage-side cache for duplicate forward requests. (Current implementation"
            " is dedupe/retry cache; full transformer KV state cache can be extended here.)"
        ),
    },
    {
        "key": "rebalance",
        "title": "Partition Rebalance Profile",
        "summary": "Select rebalance profile marker for comparative experiments.",
    },
    {
        "key": "topology_aware",
        "title": "Topology-Aware Routing",
        "summary": "Enable topology-aware next-hop URL override policy.",
    },
    {
        "key": "persistent_sessions_backpressure",
        "title": "Persistent Sessions + Backpressure",
        "summary": "Reuse HTTP sessions and apply bounded single-flight/queue backpressure.",
    },
]

