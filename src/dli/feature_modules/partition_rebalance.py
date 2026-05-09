from __future__ import annotations

from typing import Any, Dict

from dli.common.feature_flags import FeatureFlags


class PartitionRebalanceModule:
    KEY = "rebalance"

    PROFILE_DESCRIPTIONS = {
        "baseline": "Original static split from baseline deployment.",
        "latency_balanced_v1": (
            "Latency-balanced profile marker. Requires pre-generated rebalance partitions"
            " and matching rollout manifests to take full effect."
        ),
    }

    @staticmethod
    def profile(flags: FeatureFlags) -> str:
        return flags.rebalance_profile

    @staticmethod
    def describe(flags: FeatureFlags) -> Dict[str, Any]:
        profile = flags.rebalance_profile
        return {
            "profile": profile,
            "description": PartitionRebalanceModule.PROFILE_DESCRIPTIONS.get(profile, "Unknown"),
            "runtime_applied": profile != "baseline",
        }

