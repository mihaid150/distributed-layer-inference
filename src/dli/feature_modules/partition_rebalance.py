from __future__ import annotations

import os
from typing import Any, Dict

from dli.common.feature_flags import FeatureFlags


class PartitionRebalanceModule:
    KEY = "rebalance"

    PROFILE_DESCRIPTIONS = {
        "baseline": "Original static split from baseline deployment.",
        "latency_balanced_v1": (
            "Latency-balanced 4-stage split that moves one decoder layer from stage 4"
            " to stage 3 when matching partition files are available."
        ),
    }

    LAYER_RANGES = {
        "baseline": None,
        "latency_balanced_v1": ["0-5", "6-11", "12-17", "18-21"],
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
            "layer_ranges": PartitionRebalanceModule.LAYER_RANGES.get(profile),
            "runtime_applied": profile != "baseline",
        }

    @staticmethod
    def resolve_partition_file(
        *,
        profile: str,
        partition_profiles: Dict[str, str],
        baseline_partition_file: str,
    ) -> tuple[str, bool]:
        selected = partition_profiles.get(profile)
        if not selected:
            return baseline_partition_file, False

        return selected, os.path.abspath(selected) != os.path.abspath(baseline_partition_file)
