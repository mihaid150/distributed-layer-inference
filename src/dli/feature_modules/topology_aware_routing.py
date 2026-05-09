from __future__ import annotations

from typing import Any, Dict, Optional

from dli.common.feature_flags import FeatureFlags


class TopologyAwareRoutingModule:
    KEY = "topology_aware"

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.topology_aware_routing

    @staticmethod
    def resolve_next_stage_url(
        *,
        stage_id: int,
        default_next_stage_url: Optional[str],
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        if not default_next_stage_url:
            return None

        feature_flags = metadata.get("feature_flags") or {}
        if not feature_flags.get("topology_aware_routing"):
            return default_next_stage_url

        overrides = metadata.get("topology_route_overrides") or {}
        stage_key = str(stage_id)
        override = overrides.get(stage_key) or overrides.get(f"stage_{stage_id}")
        if isinstance(override, str) and override.strip():
            return override.strip()

        return default_next_stage_url

