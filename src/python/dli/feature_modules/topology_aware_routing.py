from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

import requests

from dli.common.feature_flags import FeatureFlags
from dli.common.timing import elapsed_ms, now_ms


@dataclass
class RouteObservation:
    url: str
    rtt_ms: float
    bandwidth_mbps: float
    observed_at_ms: float
    source: str


class TopologyAwareRoutingModule:
    KEY = "topology_aware"
    _observations: Dict[str, RouteObservation] = {}
    _lock = threading.Lock()

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

    @classmethod
    def build_route_overrides(
        cls,
        *,
        flags: FeatureFlags,
        route_candidates: Dict[str, list[str]],
        probe_timeout_seconds: float = 0.25,
    ) -> Dict[str, str]:
        if not cls.is_enabled(flags):
            return {}

        overrides: Dict[str, str] = {}
        for stage_id, candidates in route_candidates.items():
            normalized_candidates = [url for url in candidates if isinstance(url, str) and url.strip()]
            if not normalized_candidates:
                continue
            if len(normalized_candidates) == 1:
                overrides[str(stage_id)] = normalized_candidates[0]
                continue

            best_url = cls._select_best_candidate(
                normalized_candidates,
                probe_timeout_seconds=probe_timeout_seconds,
            )
            if best_url:
                overrides[str(stage_id)] = best_url

        return overrides

    @classmethod
    def observe_stage_metrics(cls, metrics: Iterable[Dict[str, Any]]) -> None:
        for metric in metrics:
            url = metric.get("next_stage_url") or metric.get("stage1_url")
            if not isinstance(url, str) or not url.strip():
                continue

            transfer_time_ms = float(
                metric.get("rpc_wall_time_ms", metric.get("transfer_time_ms", 0.0)) or 0.0
            )
            estimated_link_mbps = float(metric.get("estimated_link_mbps", 0.0) or 0.0)
            if transfer_time_ms <= 0.0 and estimated_link_mbps <= 0.0:
                continue

            cls._record_observation(
                url=url.strip(),
                rtt_ms=max(0.0, transfer_time_ms),
                bandwidth_mbps=max(0.0, estimated_link_mbps),
                source="stage_metric",
            )

    @classmethod
    def _select_best_candidate(
        cls,
        candidates: list[str],
        *,
        probe_timeout_seconds: float,
    ) -> Optional[str]:
        observations = [
            cls._observation_for_candidate(
                url,
                probe_timeout_seconds=probe_timeout_seconds,
            )
            for url in candidates
        ]
        observations = [item for item in observations if item is not None]
        if not observations:
            return candidates[0] if candidates else None

        observations.sort(key=cls._route_score)
        return observations[0].url

    @classmethod
    def _observation_for_candidate(
        cls,
        url: str,
        *,
        probe_timeout_seconds: float,
    ) -> Optional[RouteObservation]:
        with cls._lock:
            existing = cls._observations.get(url)
        if existing is not None:
            return existing

        return cls._probe_candidate(url, timeout_seconds=probe_timeout_seconds)

    @classmethod
    def _probe_candidate(cls, url: str, *, timeout_seconds: float) -> Optional[RouteObservation]:
        health_url = cls._health_url_for_forward_url(url)
        start = now_ms()
        try:
            response = requests.get(health_url, timeout=timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            return None

        rtt_ms = elapsed_ms(start)
        response_bytes = max(1, len(response.content))
        bandwidth_mbps = (
            response_bytes * 8.0 / (rtt_ms / 1000.0) / 1_000_000.0
            if rtt_ms > 0.0
            else 0.0
        )
        return cls._record_observation(
            url=url,
            rtt_ms=rtt_ms,
            bandwidth_mbps=bandwidth_mbps,
            source="health_probe",
        )

    @classmethod
    def _record_observation(
        cls,
        *,
        url: str,
        rtt_ms: float,
        bandwidth_mbps: float,
        source: str,
    ) -> RouteObservation:
        observation = RouteObservation(
            url=url,
            rtt_ms=rtt_ms,
            bandwidth_mbps=bandwidth_mbps,
            observed_at_ms=now_ms(),
            source=source,
        )
        with cls._lock:
            cls._observations[url] = observation
        return observation

    @staticmethod
    def _route_score(observation: RouteObservation) -> float:
        bandwidth_penalty = 10_000.0
        if observation.bandwidth_mbps > 0.0:
            bandwidth_penalty = 1_000.0 / observation.bandwidth_mbps
        return max(0.0, observation.rtt_ms) + bandwidth_penalty

    @staticmethod
    def _health_url_for_forward_url(url: str) -> str:
        parsed = urlparse(url)
        return parsed._replace(path="/health", query="", fragment="").geturl()
