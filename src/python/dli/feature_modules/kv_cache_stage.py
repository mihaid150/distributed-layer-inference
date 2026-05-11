from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any, Optional

import torch

from dli.common.feature_flags import FeatureFlags
from dli.common.timing import now_ms

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - import failure is surfaced when the module is enabled.
    DynamicCache = None  # type: ignore[assignment]


@dataclass
class StageKVCacheEntry:
    request_id: str
    cache: Any
    created_at_ms: float
    last_access_ms: float
    lock: RLock = field(default_factory=RLock)


class StageForwardCache:
    """
    Stage-local idempotency cache for duplicate forwards.

    This supports the transformer KV cache below: if an HTTP retry repeats the same
    decode step, the stage returns the previous activation instead of appending the
    same token into the live KV state a second time.
    """

    KEY = "forward_dedupe_cache"

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, max_entries)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return bool(getattr(flags, "forward_dedupe_enabled", False))

    @staticmethod
    def build_key(
        *,
        request_id: str,
        token_index: int,
        tensor: torch.Tensor,
        generation_mode: str = "legacy",
        cache_position_start: Optional[int] = None,
    ) -> str:
        blob = tensor.detach().cpu().numpy().tobytes()
        digest = hashlib.sha1(blob).hexdigest()
        position = "none" if cache_position_start is None else str(cache_position_start)
        return f"{request_id}:{token_index}:{generation_mode}:{position}:{digest}"

    def get(self, key: str) -> Optional[torch.Tensor]:
        with self._lock:
            tensor = self._store.get(key)
            if tensor is None:
                return None
            self._store.move_to_end(key)
            return tensor.clone()

    def put(self, key: str, value: torch.Tensor) -> None:
        with self._lock:
            self._store[key] = value.detach().cpu().clone()
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._store)


class StageKVCacheManager:
    """
    Stage-local transformer past_key_values cache.

    The cache is keyed by gateway request_id. Each inference stage owns only the KV
    entries for its local transformer layers; the gateway controls prefill/decode
    sequencing through request metadata.
    """

    KEY = "kv_cache"

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, StageKVCacheEntry] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return bool(flags.kv_cache_enabled)

    def get_or_create(self, *, request_id: str, config: Any) -> tuple[StageKVCacheEntry, bool]:
        if DynamicCache is None:
            raise RuntimeError("transformers DynamicCache is unavailable.")

        now = now_ms()
        with self._lock:
            existing = self._entries.get(request_id)
            if existing is not None:
                existing.last_access_ms = now
                self._entries.move_to_end(request_id)
                return existing, False

            entry = StageKVCacheEntry(
                request_id=request_id,
                cache=DynamicCache(config=config),
                created_at_ms=now,
                last_access_ms=now,
            )
            self._entries[request_id] = entry
            self._evict_if_needed()
            return entry, True

    def get(self, request_id: str) -> Optional[StageKVCacheEntry]:
        now = now_ms()
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            entry.last_access_ms = now
            self._entries.move_to_end(request_id)
            return entry

    def reset(self, request_id: str) -> None:
        with self._lock:
            self._entries.pop(request_id, None)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def seq_length(self, *, request_id: str, layer_idx: int = 0) -> int:
        entry = self.get(request_id)
        if entry is None:
            return 0
        try:
            return int(entry.cache.get_seq_length(layer_idx))
        except Exception:
            return 0

    def describe(self, *, request_id: str, layer_idx: int = 0) -> dict[str, Any]:
        entry = self.get(request_id)
        if entry is None:
            return {
                "request_id": request_id,
                "cache_present": False,
                "entries": self.size(),
                "seq_length": 0,
            }

        return {
            "request_id": request_id,
            "cache_present": True,
            "entries": self.size(),
            "seq_length": self.seq_length(request_id=request_id, layer_idx=layer_idx),
            "created_at_ms": entry.created_at_ms,
            "last_access_ms": entry.last_access_ms,
        }

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
