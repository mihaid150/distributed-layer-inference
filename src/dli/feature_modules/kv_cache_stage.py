from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Optional

import torch

from dli.common.feature_flags import FeatureFlags


class StageForwardCache:
    """
    Stage-local cache for idempotent/deduplicated forwards.

    This does not replace full transformer KV-cache internals yet.
    It avoids recompute when the same request_id/token_index/input arrives again
    (e.g., retries/timeouts causing duplicate forwards).
    """

    KEY = "kv_cache"

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max(1, max_entries)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.kv_cache_enabled

    @staticmethod
    def build_key(
        *,
        request_id: str,
        token_index: int,
        tensor: torch.Tensor,
    ) -> str:
        blob = tensor.detach().cpu().numpy().tobytes()
        digest = hashlib.sha1(blob).hexdigest()
        return f"{request_id}:{token_index}:{digest}"

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

