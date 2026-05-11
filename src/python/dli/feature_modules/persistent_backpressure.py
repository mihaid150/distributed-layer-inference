from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict

import requests

from dli.common.feature_flags import FeatureFlags


class PersistentSessionPool:
    KEY = "persistent_sessions_backpressure"
    _sessions: Dict[str, requests.Session] = {}
    _created_count = 0
    _reuse_count = 0
    _lock = threading.Lock()

    @staticmethod
    def use_persistent_session(flags: FeatureFlags) -> bool:
        return flags.persistent_sessions_enabled

    @classmethod
    def get_or_create(cls, key: str) -> requests.Session:
        with cls._lock:
            session = cls._sessions.get(key)
            if session is not None:
                cls._reuse_count += 1
                return session
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=64)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            cls._sessions[key] = session
            cls._created_count += 1
            return session

    @classmethod
    def snapshot(cls) -> Dict[str, int]:
        with cls._lock:
            return {
                "sessions": len(cls._sessions),
                "created_count": cls._created_count,
                "reuse_count": cls._reuse_count,
            }


@dataclass
class BackpressureSnapshot:
    inflight: int
    waiting: int
    queue_limit: int


class GatewayBackpressureGuard:
    def __init__(self) -> None:
        self._inflight = 0
        self._waiting = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @staticmethod
    def is_enabled(flags: FeatureFlags) -> bool:
        return flags.persistent_sessions_enabled or flags.backpressure_enabled

    def acquire(self, queue_limit: int) -> bool:
        limit = max(0, int(queue_limit))
        with self._cond:
            if self._inflight == 0:
                self._inflight = 1
                return True

            if limit == 0:
                return False

            if self._waiting >= limit:
                return False

            self._waiting += 1
            try:
                while self._inflight > 0:
                    self._cond.wait()
                self._inflight = 1
                return True
            finally:
                self._waiting -= 1

    def release(self) -> None:
        with self._cond:
            if self._inflight > 0:
                self._inflight -= 1
            self._cond.notify_all()

    def snapshot(self, queue_limit: int) -> BackpressureSnapshot:
        with self._lock:
            return BackpressureSnapshot(
                inflight=self._inflight,
                waiting=self._waiting,
                queue_limit=max(0, int(queue_limit)),
            )
