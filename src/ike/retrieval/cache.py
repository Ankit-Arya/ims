from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from typing import Any, Hashable


class TTLCache:
    """Small process-local TTL/LRU cache for non-answer retrieval metadata.

    Values are copied on read/write so request-local mutation cannot corrupt a cached
    object. Callers are responsible for including ACL and corpus-revision information in
    the key. IMS deliberately does not cache final answers across users in this release.
    """

    def __init__(self, *, max_items: int = 512, ttl_seconds: int = 600) -> None:
        self.max_items = max(1, max_items)
        self.ttl_seconds = max(1, ttl_seconds)
        self._items: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: Hashable) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            expires, value = item
            if expires <= now:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return deepcopy(value)

    def set(self, key: Hashable, value: Any) -> None:
        expires = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._items[key] = (expires, deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
