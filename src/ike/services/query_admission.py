from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Callable, TypeVar

from ike.core.config import get_settings
from ike.services.query_metrics import QUERY_ACTIVE, QUERY_ADMISSION_WAIT, QUERY_WAITING

T = TypeVar("T")


class QueryCapacityError(RuntimeError):
    pass


class QueryAdmission:
    """Bounded per-API execution pool.

    API replication improves HTTP concurrency, but each replica still limits expensive
    Q&A workflows so a burst cannot create an unbounded number of local inference calls.
    The inference-query service independently serializes/bounds model execution.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.max_active = max(1, settings.query_max_active_per_api)
        self.max_waiting = max(0, settings.query_max_waiting_per_api)
        self.instance = os.getenv("HOSTNAME", "api")
        self._capacity = BoundedSemaphore(self.max_active + self.max_waiting)
        self._executor = ThreadPoolExecutor(max_workers=self.max_active, thread_name_prefix="qa")
        self._state_lock = Lock()
        self._waiting = 0
        self._active = 0

    def _set_waiting(self, delta: int) -> None:
        with self._state_lock:
            self._waiting += delta
            QUERY_WAITING.labels(self.instance).set(max(0, self._waiting))

    def _set_active(self, delta: int) -> None:
        with self._state_lock:
            self._active += delta
            QUERY_ACTIVE.labels(self.instance).set(max(0, self._active))

    def submit(self, fn: Callable[[], T], *, on_wait: Callable[[int], None] | None = None) -> Future[T]:
        if not self._capacity.acquire(blocking=False):
            raise QueryCapacityError("Q&A capacity is temporarily full; retry shortly.")
        enqueued = time.perf_counter()
        self._set_waiting(1)

        def wrapped() -> T:
            wait_ms = int((time.perf_counter() - enqueued) * 1000)
            self._set_waiting(-1)
            QUERY_ADMISSION_WAIT.labels(self.instance).observe(max(0, wait_ms) / 1000)
            self._set_active(1)
            if on_wait:
                on_wait(wait_ms)
            try:
                return fn()
            finally:
                self._set_active(-1)
                self._capacity.release()

        return self._executor.submit(wrapped)


query_admission = QueryAdmission()
