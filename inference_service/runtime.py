from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from prometheus_client import Gauge, Histogram


INFERENCE_QUEUE_WAIT = Histogram(
    "ike_inference_queue_wait_seconds",
    "Time spent waiting for local inference capacity",
    ["stage", "role"],
)
INFERENCE_EXECUTION = Histogram(
    "ike_inference_execution_seconds",
    "Time spent executing local inference",
    ["stage", "role"],
)
INFERENCE_WAITING = Gauge(
    "ike_inference_waiting",
    "Requests currently waiting for local inference capacity",
    ["stage", "role"],
)
INFERENCE_ACTIVE = Gauge(
    "ike_inference_active",
    "Requests currently executing local inference",
    ["stage", "role"],
)


@dataclass(slots=True)
class StageTiming:
    queue_wait_ms: int = 0
    execution_ms: int = 0


class TimedGate:
    """A bounded local model gate with explicit queue/execution timing.

    Separate inference-query and inference-ingest processes own separate gates, so
    background document embedding can never acquire the gate used by live Q&A.
    """

    def __init__(self, *, stage: str, role: str, concurrency: int = 1) -> None:
        self.stage = stage
        self.role = role
        self._semaphore = BoundedSemaphore(max(1, concurrency))
        self._state_lock = Lock()
        self._waiting = 0
        self._active = 0

    @contextmanager
    def acquire(self):
        wait_started = time.perf_counter()
        with self._state_lock:
            self._waiting += 1
            INFERENCE_WAITING.labels(self.stage, self.role).set(self._waiting)
        self._semaphore.acquire()
        queue_wait = time.perf_counter() - wait_started
        with self._state_lock:
            self._waiting -= 1
            self._active += 1
            INFERENCE_WAITING.labels(self.stage, self.role).set(self._waiting)
            INFERENCE_ACTIVE.labels(self.stage, self.role).set(self._active)
        INFERENCE_QUEUE_WAIT.labels(self.stage, self.role).observe(queue_wait)

        execution_started = time.perf_counter()
        try:
            yield StageTiming(queue_wait_ms=int(queue_wait * 1000))
        finally:
            execution = time.perf_counter() - execution_started
            INFERENCE_EXECUTION.labels(self.stage, self.role).observe(execution)
            with self._state_lock:
                self._active -= 1
                INFERENCE_ACTIVE.labels(self.stage, self.role).set(self._active)
            self._semaphore.release()
