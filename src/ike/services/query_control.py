from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from redis import Redis

from ike.core.config import get_settings


class QueryCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class QueryControl:
    request_id: UUID
    user_id: UUID
    redis: Redis = field(init=False, repr=False)
    key: str = field(init=False)

    def __post_init__(self) -> None:
        settings = get_settings()
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.key = f"ims:query-cancel:{self.user_id}:{self.request_id}"

    def cancel(self) -> None:
        self.redis.setex(self.key, 3600, "1")

    def is_cancelled(self) -> bool:
        try:
            return bool(self.redis.exists(self.key))
        except Exception:
            # Availability of cancellation control must not break Q&A.
            return False

    def checkpoint(self) -> None:
        if self.is_cancelled():
            raise QueryCancelled("Query cancelled by user")

    def clear(self) -> None:
        try:
            self.redis.delete(self.key)
        except Exception:
            pass
