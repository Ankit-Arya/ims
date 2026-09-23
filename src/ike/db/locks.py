from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session


def advisory_key(value: UUID) -> int:
    """Map a UUID deterministically into PostgreSQL's positive signed-bigint lock space."""
    return value.int & 0x7FFF_FFFF_FFFF_FFFF


def try_advisory_lock(db: Session, value: UUID) -> bool:
    return bool(db.scalar(select(func.pg_try_advisory_lock(advisory_key(value)))))


def advisory_unlock(db: Session, value: UUID) -> None:
    db.scalar(select(func.pg_advisory_unlock(advisory_key(value))))
