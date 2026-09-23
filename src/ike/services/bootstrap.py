import logging

from sqlalchemy import select

from ike.core.config import get_settings
from ike.core.security import hash_password
from ike.db.models import User
from ike.db.session import SessionLocal

logger = logging.getLogger(__name__)


def ensure_bootstrap_admin() -> None:
    settings = get_settings()
    with SessionLocal() as db:
        existing = db.scalar(select(User).where(User.username == settings.bootstrap_admin_username))
        if existing:
            return
        if settings.app_env == "production" and settings.bootstrap_admin_password == "change-this-before-running":
            raise RuntimeError("Refusing to create production admin with the default password")
        user = User(
            username=settings.bootstrap_admin_username,
            password_hash=hash_password(settings.bootstrap_admin_password),
            role="admin",
            department=settings.bootstrap_admin_department,
        )
        db.add(user)
        db.commit()
        logger.warning("bootstrap_admin_created", extra={"username": user.username})
