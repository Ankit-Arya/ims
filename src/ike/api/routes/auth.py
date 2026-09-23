from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user, require_admin
from ike.core.config import get_settings
from ike.core.security import create_access_token, hash_password, verify_password
from ike.db.models import User
from ike.db.session import get_db
from ike.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserAdminOut,
    UserCreate,
    UserDirectoryOut,
    UserOut,
    UserPasswordReset,
    UserStatusUpdate,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.username == payload.username))
    if not user or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    token = create_access_token(user.id, user.username, user.role)
    settings = get_settings()
    response.set_cookie(
        "ike_session",
        token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=8 * 60 * 60,
        path="/",
    )
    return TokenResponse(access_token=token)


@router.post("/logout", status_code=204)
def logout(response: Response) -> None:
    response.delete_cookie("ike_session", path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.get("/directory", response_model=list[UserDirectoryOut])
def user_directory(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[User]:
    """Return active colleagues that may receive access to a personal PDF."""
    return list(
        db.scalars(
            select(User)
            .where(User.is_active.is_(True), User.id != user.id)
            .order_by(User.username.asc())
        ).all()
    )


@router.get("/users", response_model=list[UserAdminOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> list[User]:
    return list(db.scalars(select(User).order_by(User.created_at.asc())).all())


@router.post("/users", response_model=UserAdminOut, status_code=201)
def create_user(payload: UserCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)) -> User:
    if db.scalar(select(User).where(User.username == payload.username)):
        raise HTTPException(status_code=409, detail="Username already exists")
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role=payload.role,
        department=payload.department,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/users/{user_id}/status", response_model=UserAdminOut)
def set_user_status(
    user_id: UUID,
    payload: UserStatusUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id and not payload.is_active:
        raise HTTPException(status_code=409, detail="You cannot deactivate your own active administrator session")
    user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return user


@router.post("/users/{user_id}/reset-password", status_code=204)
def reset_user_password(
    user_id: UUID,
    payload: UserPasswordReset,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> None:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(payload.password)
    db.commit()
