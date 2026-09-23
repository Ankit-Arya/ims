from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: UUID
    username: str
    role: str
    department: str | None

    model_config = {"from_attributes": True}


class UserAdminOut(UserOut):
    is_active: bool
    created_at: datetime


class UserDirectoryOut(BaseModel):
    id: UUID
    username: str
    department: str | None

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=12, max_length=200)
    role: str = Field(default="user", pattern="^(admin|analyst|user)$")
    department: str | None = Field(default=None, max_length=120)


class UserStatusUpdate(BaseModel):
    is_active: bool


class UserPasswordReset(BaseModel):
    password: str = Field(min_length=12, max_length=200)
