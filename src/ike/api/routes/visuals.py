from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ike.api.deps import get_current_user
from ike.db.models import User
from ike.db.session import get_db
from ike.services.visuals import render_visual_to_cache

router = APIRouter(prefix="/visuals", tags=["visuals"])


@router.get("/render/{token}")
def render_visual(
    token: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    try:
        path = render_visual_to_cache(db, user, token)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid or expired visual reference") from exc
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
