from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ike.agent.tools import CorpusTools
from ike.api.deps import require_analyst
from ike.db.models import User
from ike.db.session import get_db

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/retrieval")
def debug_retrieval(
    q: str = Query(min_length=2, max_length=2000),
    user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> dict:
    """Expose the same raw hybrid search observation available to the v4 agent."""

    result = CorpusTools(db, user).search(q)
    return {
        "tool": result.tool,
        "arguments": result.arguments,
        "metadata": result.metadata,
        "elapsed_ms": result.elapsed_ms,
        "note": result.note,
        "evidence": result.items,
    }
