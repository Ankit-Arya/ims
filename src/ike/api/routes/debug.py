from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ike.api.deps import require_analyst
from ike.db.models import User
from ike.db.session import get_db
from ike.retrieval.engine import RetrievalEngine

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/retrieval")
def debug_retrieval(
    q: str = Query(min_length=2, max_length=2000),
    user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
) -> dict:
    evidence, trace = RetrievalEngine(db).retrieve(q, user)
    return {
        "trace": trace,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "document": item.candidate.document_title,
                "page_from": item.candidate.page_from,
                "page_to": item.candidate.page_to,
                "section_path": item.candidate.section_path,
                "rerank_score": item.candidate.rerank_score,
                "sources": sorted(item.candidate.sources),
                "text": item.candidate.text,
            }
            for item in evidence
        ],
    }
