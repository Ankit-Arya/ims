import json
import time
from sqlalchemy import select

from ike.db.models import User
from ike.db.session import SessionLocal
from ike.retrieval.engine import RetrievalEngine
from ike.retrieval.query_plan import build_query_plan
from ike.workflows.evidence_planning import build_deterministic_evidence_plan
from ike.workflows.verification_policy import retrieval_quality_issues

QUESTIONS = [
    "Sc 7a",
    "Coupling speeds",
    "i'm on line-7, wind speed is 65..pls give instruction",
    "how to deal in high speed wind conditions",
]

with SessionLocal() as db:
    user = db.scalar(select(User).where(User.role == "admin", User.is_active.is_(True)).limit(1))
    if user is None:
        raise RuntimeError("No active admin user")
    engine = RetrievalEngine(db)
    for question in QUESTIONS:
        qp = build_query_plan(question)
        ep = build_deterministic_evidence_plan(question, qp)
        profile = "research" if qp.coverage_sensitive else "direct"
        started = time.perf_counter()
        evidence, trace = engine.retrieve(
            question, user, None, profile=profile, query_plan=qp, evidence_plan=ep
        )
        issues = retrieval_quality_issues(question, evidence, trace, qp)
        print(json.dumps({
            "question": question,
            "profile": profile,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "lookup_term": qp.lookup_term,
            "line": qp.line,
            "coverage_kind": qp.coverage_kind,
            "exact_terms": qp.exact_terms,
            "quality_issues": issues,
            "documents": list(dict.fromkeys(item.candidate.document_title for item in evidence))[:8],
            "evidence_preview": [
                {
                    "document": item.candidate.document_title,
                    "page": item.candidate.page_from,
                    "section": item.candidate.section_path[-1] if item.candidate.section_path else None,
                    "score": round(item.candidate.rerank_score, 4),
                    "text": (item.candidate.contextual_text or item.candidate.text)[:420],
                }
                for item in evidence[:5]
            ],
        }, ensure_ascii=False))
