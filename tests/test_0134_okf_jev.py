from types import SimpleNamespace
from uuid import uuid4

import httpx

from ike.retrieval.types import Candidate
from ike.services.jev import JevEvidenceJudge
from ike.services.okf import render_document_concept


def _candidate(index: int, score: float) -> Candidate:
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=index,
        page_from=1,
        page_to=1,
        section_path=["Procedure"],
        content_kind="text",
        text=f"passage {index}",
        contextual_text=f"passage {index}",
        document_title="Manual",
        filename="manual.pdf",
        revision="1",
        authority="Ops",
        final_retrieval_score=score,
        rank_method="hybrid_fusion",
    )
def _settings(mode: str):
    return SimpleNamespace(
        jev_mode=mode,
        typesafe_api_key="test",
        jev_base_url="https://api.typesafe.ai",
        jev_model="jev-latest",
        jev_timeout_seconds=1.0,
        jev_max_candidates=2,
        jev_candidate_chars=1000,
        jev_weight=0.9,
        jev_fail_open=True,
    )


def _answers():
    return {
        "relevance_0": {"type": "noul", "noul": 0.1},
        "applicability_0": {"type": "noul", "noul": 0.1},
        "support_0": {"type": "noul", "noul": 0.1},
        "relevance_1": {"type": "noul", "noul": 0.99},
        "applicability_1": {"type": "noul", "noul": 0.99},
        "support_1": {"type": "noul", "noul": 0.99},
    }


def _client():
    return httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"model": "jev-latest", "answers": _answers()})
    ))
def test_jev_apply_reorders_with_typed_judgment():
    candidates = [_candidate(0, 0.9), _candidate(1, 0.8)]
    client = _client()
    ranked, trace = JevEvidenceJudge(_settings("apply"), client).judge_and_apply("question", candidates)
    assert trace["called"] is True
    assert ranked[0].ordinal == 1
    assert ranked[0].judge_score > ranked[1].judge_score
    client.close()


def test_jev_shadow_does_not_reorder():
    candidates = [_candidate(0, 0.9), _candidate(1, 0.8)]
    client = _client()
    ranked, _ = JevEvidenceJudge(_settings("shadow"), client).judge_and_apply("question", candidates)
    assert [c.ordinal for c in ranked] == [0, 1]
    assert "jev_shadow" in ranked[0].sources
    client.close()


def test_okf_concept_has_required_v02_shape():
    document = SimpleNamespace(
        id=uuid4(), title="Operations Manual", original_filename="ops.pdf", checksum_sha256="a" * 64,
        source_role="operating_procedure", family_key="ops", lifecycle_status="active",
        authority="Operations", revision="5", effective_from=None, effective_to=None, page_count=10,
    )
    chunk = SimpleNamespace(
        ordinal=0, page_from=1, page_to=1, section_path=["Emergency"], content_kind="text",
        text="x", contextual_text="x", metadata={},
    )
    text = render_document_concept(
        document,
        [chunk],
        {"document_type": "operating_manual", "primary_line_codes": ["Line 7"]},
    )
    assert text.startswith("---\n")
    assert 'type: "IMS Document Knowledge"' in text
    assert "sources:" in text
    assert "generated:" in text
    assert 'resource: "ims://documents/' in text
