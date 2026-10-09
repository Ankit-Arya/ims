from types import SimpleNamespace

from ike.agent.models import ResearchTask
from ike.agent.research import ResearchBundle, ResearchExecutor
from ike.agent.tools import CorpusTools, ToolResult


class _FakeTools:
    def __init__(self):
        self.calls = []

    def search_documents(self, query):
        self.calls.append(("source_lookup", query))
        return ToolResult(
            tool="source_lookup",
            arguments={"query": query},
            items=[
                {"document_id": f"00000000-0000-0000-0000-00000000000{i}"}
                for i in range(1, 6)
            ],
        )

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return ToolResult(
            tool="search",
            arguments={"query": query, **kwargs},
        )

    def enumerate(self, query, **kwargs):
        self.calls.append(("enumerate", query, kwargs))
        return ToolResult(
            tool="enumerate",
            arguments={"query": query, **kwargs},
        )


def _executor():
    executor = ResearchExecutor.__new__(ResearchExecutor)
    executor.tools = _FakeTools()
    executor.cancel_check = None
    executor.bundle = ResearchBundle()
    return executor


def test_inferred_source_query_is_soft_boost_not_hard_scope():
    executor = _executor()
    task = ResearchTask(
        id="T1",
        purpose="Find the requested fact",
        requirement_ids=["R1"],
        kind="search",
        query="specific fact in the corpus",
        source_query="likely governing documents",
        exact_terms=["specific fact"],
        top_k=14,
    )

    executor._execute_task(task, "primary")

    search_call = next(call for call in executor.tools.calls if call[0] == "search")
    kwargs = search_call[2]
    assert kwargs["document_ids"] == []
    assert len(kwargs["boost_document_ids"]) == 5
    assert kwargs["boost_document_ids"][0].endswith("001")
    assert kwargs["boost_document_ids"][-1].endswith("005")


def test_explicit_document_id_remains_hard_scope():
    executor = _executor()
    document_id = "00000000-0000-0000-0000-000000000099"
    task = ResearchTask(
        id="T1",
        purpose="Inspect the explicitly selected source",
        requirement_ids=["R1"],
        kind="search",
        query="specific fact",
        source_query="likely related sources",
        document_id=document_id,
        top_k=14,
    )

    executor._execute_task(task, "primary")

    search_call = next(call for call in executor.tools.calls if call[0] == "search")
    kwargs = search_call[2]
    assert kwargs["document_ids"] == [document_id]
    assert len(kwargs["boost_document_ids"]) == 5


def test_canonical_acronym_inside_filename_gets_source_routing_boost():
    document = SimpleNamespace(
        title="02. ABCD 2020(1).pdf",
        original_filename="02. ABCD 2020(1).pdf",
        family_key=None,
        source_role=None,
    )

    named_score = CorpusTools._title_score(
        "what is rule 31 in ABCD 2020 book",
        document,
    )
    unnamed_score = CorpusTools._title_score(
        "what is rule 31 in the general rule book",
        document,
    )

    assert named_score >= unnamed_score + 18.0


def test_answer_observations_strip_snippet_payloads_but_keep_evidence_identity():
    bundle = ResearchBundle()
    bundle.observations.append(
        {
            "round": "primary",
            "task_id": "T1",
            "tool": "search",
            "arguments": {"query": "fact"},
            "items": [
                {
                    "evidence_id": "E1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "document_title": "Document.pdf",
                    "page_from": 7,
                    "score": 0.9,
                    "snippet": "x" * 5000,
                }
            ],
            "metadata": {"candidate_count": 1},
        }
    )

    observations = bundle.answer_observations()

    item = observations[0]["items"][0]
    assert item["evidence_id"] == "E1"
    assert item["document_title"] == "Document.pdf"
    assert "snippet" not in item



def test_table_section_prior_fuses_exact_terms_across_split_table_chunks():
    document_id = "00000000-0000-0000-0000-000000000010"
    first = SimpleNamespace(
        chunk_id="00000000-0000-0000-0000-000000000101",
        document_id=document_id,
        section_path=["LIST OF ITEMS"],
        ordinal=10,
    )
    second = SimpleNamespace(
        chunk_id="00000000-0000-0000-0000-000000000102",
        document_id=document_id,
        section_path=["LIST OF ITEMS"],
        ordinal=11,
    )
    unrelated = SimpleNamespace(
        chunk_id="00000000-0000-0000-0000-000000000201",
        document_id="00000000-0000-0000-0000-000000000020",
        section_path=["OTHER TABLE"],
        ordinal=1,
    )
    lanes = [
        ("table_exact_term_1", [first, unrelated], 1.0),
        ("table_exact_term_2", [second], 1.0),
    ]

    prior = CorpusTools._table_section_prior_ids(lanes)

    assert first.chunk_id in prior
    assert second.chunk_id in prior
    assert unrelated.chunk_id not in prior



def test_table_dedup_preserves_continuation_rows():
    from uuid import UUID

    from ike.retrieval.evidence_selection import deduplicate_candidates
    from ike.retrieval.types import Candidate

    common = dict(
        document_id=UUID("00000000-0000-0000-0000-000000000010"),
        page_from=1,
        page_to=1,
        section_path=["LIST OF ITEMS"],
        content_kind="table",
        document_title="Directory.pdf",
        filename="Directory.pdf",
        revision=None,
        authority=None,
    )
    first = Candidate(
        chunk_id=UUID("00000000-0000-0000-0000-000000000101"),
        ordinal=1,
        text="Station Name Mobile No Alpha 111111",
        contextual_text="LIST OF ITEMS Station Name Mobile No Alpha 111111",
        **common,
    )
    second = Candidate(
        chunk_id=UUID("00000000-0000-0000-0000-000000000102"),
        ordinal=2,
        text="Station Name Mobile No Beta 222222",
        contextual_text="LIST OF ITEMS Station Name Mobile No Beta 222222",
        **common,
    )

    kept = deduplicate_candidates([first, second], threshold=0.2)

    assert [item.chunk_id for item in kept] == [first.chunk_id, second.chunk_id]
