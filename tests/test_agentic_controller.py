from types import SimpleNamespace

from ike.agent.controller import AgentDecision, AgenticQAService
from ike.mcp.server import CorpusMCPServer, ToolExecution


def test_agent_tool_catalog_is_small_generic_and_read_only():
    names = {item["name"] for item in CorpusMCPServer.tool_catalog()}
    assert names == {
        "search",
        "search_documents",
        "get_document_structure",
        "get_section",
    }
    assert "search_many" not in names
    assert "search_lists" not in names
    assert "exact_lookup" not in names
    assert "sql" not in " ".join(names).casefold()


def test_document_title_scoring_is_query_driven_not_domain_hardcoded():
    document = SimpleNamespace(
        title="General Operating Rulebook 2026",
        original_filename="general-rulebook.pdf",
    )
    assert CorpusMCPServer._title_score("operating rulebook", document) > 0
    assert CorpusMCPServer._title_score("completely unrelated phrase", document) == 0


def test_agent_decision_can_finish_with_selected_evidence():
    decision = AgentDecision(
        action="finish",
        decision_summary="The returned evidence directly answers the request.",
        evidence_status="sufficient",
        selected_chunk_ids=["11111111-1111-1111-1111-111111111111"],
    )
    assert decision.action == "finish"
    assert decision.evidence_status == "sufficient"
    assert len(decision.selected_chunk_ids) == 1


def test_agent_prompt_uses_generic_search_and_literal_anchor_preservation():
    service = object.__new__(AgenticQAService)
    prompt = AgenticQAService._controller_system_prompt(service)
    folded = prompt.casefold()
    assert "stop as soon" in folded
    assert "predefined types" in folded
    assert "literal anchors" in folded
    assert "generic search tool" in folded
    assert "search_many" not in folded
    assert "search_lists" not in folded
    assert "one semantic query per part" not in folded


def test_agent_decision_preserves_semantic_query_and_literal_anchors():
    decision = AgentDecision(
        action="search",
        decision_summary="Investigate both requested terms without losing their literal identities.",
        semantic_query="What do the requested terms mean?",
        anchors=["LDS", "LDCE"],
    )
    args = AgenticQAService._decision_arguments(decision, "What are LDS and LDCE?")
    assert args == {
        "semantic_query": "What do the requested terms mean?",
        "anchors": ["LDS", "LDCE"],
        "document_ids": [],
    }


def test_generic_search_executes_semantic_probe_and_each_anchor_independently():
    server = object.__new__(CorpusMCPServer)
    server.settings = SimpleNamespace(agent_search_top_k=8)

    calls = []

    def fake_hybrid(query, *, document_ids=None, top_k=None, tables_only=False, exact_only=False):
        calls.append(query)
        return ToolExecution(
            tool="search_chunks",
            arguments={"query": query},
            items=[{"chunk_id": query, "snippet": query}],
            candidates=[],
            elapsed_ms=1,
        )

    server._hybrid_search = fake_hybrid

    result = CorpusMCPServer.search(
        server,
        "What do the requested terms mean?",
        anchors=["LDS", "LDCE"],
    )

    assert calls == [
        "What do the requested terms mean?",
        "LDS",
        "LDCE",
    ]
    assert [group["kind"] for group in result.query_groups] == [
        "semantic",
        "anchor",
        "anchor",
    ]
    assert [group["query"] for group in result.query_groups] == calls
    assert result.arguments["anchors"] == ["LDS", "LDCE"]
