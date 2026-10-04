from types import SimpleNamespace

from ike.agent.controller import AgentDecision, AgenticQAService
from ike.mcp.server import CorpusMCPServer


def test_agent_tool_catalog_is_narrow_and_read_only():
    names = {item["name"] for item in CorpusMCPServer.tool_catalog()}
    assert names == {
        "search_documents",
        "search_chunks",
        "search_many",
        "search_lists",
        "search_sections",
        "get_document_structure",
        "get_section",
        "search_tables",
        "exact_lookup",
    }
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
        decision_summary="The returned procedure directly answers both requested roles.",
        evidence_status="sufficient",
        selected_chunk_ids=["11111111-1111-1111-1111-111111111111"],
    )
    assert decision.action == "finish"
    assert decision.evidence_status == "sufficient"
    assert len(decision.selected_chunk_ids) == 1


def test_agent_prompt_requires_early_stop_and_forbids_requirement_expansion():
    service = object.__new__(AgenticQAService)
    prompt = AgenticQAService._controller_system_prompt(service)
    folded = prompt.casefold()
    assert "stop as soon" in folded
    assert "do not invent additional requirements" in folded
    assert "predefined intent taxonomy" in folded
    assert "arbitrary sql" not in folded
    assert "search_many" in folded
    assert "search_lists" in folded
    assert "one semantic query per part" in folded
    assert "minimal semantic subject" in folded


def test_agent_decision_supports_independent_multi_search():
    decision = AgentDecision(
        action="search_many",
        decision_summary="Search each requested part independently.",
        queries=["first requested set", "second requested set", "third requested set"],
    )
    args = AgenticQAService._decision_arguments(decision, "combined request")
    assert args["queries"] == [
        "first requested set",
        "second requested set",
        "third requested set",
    ]


def test_agent_decision_supports_structured_list_search():
    decision = AgentDecision(
        action="search_lists",
        decision_summary="Search each requested set independently.",
        subjects=["crew controls", "depots", "interchange stations"],
    )
    args = AgenticQAService._decision_arguments(decision, "combined list request")
    assert args["subjects"] == ["crew controls", "depots", "interchange stations"]
