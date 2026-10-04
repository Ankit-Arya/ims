from ike.services.query_debug import debug_report_markdown


def test_debug_report_surfaces_semantic_queries_and_search_results():
    report = {
        "query_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-10-03T10:00:00+00:00",
        "mode": "research",
        "original_question": "Show the requested section from the rulebook",
        "semantic_interpretation": "Navigate the named rulebook and explain the requested section.",
        "answer_shape": "synthesis",
        "source_hints": ["RULEBOOK"],
        "entities": ["requested section", "RULEBOOK"],
        "evidence_goals": [{
            "id": "g1",
            "kind": "overview",
            "question": "Retrieve the requested section.",
            "coverage_contract": "enumerate_set",
            "retrieval_tools": ["section_navigation"],
            "search_queries": ["requested section RULEBOOK"],
        }],
        "inferred_or_rephrased_queries": ["requested section RULEBOOK", "RULEBOOK requested section"],
        "executed_search_queries": ["requested section RULEBOOK"],
        "recovery_search_queries": ["RULEBOOK section heading"],
        "search_results": [{
            "debug_rank": 1,
            "_trace_lane": "reranked_candidate_preview",
            "document_title": "Rulebook.pdf",
            "page_from": 12,
            "section_path": ["PART VII"],
            "rerank_score": 0.91,
            "sources": ["section"],
            "snippet": "PART VII ...",
        }],
        "goal_satisfaction": {"complete": True},
        "recovery_summary": {},
        "okf_resolution": {},
        "corpus_discovery": {},
        "final_answer": "Supported answer [E1].",
        "citations": [],
        "notes": [],
    }

    markdown = debug_report_markdown(report)

    assert "AI interpretation" in markdown
    assert "Inferred / rephrased search questions" in markdown
    assert "Actually executed search queries" in markdown
    assert "Recovery search queries" in markdown
    assert "Rulebook.pdf" in markdown
    assert "PART VII" in markdown


def test_debug_report_renders_agent_tool_loop():
    report = {
        "query_id": "00000000-0000-0000-0000-000000000002",
        "created_at": "2026-10-04T10:00:00+00:00",
        "mode": "direct",
        "original_question": "What should role A do during event B?",
        "semantic_interpretation": "Search the documented responsibilities for role A during event B.",
        "answer_shape": None,
        "source_hints": [],
        "entities": [],
        "evidence_goals": [],
        "agentic": True,
        "agent_architecture": "bounded_mcp_tool_loop_v1",
        "agent_evidence_status": "sufficient",
        "agent_stop_reason": "The procedure directly answers the request.",
        "agent_unresolved": [],
        "agent_decisions": [
            {
                "action": "search_chunks",
                "decision_summary": "Search directly for the requested role and event.",
            },
            {
                "action": "finish",
                "decision_summary": "The procedure directly answers the request.",
            },
        ],
        "tool_calls": [
            {
                "tool": "search_chunks",
                "arguments": {"query": "role A event B procedure"},
                "elapsed_ms": 120,
            }
        ],
        "inferred_or_rephrased_queries": ["role A event B procedure"],
        "executed_search_queries": ["role A event B procedure"],
        "recovery_search_queries": [],
        "per_query_search_results": [],
        "search_results": [],
        "goal_satisfaction": None,
        "recovery_summary": {},
        "okf_resolution": {},
        "corpus_discovery": {},
        "final_answer": "Role A should follow the documented procedure [E1].",
        "citations": [],
        "notes": [],
    }

    markdown = debug_report_markdown(report)

    assert "Agent research loop" in markdown
    assert "bounded_mcp_tool_loop_v1" in markdown
    assert "search_chunks" in markdown
    assert "The procedure directly answers the request." in markdown
