from ike.services.query_debug import debug_report_markdown


def test_debug_report_renders_v5_plan_research_and_selected_evidence():
    report = {
        "debug_version": 5,
        "query_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-10-06T10:00:00+00:00",
        "mode": "research",
        "original_question": "How many chapters are there in RULEBOOK?",
        "agentic": True,
        "agent_architecture": "planned_ai_research_v5",
        "planner_model": "strong-model",
        "planner_reasoning": "medium",
        "answer_model": "strong-model",
        "answer_reasoning": "medium",
        "agent_evidence_status": "sufficient",
        "agent_stop_reason": "answered_after_targeted_gap",
        "research_plan": {
            "interpretation": "Count and list the chapters of the named rulebook.",
            "answer_requirements": ["chapter count", "chapter list"],
            "tasks": [
                {
                    "id": "T1",
                    "purpose": "Resolve the named rulebook.",
                    "kind": "source_lookup",
                    "source_query": "RULEBOOK",
                    "depends_on": [],
                }
            ],
        },
        "tool_calls": [
            {
                "round": "primary",
                "task_id": "T1",
                "purpose": "Resolve the named rulebook.",
                "depends_on": [],
                "tool": "source_lookup",
                "arguments": {"query": "RULEBOOK"},
                "metadata": {"candidate_count": 1},
                "elapsed_ms": 20,
                "note": "Document routing candidates.",
                "items": [
                    {
                        "document_id": "00000000-0000-0000-0000-000000000010",
                        "title": "Rulebook.pdf",
                    }
                ],
            },
            {
                "round": "gap",
                "task_id": "G1",
                "purpose": "Inspect chapter hierarchy.",
                "depends_on": [],
                "tool": "structure",
                "arguments": {
                    "document_id": "00000000-0000-0000-0000-000000000010",
                    "query": "chapter",
                },
                "metadata": {
                    "matched_node_count": 13,
                    "returned_node_count": 13,
                    "truncated": False,
                },
                "elapsed_ms": 40,
                "note": "Raw indexed hierarchy; the AI interprets the structure.",
                "items": [
                    {
                        "evidence_id": "E1",
                        "label": "CHAPTER I",
                        "section_path": ["CHAPTER I"],
                        "page_from": 1,
                    }
                ],
            },
        ],
        "first_answer_decision": {
            "status": "needs_evidence",
            "missing_requirements": ["chapter hierarchy"],
            "gap_tasks": [
                {
                    "id": "G1",
                    "kind": "structure",
                    "purpose": "Inspect chapter hierarchy.",
                    "document_id": "00000000-0000-0000-0000-000000000010",
                    "query": "chapter",
                }
            ],
        },
        "final_answer_decision": {
            "answer": "There are 13 chapters. [E1]",
            "selected_evidence_ids": ["E1"],
            "confidence": "high",
        },
        "completed_task_ids": ["T1", "G1"],
        "skipped_task_ids": [],
        "selected_evidence": [
            {
                "rank": 1,
                "document_title": "Rulebook.pdf",
                "page_from": 1,
                "section_path": ["CHAPTER I"],
                "chunk_id": "00000000-0000-0000-0000-000000000020",
                "snippet": "Chapter I source text",
            }
        ],
        "workflow_timings_ms": {
            "planning": 1500,
            "primary_research": 20,
            "evidence_reasoning": 1200,
            "gap_research": 40,
            "final_reasoning": 1000,
            "agent_total": 3760,
        },
        "final_answer": "There are 13 chapters. [E1]",
        "citations": [{"evidence_id": "E1"}],
        "notes": [],
    }

    markdown = debug_report_markdown(report)

    assert "planned_ai_research_v5" in markdown
    assert "Query Intelligence plan" in markdown
    assert "Resolve the named rulebook" in markdown
    assert "targeted gap research" in markdown
    assert "Rulebook.pdf" in markdown
    assert "CHAPTER I" in markdown
    assert "AI-selected evidence" in markdown
    assert "There are 13 chapters" in markdown


def test_debug_report_renders_primary_answer_without_gap_round():
    report = {
        "debug_version": 5,
        "query_id": "00000000-0000-0000-0000-000000000002",
        "created_at": "2026-10-06T10:00:00+00:00",
        "mode": "direct",
        "original_question": "What is SS/UI?",
        "agent_architecture": "planned_ai_research_v5",
        "planner_model": "strong-model",
        "planner_reasoning": "medium",
        "answer_model": "strong-model",
        "answer_reasoning": "medium",
        "agent_evidence_status": "sufficient",
        "agent_stop_reason": "answered_from_primary_research",
        "research_plan": {
            "interpretation": "Define SS/UI.",
            "answer_requirements": ["definition"],
            "tasks": [],
        },
        "tool_calls": [],
        "first_answer_decision": {
            "status": "answer",
            "answer": "Definition. [E1]",
            "selected_evidence_ids": ["E1"],
            "confidence": "high",
        },
        "final_answer_decision": {},
        "completed_task_ids": [],
        "skipped_task_ids": [],
        "selected_evidence": [],
        "workflow_timings_ms": {},
        "final_answer": "Definition. [E1]",
        "citations": [],
        "notes": [],
    }

    markdown = debug_report_markdown(report)

    assert "answered_from_primary_research" in markdown
    assert "First evidence/answer decision" in markdown
    assert "Final decision after targeted gap research" not in markdown
