from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ike.retrieval.table_context import table_query_affinity
from ike.retrieval.types import Candidate

ROOT = Path(__file__).parents[1]


def test_table_affinity_ignores_conversational_function_words():
    candidate = Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=1,
        page_from=1,
        page_to=1,
        section_path=["General Information"],
        content_kind="table",
        text="Crew Controls | 02 | Example A, Example B",
        contextual_text="Crew Controls | 02 | Example A, Example B",
        document_title="Reference.pdf",
        filename="Reference.pdf",
        revision=None,
        authority=None,
    )
    focused = table_query_affinity(
        "List of depots and crew controls",
        candidate,
    )
    noisy = table_query_affinity(
        "what is the and of for",
        candidate,
    )
    assert focused > 0
    assert noisy == 0


def test_copy_answer_and_inline_source_hover_contract():
    js = (
        ROOT / "src" / "ike" / "frontend" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    css = (
        ROOT / "src" / "ike" / "frontend" / "static" / "app.css"
    ).read_text(encoding="utf-8")

    assert "copyTextToClipboard" in js
    assert 'class="copy-answer secondary"' in js
    assert "navigator.clipboard?.writeText" in js
    assert "button.textContent = 'Copied'" in js

    assert 'data-evidence-id="$1"' in js
    assert "bindCitationPreviews" in js
    assert "showSourcePreview" in js
    assert "citation.excerpt" in js
    assert "sourceCitationPreview" in js
    assert ".source-citation-preview" in css
    assert ".inline-citation.source-preview-enabled:hover" in css
