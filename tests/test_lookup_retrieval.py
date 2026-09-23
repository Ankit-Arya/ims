from uuid import uuid4

from ike.retrieval.lookup import definition_score
from ike.retrieval.types import Candidate


def candidate(text: str, *, section: list[str] | None = None, kind: str = "text") -> Candidate:
    return Candidate(
        chunk_id=uuid4(),
        document_id=uuid4(),
        ordinal=1,
        page_from=1,
        page_to=1,
        section_path=section or [],
        content_kind=kind,
        text=text,
        contextual_text=text,
        document_title="Test manual",
        filename="test.pdf",
        revision=None,
        authority=None,
    )


def test_definition_patterns_are_protected_over_plain_mentions():
    plain = candidate("The BIC shall be inspected during maintenance.")
    equals = candidate("BIC = Bogie Isolation Cock", section=["Abbreviations"])
    parenthetical = candidate("Brake Isolation Cock (BIC)")

    assert definition_score("BIC", equals) > definition_score("BIC", plain)
    assert definition_score("BIC", parenthetical) > definition_score("BIC", plain)


def test_table_definition_receives_additional_signal():
    text = "BIC = Battery Isolation Contactor"
    normal = candidate(text)
    table = candidate(text, kind="table")
    assert definition_score("BIC", table) > definition_score("BIC", normal)
