from ike.workflows.routing import extract_lookup_term, is_complex_question, is_lookup_question


def test_procedure_is_complex():
    assert is_complex_question("What is the complete procedure for door isolation?")


def test_short_definition_is_not_complex():
    assert not is_complex_question("Define MPA")


def test_single_uppercase_acronym_is_lookup():
    assert extract_lookup_term("BIC") == "BIC"
    assert is_lookup_question("BIC")


def test_cued_lowercase_lookup_is_normalized():
    assert extract_lookup_term("What is bic?") == "BIC"


def test_ordinary_single_word_is_not_forced_into_acronym_path():
    assert extract_lookup_term("doors") is None
