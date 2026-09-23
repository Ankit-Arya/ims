from ike.services.storage import LocalStorage


def test_safe_name_removes_path_and_unsafe_characters():
    assert LocalStorage.safe_name("../../bad name?.pdf") == "bad_name_.pdf"
