from src.dataset.normalize import clean_text, get_options, parse_correct_answers


def test_clean_text_returns_empty_string_for_none() -> None:
    assert clean_text(None) == ""


def test_clean_text_normalizes_spaces() -> None:
    assert clean_text("hello   world") == "hello world"


def test_clean_text_normalizes_newlines() -> None:
    assert clean_text("hello\n\n\nworld") == "hello\n\nworld"


def test_clean_text_strips_text() -> None:
    assert clean_text("   bonjour   ") == "bonjour"


def test_get_options_returns_only_filled_answers() -> None:
    row = {
        "answer_a": "Option A",
        "answer_b": "Option B",
        "answer_c": "",
        "answer_d": "Option D",
        "answer_e": None,
    }

    assert get_options(row) == [
        ("A", "Option A"),
        ("B", "Option B"),
        ("D", "Option D"),
    ]


def test_parse_correct_answers_from_int() -> None:
    assert parse_correct_answers(0) == ["A"]
    assert parse_correct_answers(1) == ["B"]
    assert parse_correct_answers(2) == ["C"]
    assert parse_correct_answers(4) == ["E"]


def test_parse_correct_answers_from_single_letter() -> None:
    assert parse_correct_answers("A") == ["A"]
    assert parse_correct_answers("c") == ["C"]


def test_parse_correct_answers_from_digit_string() -> None:
    assert parse_correct_answers("0") == ["A"]
    assert parse_correct_answers("3") == ["D"]


def test_parse_correct_answers_from_multiple_letters() -> None:
    assert parse_correct_answers("A,C,E") == ["A", "C", "E"]


def test_parse_correct_answers_from_list() -> None:
    assert parse_correct_answers(["A", "C"]) == ["A", "C"]


def test_parse_correct_answers_returns_empty_list_for_invalid_value() -> None:
    assert parse_correct_answers(None) == []
    assert parse_correct_answers("Z") == []
    assert parse_correct_answers(9) == []