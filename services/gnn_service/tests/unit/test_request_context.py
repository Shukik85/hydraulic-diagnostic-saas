from src.inference.request_context import (
    ensure_request_id,
    generate_request_id,
    get_request_id,
    set_request_id,
)


def test_request_context_set_get() -> None:
    assert get_request_id() is None

    set_request_id("req_123")
    assert get_request_id() == "req_123"

    assert ensure_request_id() == "req_123"


def test_ensure_request_id_generates() -> None:
    set_request_id(None)
    rid = ensure_request_id()
    assert rid is not None
    assert get_request_id() == rid


def test_generate_request_id_unique() -> None:
    rid1 = generate_request_id()
    rid2 = generate_request_id()
    assert rid1 != rid2
    assert isinstance(rid1, str)
