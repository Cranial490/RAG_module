import pytest

from memory_module.retrieval.data_models import RetrievalRequest


def test_retrieval_request_requires_query_text_and_embedding():
    req = RetrievalRequest(query_text="hello", query_embedding=[0.1, 0.2])
    assert req.query_text == "hello"
    assert req.query_embedding == [0.1, 0.2]


def test_retrieval_request_defaults():
    req = RetrievalRequest(query_text="hello", query_embedding=[0.1])
    assert req.top_k == 5
    assert req.filters is None


def test_retrieval_request_accepts_filters_and_top_k():
    req = RetrievalRequest(
        query_text="hello",
        query_embedding=[0.1],
        top_k=10,
        filters={"tag": "x"},
    )
    assert req.top_k == 10
    assert req.filters == {"tag": "x"}


def test_retrieval_request_rejects_missing_required_fields():
    with pytest.raises(Exception):
        RetrievalRequest(query_text="hello")  # type: ignore[call-arg]
