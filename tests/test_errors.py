import inspect

import pytest

import memory_module.errors as errors_module
from memory_module.errors import ConfigError, InvalidQuery, NoChunksProduced, RAGError


def test_all_subclasses_inherit_rag_error_and_have_code():
    subclasses = [
        cls
        for _, cls in inspect.getmembers(errors_module, inspect.isclass)
        if issubclass(cls, RAGError) and cls is not RAGError
    ]
    assert subclasses, "No RAGError subclasses found in errors.py"
    for cls in subclasses:
        assert issubclass(cls, RAGError), f"{cls.__name__} does not inherit RAGError"
        assert hasattr(cls, "code"), f"{cls.__name__} missing 'code' attribute"
        assert cls.code, f"{cls.__name__}.code must be a non-empty string"


def test_config_error_is_rag_error():
    exc = ConfigError("bad config")
    assert isinstance(exc, RAGError)
    assert ConfigError.code == "config_error"


def test_invalid_query_is_rag_error():
    exc = InvalidQuery("empty query")
    assert isinstance(exc, RAGError)
    assert InvalidQuery.code == "invalid_query"


def test_no_chunks_produced_is_rag_error():
    exc = NoChunksProduced("no chunks")
    assert isinstance(exc, RAGError)
    assert NoChunksProduced.code == "no_chunks_produced"
