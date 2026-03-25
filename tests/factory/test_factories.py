import pytest

from memory_module.factory.chunking_factory import get_chunker, list_chunkers, register_chunker
from memory_module.factory.embedder_factory import get_embedder, list_embedders, register_embedder
from memory_module.factory.parser_factory import get_parser, list_parsers, register_parser
from memory_module.factory.vector_db_factory import get_vector_db, list_vector_dbs, register_vector_db


class DummyParser:
    pass


class DummyChunker:
    pass


class DummyEmbedder:
    pass


class DummyVectorDB:
    pass


def test_parser_factory_registers_and_lists():
    register_parser("dummy_parser", DummyParser)
    assert "dummy_parser" in list_parsers()
    assert isinstance(get_parser("dummy_parser"), DummyParser)


def test_chunker_factory_registers_and_lists():
    register_chunker("dummy_chunker", DummyChunker)
    assert "dummy_chunker" in list_chunkers()
    assert isinstance(get_chunker("dummy_chunker"), DummyChunker)


def test_embedder_factory_registers_and_lists():
    register_embedder("dummy_embedder", DummyEmbedder)
    assert "dummy_embedder" in list_embedders()
    assert isinstance(get_embedder("dummy_embedder"), DummyEmbedder)


def test_vector_db_factory_registers_and_lists():
    register_vector_db("dummy_vector_db", DummyVectorDB)
    assert "dummy_vector_db" in list_vector_dbs()
    assert isinstance(get_vector_db("dummy_vector_db"), DummyVectorDB)


@pytest.mark.parametrize(
    ("factory", "key", "message"),
    [
        (get_parser, "missing_parser", "Invalid parser key"),
        (get_chunker, "missing_chunker", "Invalid chunker key"),
        (get_embedder, "missing_embedder", "Invalid embedder key"),
        (get_vector_db, "missing_vector_db", "Invalid vector db key"),
    ],
)
def test_factories_raise_for_invalid_keys(factory, key, message):
    with pytest.raises(ValueError, match=message):
        factory(key)

