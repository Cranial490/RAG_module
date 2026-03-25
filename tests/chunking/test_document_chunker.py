from uuid import NAMESPACE_URL, uuid5

import pytest

from memory_module.chunking.document_chunker import DocumentChunker
from memory_module.parser.data_models import DocumentParserResult


def test_chunker_uses_parsed_document_text(parsed_document):
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0, tokenizer="character")

    chunks = chunker.chunk(parsed_document, {})

    assert len(chunks) >= 1
    assert all(chunk.metadata.document_id == "doc_test" for chunk in chunks)


def test_chunker_includes_tags_and_chunk_version(parsed_document):
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0, tokenizer="character")

    chunks = chunker.chunk(parsed_document, {"tags": ["x", "y"], "ignored": "nope"})

    assert chunks[0].metadata.tags == ["x", "y"]
    assert chunks[0].metadata.chunk_version == "doc_test_chunk_1"


def test_chunker_generates_deterministic_chunk_ids(parsed_document):
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0, tokenizer="character")

    first = chunker.chunk(parsed_document, {"tags": ["a"]})
    second = chunker.chunk(parsed_document, {"tags": ["a"]})

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == str(uuid5(NAMESPACE_URL, "doc_test_chunk_1"))


def test_chunker_raises_when_parser_metadata_missing():
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0, tokenizer="character")
    parsed_document = DocumentParserResult(text="hello", file_metadata=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires parser file metadata"):
        chunker.chunk(parsed_document, {})

