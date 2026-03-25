import pytest

from memory_module.chunking.data_models import Chunk, ChunkMetadata


def test_chunk_metadata_builds_with_minimal_schema():
    metadata = ChunkMetadata(
        document_id="doc1",
        document_title="Doc 1",
        tags=["a"],
        chunk_version="doc1_chunk_1",
    )

    assert metadata.document_id == "doc1"
    assert metadata.tags == ["a"]


def test_chunk_requires_current_simplified_fields():
    chunk = Chunk(
        chunk_id="chunk1",
        text="hello",
        embedding=[0.1, 0.2],
        metadata=ChunkMetadata(document_id="doc1"),
        token_count=5,
    )

    assert chunk.chunk_id == "chunk1"
    assert chunk.token_count == 5

