"""
Integration test for QdrantVectorMemory against a real Qdrant instance.

Requires a running Qdrant (e.g. via docker-compose). When Qdrant is not
reachable the test skips instead of failing, so it is safe to run in CI or
locally without the service up.

Run with:
    pytest memory_module/integration_test.py
"""

import uuid
from datetime import datetime

import pytest

from .chunking.data_models import Chunk, ChunkMetadata
from .retrieval.data_models import ScoredChunk
from .vector_db.qdrant_vector_db import QdrantVectorMemory


def _make_chunk(text: str, embedding: list[float], topic: str) -> Chunk:
    return Chunk(
        chunk_id=f"test-{uuid.uuid4().hex[:8]}",
        text=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            document_id=f"doc-{uuid.uuid4().hex[:8]}",
            tags=[topic],
            created_at=datetime.now(),
        ),
        token_count=len(text.split()),
    )


@pytest.fixture
def qdrant_memory():
    collection_name = f"test_memory_{uuid.uuid4().hex[:8]}"
    try:
        memory = QdrantVectorMemory(
            collection_name=collection_name,
            vector_size=4,
            create_collection_if_not_exists=True,
        )
    except Exception as exc:  # Qdrant not running / unreachable
        pytest.skip(f"Qdrant not available: {exc}")

    try:
        yield memory
    finally:
        try:
            memory.vector_client.delete_collection(collection_name=collection_name)
        except Exception:
            pass


def test_add_retrieve_and_delete_roundtrip(qdrant_memory):
    chunks = [
        _make_chunk(
            "Qdrant is a vector database for similarity search.",
            [0.1, 0.2, 0.3, 0.4],
            "vector_db",
        ),
        _make_chunk(
            "Vector databases power semantic search applications.",
            [0.2, 0.3, 0.4, 0.5],
            "vector_db",
        ),
        _make_chunk(
            "Python is a popular language for data science.",
            [0.3, 0.4, 0.5, 0.6],
            "programming",
        ),
    ]

    qdrant_memory.add_chunks(chunks)

    query_vector = [0.2, 0.3, 0.4, 0.5]
    results = qdrant_memory.retrieve(embedded_query=query_vector, top_k=2)

    assert results, "expected at least one result"
    assert all(isinstance(r, ScoredChunk) for r in results)
    assert all(isinstance(r.score, float) for r in results)

    filtered = qdrant_memory.retrieve(
        embedded_query=query_vector,
        top_k=5,
        filters={"metadata.tags": "programming"},
    )
    assert all("programming" in (r.chunk.metadata.tags or []) for r in filtered)

    chunk_to_delete = chunks[0].chunk_id
    qdrant_memory.delete(chunk_to_delete)

    remaining = qdrant_memory.retrieve(embedded_query=query_vector, top_k=10)
    remaining_ids = [r.chunk.chunk_id for r in remaining]
    assert chunk_to_delete not in remaining_ids
