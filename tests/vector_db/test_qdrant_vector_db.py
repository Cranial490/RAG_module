from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import PointStruct

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.retrieval.data_models import ScoredChunk
from memory_module.vector_db.qdrant_vector_db import QdrantVectorMemory


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    client = MagicMock()
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response

    monkeypatch.setattr(
        "memory_module.vector_db.qdrant_vector_db.QdrantClient",
        lambda **kwargs: client,
    )
    return client


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chunk-1",
            text="hello",
            embedding=[0.1, 0.2],
            metadata=ChunkMetadata(
                document_id="doc-1",
                document_title="Doc 1",
                tags=["a"],
                chunk_version="doc-1_chunk_1",
                created_at=datetime.utcnow(),
            ),
            token_count=5,
        )
    ]


def test_init_creates_collection_if_missing(mock_qdrant_client):
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    mock_qdrant_client.create_collection.assert_called_once()
    assert memory.collection_name == "test_collection"


def test_add_chunks_serializes_current_payload_shape(mock_qdrant_client, sample_chunks):
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    memory.add_chunks(sample_chunks)

    mock_qdrant_client.upsert.assert_called_once()
    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert isinstance(points[0], PointStruct)
    assert points[0].payload["text"] == "hello"
    assert points[0].payload["token_count"] == 5
    assert points[0].payload["metadata"]["document_id"] == "doc-1"


def test_add_chunks_does_not_delete_on_success(mock_qdrant_client, sample_chunks):
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    memory.add_chunks(sample_chunks)

    mock_qdrant_client.delete.assert_not_called()


def test_retrieve_returns_scored_chunks_with_scores(mock_qdrant_client):
    mock_qdrant_client.search.return_value = [
        SimpleNamespace(
            id="chunk-1",
            score=0.92,
            payload={
                "text": "hello",
                "token_count": 5,
                "metadata": {
                    "document_id": "doc-1",
                    "document_title": "Doc 1",
                    "tags": ["a"],
                    "chunk_version": "doc-1_chunk_1",
                    "created_at": datetime.utcnow().isoformat(),
                },
            },
            vector=[0.1, 0.2],
        )
    ]

    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)
    results = memory.retrieve([0.1, 0.2], top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], ScoredChunk)
    assert results[0].score == 0.92
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.metadata.document_id == "doc-1"
    assert results[0].chunk.token_count == 5


def test_retrieve_preserves_score_from_qdrant(mock_qdrant_client):
    mock_qdrant_client.search.return_value = [
        SimpleNamespace(
            id="a",
            score=0.95,
            payload={"text": "a", "token_count": 1, "metadata": {"document_id": "d1"}},
            vector=[],
        ),
        SimpleNamespace(
            id="b",
            score=0.60,
            payload={"text": "b", "token_count": 1, "metadata": {"document_id": "d2"}},
            vector=[],
        ),
    ]

    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)
    results = memory.retrieve([0.1], top_k=2)

    assert results[0].score == 0.95
    assert results[1].score == 0.60


def test_retrieve_builds_qdrant_filters(mock_qdrant_client):
    mock_qdrant_client.search.return_value = []
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    memory.retrieve([0.1, 0.2], top_k=1, filters={"metadata.tags": "a"})

    query_filter = mock_qdrant_client.search.call_args.kwargs["query_filter"]
    assert query_filter.must[0].key == "metadata.tags"
    assert query_filter.must[0].match.value == "a"


def test_delete_calls_qdrant_client(mock_qdrant_client):
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    memory.delete("chunk-1")

    assert mock_qdrant_client.delete.call_args.kwargs["points_selector"] == ["chunk-1"]


def test_add_chunks_wraps_client_errors(mock_qdrant_client, sample_chunks):
    mock_qdrant_client.upsert.side_effect = Exception("boom")
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    with pytest.raises(RuntimeError, match="Failed to add chunks to Qdrant"):
        memory.add_chunks(sample_chunks)


def test_add_chunks_compensates_by_deleting_batch_ids_on_upsert_failure(
    mock_qdrant_client, sample_chunks
):
    mock_qdrant_client.upsert.side_effect = Exception("boom")
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    with pytest.raises(RuntimeError, match="Failed to add chunks to Qdrant"):
        memory.add_chunks(sample_chunks)

    mock_qdrant_client.delete.assert_called_once()
    assert mock_qdrant_client.delete.call_args.kwargs["points_selector"] == ["chunk-1"]


def test_add_chunks_surfaces_original_error_when_compensating_delete_also_fails(
    mock_qdrant_client, sample_chunks
):
    mock_qdrant_client.upsert.side_effect = Exception("upsert boom")
    mock_qdrant_client.delete.side_effect = Exception("delete boom")
    memory = QdrantVectorMemory(collection_name="test_collection", vector_size=2)

    with pytest.raises(RuntimeError, match="upsert boom"):
        memory.add_chunks(sample_chunks)

