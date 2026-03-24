import pytest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from qdrant_client.models import PointStruct, ScoredPoint

from .data_models import Chunk, ChunkMetadata
from .QdrantVectorMemory import QdrantVectorMemory

@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = MagicMock()
    
    # Mock get_collections
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response
    
    # Mock search
    search_result = [
        ScoredPoint(
            id="test-id-1",
            version=1,
            score=0.9,
            payload={
                "text": "Test content 1",
                "metadata": {
                    "document_id": "doc1",
                    "document_title": "Test Document 1",
                    "tags": ["test", "AI"]
                },
                "created_at": datetime.now().isoformat()
            },
            vector=[0.1, 0.2, 0.3, 0.4]
        ),
        ScoredPoint(
            id="test-id-2",
            version=1,
            score=0.8,
            payload={
                "text": "Test content 2",
                "metadata": {
                    "document_id": "doc2",
                    "document_title": "Test Document 2",
                    "tags": ["test", "databases"]
                },
                "created_at": datetime.now().isoformat()
            },
            vector=[0.2, 0.3, 0.4, 0.5]
        )
    ]
    client.search.return_value = search_result
    
    return client

@pytest.fixture
def memory_chunks():
    """Create sample memory chunks for testing."""
    return [
        Chunk(
            chunk_id="test-id-1",
            text="Test content 1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata=ChunkMetadata(
                document_id="doc1",
                document_title="Test Document 1",
                tags=["test", "AI"],
                created_at=datetime.now()
            ),
            token_count=3
        ),
        Chunk(
            chunk_id="test-id-2",
            text="Test content 2",
            embedding=[0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(
                document_id="doc2",
                document_title="Test Document 2",
                tags=["test", "databases"],
                created_at=datetime.now()
            ),
            token_count=3
        )
    ]

def test_init_creates_collection_if_not_exists(mock_qdrant_client):
    """Test that the collection is created if it doesn't exist."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    mock_qdrant_client.create_collection.assert_called_once()
    args, kwargs = mock_qdrant_client.create_collection.call_args
    assert kwargs["collection_name"] == "test_collection"

def test_add_chunks(mock_qdrant_client, memory_chunks):
    """Test adding chunks to the vector database."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    memory.add_chunks(memory_chunks)
    
    mock_qdrant_client.upsert.assert_called_once()
    args, kwargs = mock_qdrant_client.upsert.call_args
    
    assert kwargs["collection_name"] == "test_collection"
    assert len(kwargs["points"]) == 2
    
    # Check that the points are correctly converted to PointStruct
    points = kwargs["points"]
    assert isinstance(points[0], PointStruct)
    assert points[0].id == "test-id-1"
    assert points[0].vector == [0.1, 0.2, 0.3, 0.4]
    assert points[0].payload["text"] == "Test content 1"
    assert "metadata" in points[0].payload
    assert "created_at" in points[0].payload

def test_retrieve(mock_qdrant_client):
    """Test retrieving chunks from the vector database."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    query_vector = [0.1, 0.2, 0.3, 0.4]
    results = memory.retrieve(embedded_query=query_vector, top_k=2)
    
    mock_qdrant_client.search.assert_called_once()
    args, kwargs = mock_qdrant_client.search.call_args
    
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["query_vector"] == query_vector
    assert kwargs["limit"] == 2
    assert kwargs["query_filter"] is None
    
    # Check that the results are correctly converted to Chunk
    assert len(results) == 2
    assert isinstance(results[0], Chunk)
    assert results[0].chunk_id == "test-id-1"
    assert results[0].text == "Test content 1"
    assert isinstance(results[0].metadata, ChunkMetadata)

def test_retrieve_with_filters(mock_qdrant_client):
    """Test retrieving chunks with filters."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    query_vector = [0.1, 0.2, 0.3, 0.4]
    filters = {"metadata.tags": "AI"}
    
    results = memory.retrieve(embedded_query=query_vector, top_k=2, filters=filters)
    
    mock_qdrant_client.search.assert_called_once()
    args, kwargs = mock_qdrant_client.search.call_args
    
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["query_vector"] == query_vector
    assert kwargs["limit"] == 2
    assert kwargs["query_filter"] is not None
    
    # Check that the filter is correctly constructed
    filter_conditions = kwargs["query_filter"].must
    assert len(filter_conditions) == 1
    assert filter_conditions[0].key == "metadata.tags"
    assert filter_conditions[0].match.value == "AI"

def test_delete(mock_qdrant_client):
    """Test deleting a chunk from the vector database."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    chunk_id = "test-id-1"
    memory.delete(chunk_id)
    
    mock_qdrant_client.delete.assert_called_once()
    args, kwargs = mock_qdrant_client.delete.call_args
    
    assert kwargs["collection_name"] == "test_collection"
    assert kwargs["points_selector"] == [chunk_id]
    assert kwargs["wait"] is True

def test_error_handling(mock_qdrant_client):
    """Test error handling in the QdrantVectorMemory class."""
    memory = QdrantVectorMemory(
        vector_client=mock_qdrant_client,
        collection_name="test_collection",
        vector_size=4
    )
    
    # Mock an exception in the Qdrant client
    mock_qdrant_client.upsert.side_effect = Exception("Test error")
    
    # Test that the exception is caught and wrapped in a RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        memory.add_chunks([
            Chunk(
                chunk_id="test-id",
                text="Test content",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata=ChunkMetadata(
                    document_id="doc-test",
                    created_at=datetime.now()
                ),
                token_count=2
            )
        ])
    
    assert "Failed to add chunks to Qdrant" in str(excinfo.value)
