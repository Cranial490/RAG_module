import pytest

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.retrieval.similarity_retrieval import SimilarityRetrievalStrategy


class StubVectorDB:
    def __init__(self):
        self.calls = []

    def retrieve(self, embedded_query, top_k=5, filters=None):
        self.calls.append((embedded_query, top_k, filters))
        return [
            Chunk(
                chunk_id="chunk-1",
                text="hello",
                embedding=[0.1],
                metadata=ChunkMetadata(document_id="doc-1"),
                token_count=1,
            )
        ]


def test_similarity_retrieval_uses_self_vector_db():
    vector_db = StubVectorDB()
    strategy = SimilarityRetrievalStrategy(vector_db=vector_db)

    results = strategy.retrieve(
        embedded_query=[0.1, 0.2],
        top_k=3,
        filters={"metadata.tags": "a"},
    )

    assert results[0].chunk_id == "chunk-1"
    assert vector_db.calls == [([0.1, 0.2], 3, {"metadata.tags": "a"})]


def test_similarity_retrieval_requires_vector_db():
    strategy = SimilarityRetrievalStrategy()

    with pytest.raises(RuntimeError, match="requires a vector_db"):
        strategy.retrieve(embedded_query=[0.1, 0.2])
