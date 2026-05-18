import pytest

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.errors import ConfigError, InvalidQuery
from memory_module.parser.data_models import DocumentParserResult, FileMetadata, ParsedContent
from memory_module.rag_pipeline import RAGPipeline
from memory_module.retrieval.data_models import RetrievalRequest, ScoredChunk
from tests.conftest import StubChunker, StubEmbedder, StubParser, StubVectorDB


def test_indexer_calls_components_in_order(upload_docx):
    parsed_document = DocumentParserResult(
        content=ParsedContent(mode="text", text="hello world", sections=[]),
        file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
    )
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            text="hello",
            embedding=[],
            metadata=ChunkMetadata(document_id="doc_1", document_title="Doc 1", chunk_version="doc_1_chunk_1"),
            token_count=5,
        )
    ]

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = StubChunker(chunks)
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.vector_db = StubVectorDB()

    result = pipeline.indexer(upload_docx, metadata={"tags": ["x"]})

    assert pipeline.parser.calls == ["accepts", "convert"]
    assert pipeline.chunker.calls[0][0] == "chunk"
    assert pipeline.chunker.calls[0][2] == {"tags": ["x"]}
    assert pipeline.embedder.calls == ["hello"]
    assert pipeline.vector_db.added_chunks == result
    assert result[0].embedding == [0.1, 0.2]


def test_indexer_flattens_batch_embedding(upload_docx):
    parsed_document = DocumentParserResult(
        content=ParsedContent(mode="text", text="hello world", sections=[]),
        file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
    )
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            text="hello",
            embedding=[],
            metadata=ChunkMetadata(document_id="doc_1", chunk_version="doc_1_chunk_1"),
            token_count=5,
        )
    ]

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = StubChunker(chunks)
    pipeline.embedder = StubEmbedder([[0.1, 0.2]])
    pipeline.vector_db = StubVectorDB()

    result = pipeline.indexer(upload_docx)

    assert result[0].embedding == [0.1, 0.2]


def test_retrieve_returns_scored_chunks(sample_chunk):
    scored = ScoredChunk(chunk=sample_chunk, score=0.9)

    class RetrieveStrategy:
        def __init__(self):
            self.calls = []

        def retrieve(self, request: RetrievalRequest):
            self.calls.append(request)
            return [scored]

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.3, 0.4])
    pipeline.retriever = RetrieveStrategy()

    results = pipeline.retrieve("hello", top_k=3, filters={"tags": "x"})

    assert results == [scored]
    assert isinstance(results[0], ScoredChunk)
    assert results[0].score == 0.9
    assert pipeline.embedder.calls == ["hello"]
    assert len(pipeline.retriever.calls) == 1
    req = pipeline.retriever.calls[0]
    assert req.query_text == "hello"
    assert req.query_embedding == [0.3, 0.4]
    assert req.top_k == 3
    assert req.filters == {"tags": "x"}


def test_retrieve_flattens_batch_embeddings(sample_chunk):
    scored = ScoredChunk(chunk=sample_chunk, score=0.5)

    class RetrieveStrategy:
        def retrieve(self, request: RetrievalRequest):
            self.request = request
            return [scored]

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([[0.3, 0.4]])
    pipeline.retriever = RetrieveStrategy()

    pipeline.retrieve("hello")

    assert pipeline.retriever.request.query_embedding == [0.3, 0.4]


def test_retrieve_requires_embedder_and_retrieval_strategy():
    pipeline = RAGPipeline({})

    with pytest.raises(ConfigError, match="embedder strategy"):
        pipeline.retrieve("hello")

    pipeline.embedder = StubEmbedder([0.1])
    with pytest.raises(ConfigError, match="retrieval strategy"):
        pipeline.retrieve("hello")


def test_retrieve_requires_non_empty_query():
    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.1])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(InvalidQuery):
        pipeline.retrieve("   ")


def test_indexer_raises_when_parser_rejects(upload_docx):
    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(
        DocumentParserResult(
            content=ParsedContent(mode="text", text="ignored", sections=[]),
            file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
        ),
        accepts=False,
    )
    pipeline.chunker = StubChunker([])
    pipeline.embedder = StubEmbedder([0.1])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(ValueError, match="Parser rejected the provided document: Rejected for test"):
        pipeline.indexer(upload_docx)


@pytest.mark.parametrize(
    ("parser", "chunker", "embedder", "vector_db", "message"),
    [
        (None, object(), object(), object(), "parser strategy"),
        (object(), None, object(), object(), "chunker strategy"),
        (object(), object(), None, object(), "embedder strategy"),
        (object(), object(), object(), None, "vector_db strategy"),
    ],
)
def test_indexer_requires_all_components(parser, chunker, embedder, vector_db, message, upload_docx):
    pipeline = RAGPipeline({})
    pipeline.parser = parser
    pipeline.chunker = chunker
    pipeline.embedder = embedder
    pipeline.vector_db = vector_db

    with pytest.raises(ConfigError, match=message):
        pipeline.indexer(upload_docx)
