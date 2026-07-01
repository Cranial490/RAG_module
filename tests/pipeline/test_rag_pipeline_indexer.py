import pytest

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.errors import (
    ChunkerFailed,
    ConfigError,
    EmbedderFailed,
    InvalidQuery,
    NoChunksProduced,
    ParserFailed,
    ParserRejected,
    VectorDBFailed,
)
from memory_module.parser.data_models import DocumentParserResult, FileMetadata, ParsedContent
from memory_module.rag_pipeline import RAGPipeline
from memory_module.retrieval.data_models import RetrievalRequest, ScoredChunk
from memory_module.retrieval.similarity_retrieval import SimilarityRetrievalStrategy
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
    pipeline.vector_db = StubVectorDB()

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

    with pytest.raises(ParserRejected):
        pipeline.indexer(upload_docx)


def test_indexer_raises_embedder_failed_and_skips_vector_db(upload_docx):
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

    original = RuntimeError("embedder boom")

    class FailingEmbedder:
        def embed(self, text: str):
            raise original

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = StubChunker(chunks)
    pipeline.embedder = FailingEmbedder()
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(EmbedderFailed) as exc_info:
        pipeline.indexer(upload_docx)

    assert exc_info.value.__cause__ is original
    assert pipeline.vector_db.added_chunks is None


def test_retrieve_returns_empty_list_when_retriever_returns_empty():
    class EmptyRetriever:
        def retrieve(self, request: RetrievalRequest):
            return []

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.retriever = EmptyRetriever()
    pipeline.vector_db = StubVectorDB()

    results = pipeline.retrieve("hello")

    assert results == []


def test_retrieve_raises_embedder_failed():
    original = RuntimeError("embedder boom")

    class FailingEmbedder:
        def embed(self, text: str):
            raise original

    class DummyRetriever:
        def retrieve(self, request):
            return []

    pipeline = RAGPipeline({})
    pipeline.embedder = FailingEmbedder()
    pipeline.retriever = DummyRetriever()
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(EmbedderFailed) as exc_info:
        pipeline.retrieve("hello")

    assert exc_info.value.__cause__ is original


def test_retrieve_raises_config_error_when_vector_db_unset():
    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.retriever = SimilarityRetrievalStrategy(vector_db=None)

    with pytest.raises(ConfigError, match="vector_db strategy"):
        pipeline.retrieve("hello")


def test_retrieve_raises_vector_db_failed():
    original = RuntimeError("vector db down")

    class FailingRetriever:
        def retrieve(self, request):
            raise original

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.retriever = FailingRetriever()
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(VectorDBFailed) as exc_info:
        pipeline.retrieve("hello")

    assert exc_info.value.__cause__ is original


def test_indexer_raises_no_chunks_produced_when_chunker_returns_empty(upload_docx):
    parsed_document = DocumentParserResult(
        content=ParsedContent(mode="text", text="hello world", sections=[]),
        file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
    )

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = StubChunker([])
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(NoChunksProduced):
        pipeline.indexer(upload_docx)

    assert pipeline.embedder.calls == []
    assert pipeline.vector_db.added_chunks is None


def test_indexer_raises_no_chunks_produced_for_empty_parsed_content(upload_docx):
    empty_parsed = DocumentParserResult(
        content=ParsedContent(mode="text", text="", sections=[]),
        file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
    )

    class EmptyContentChunker:
        def __init__(self):
            self.calls = []

        def chunk(self, parsed_document, extra):
            self.calls.append((parsed_document, extra))
            return [] if not parsed_document.content.text else [object()]

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(empty_parsed)
    pipeline.chunker = EmptyContentChunker()
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(NoChunksProduced) as exc_info:
        pipeline.indexer(upload_docx)

    assert type(exc_info.value) is NoChunksProduced
    assert pipeline.embedder.calls == []
    assert pipeline.vector_db.added_chunks is None


def test_indexer_raises_vector_db_failed(upload_docx):
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

    original = RuntimeError("vector db down")

    class FailingVectorDB:
        def add_chunks(self, chunks):
            raise original

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = StubChunker(chunks)
    pipeline.embedder = StubEmbedder([0.1, 0.2])
    pipeline.vector_db = FailingVectorDB()

    with pytest.raises(VectorDBFailed) as exc_info:
        pipeline.indexer(upload_docx)

    assert exc_info.value.__cause__ is original


def test_indexer_raises_parser_failed_when_convert_blows_up(upload_docx):
    original = RuntimeError("zip corruption inside docx")

    class FailingConvertParser:
        last_error = None

        def __init__(self):
            self.calls = []

        def accepts(self, file_stream):
            self.calls.append("accepts")
            return True

        def convert(self, file_stream):
            self.calls.append("convert")
            raise original

    pipeline = RAGPipeline({})
    pipeline.parser = FailingConvertParser()
    pipeline.chunker = StubChunker([])
    pipeline.embedder = StubEmbedder([0.1])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(ParserFailed) as exc_info:
        pipeline.indexer(upload_docx)

    assert exc_info.value.__cause__ is original
    assert not isinstance(exc_info.value, ParserRejected)
    assert pipeline.vector_db.added_chunks is None


def test_indexer_raises_chunker_failed_when_chunk_blows_up(upload_docx):
    parsed_document = DocumentParserResult(
        content=ParsedContent(mode="text", text="hello world", sections=[]),
        file_metadata=FileMetadata(document_id="doc_1", document_title="Doc 1"),
    )

    original = ValueError("Unsupported parsed content mode: weird")

    class FailingChunker:
        def __init__(self):
            self.calls = []

        def chunk(self, parsed_document, extra):
            self.calls.append(("chunk", parsed_document, extra))
            raise original

    pipeline = RAGPipeline({})
    pipeline.parser = StubParser(parsed_document)
    pipeline.chunker = FailingChunker()
    pipeline.embedder = StubEmbedder([0.1])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(ChunkerFailed) as exc_info:
        pipeline.indexer(upload_docx)

    assert exc_info.value.__cause__ is original
    assert pipeline.vector_db.added_chunks is None


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
