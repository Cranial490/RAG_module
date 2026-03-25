import pytest

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.parser.data_models import DocumentParserResult, FileMetadata, ParsedContent
from memory_module.rag_pipeline import RAGPipeline
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


def test_retrieve_embeds_query_and_delegates_to_retrieval_strategy(sample_chunk):
    class RetrieveStrategy:
        def __init__(self):
            self.calls = []

        def retrieve(self, embedded_query, top_k=5, filters=None):
            self.calls.append((embedded_query, top_k, filters))
            return [sample_chunk]

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.3, 0.4])
    pipeline.retriever = RetrieveStrategy()

    results = pipeline.retrieve("hello", top_k=3, filters={"tags": "x"})

    assert results == [sample_chunk]
    assert pipeline.embedder.calls == ["hello"]
    assert pipeline.retriever.calls == [([0.3, 0.4], 3, {"tags": "x"})]


def test_retrieve_flattens_batch_embeddings(sample_chunk):
    class RetrieveStrategy:
        def retrieve(self, embedded_query, top_k=5, filters=None):
            self.embedded_query = embedded_query
            return [sample_chunk]

    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([[0.3, 0.4]])
    pipeline.retriever = RetrieveStrategy()

    pipeline.retrieve("hello")

    assert pipeline.retriever.embedded_query == [0.3, 0.4]


def test_retrieve_requires_embedder_and_retrieval_strategy():
    pipeline = RAGPipeline({})

    with pytest.raises(RuntimeError, match="embedder strategy"):
        pipeline.retrieve("hello")

    pipeline.embedder = StubEmbedder([0.1])
    with pytest.raises(RuntimeError, match="retrieval strategy"):
        pipeline.retrieve("hello")


def test_retrieve_requires_non_empty_query():
    pipeline = RAGPipeline({})
    pipeline.embedder = StubEmbedder([0.1])
    pipeline.vector_db = StubVectorDB()

    with pytest.raises(ValueError, match="non-empty query string"):
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

    with pytest.raises(RuntimeError, match=message):
        pipeline.indexer(upload_docx)
