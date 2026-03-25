import pytest

from memory_module.rag_pipeline import RAGPipeline


def test_rag_pipeline_resolves_all_strategies(monkeypatch):
    parser = object()
    chunker = object()
    embedder = object()
    vector_db = object()
    retriever = object()

    monkeypatch.setattr("memory_module.rag_pipeline.get_parser", lambda key, **kwargs: parser)
    monkeypatch.setattr("memory_module.rag_pipeline.get_chunker", lambda key, **kwargs: chunker)
    monkeypatch.setattr("memory_module.rag_pipeline.get_embedder", lambda key, **kwargs: embedder)
    monkeypatch.setattr("memory_module.rag_pipeline.get_vector_db", lambda key, **kwargs: vector_db)
    monkeypatch.setattr("memory_module.rag_pipeline.get_retrieval_strategy", lambda key, **kwargs: retriever)

    pipeline = RAGPipeline(
        {
            "parser_key": "docx",
            "chunker_key": "document",
            "embedder_key": "azure_openai",
            "vector_db_key": "qdrant",
            "retrieval_key": "similarity",
        }
    )

    assert pipeline.parser is parser
    assert pipeline.chunker is chunker
    assert pipeline.embedder is embedder
    assert pipeline.vector_db is vector_db
    assert pipeline.retriever is retriever


def test_rag_pipeline_leaves_missing_strategies_as_none():
    pipeline = RAGPipeline({})

    assert pipeline.parser is None
    assert pipeline.chunker is None
    assert pipeline.embedder is None
    assert pipeline.vector_db is None
    assert pipeline.retriever is None


def test_rag_pipeline_passes_kwargs_to_factories(monkeypatch):
    captured = {}

    monkeypatch.setattr("memory_module.rag_pipeline.get_parser", lambda key, **kwargs: captured.setdefault("parser", (key, kwargs)) or "parser")
    monkeypatch.setattr("memory_module.rag_pipeline.get_chunker", lambda key, **kwargs: captured.setdefault("chunker", (key, kwargs)) or "chunker")
    monkeypatch.setattr("memory_module.rag_pipeline.get_embedder", lambda key, **kwargs: captured.setdefault("embedder", (key, kwargs)) or "embedder")
    monkeypatch.setattr("memory_module.rag_pipeline.get_vector_db", lambda key, **kwargs: captured.setdefault("vector_db", (key, kwargs)) or "vector_db")
    monkeypatch.setattr("memory_module.rag_pipeline.get_retrieval_strategy", lambda key, **kwargs: captured.setdefault("retrieval", (key, kwargs)) or "retrieval")

    pipeline = RAGPipeline(
        {
            "parser_key": "docx",
            "parser_kwargs": {"strict": True},
            "chunker_key": "document",
            "chunker_kwargs": {"chunk_size": 42},
            "embedder_key": "azure_openai",
            "embedder_kwargs": {"model": "m"},
            "vector_db_key": "qdrant",
            "vector_db_kwargs": {"collection_name": "test"},
            "retrieval_key": "similarity",
            "retrieval_kwargs": {"top_k": 7},
        }
    )

    assert pipeline.parser == ("docx", {"strict": True})
    assert pipeline.chunker == ("document", {"chunk_size": 42})
    assert pipeline.embedder == ("azure_openai", {"model": "m"})
    assert pipeline.vector_db == ("qdrant", {"collection_name": "test"})
    assert pipeline.retriever == ("similarity", {"top_k": 7, "vector_db": ("qdrant", {"collection_name": "test"})})


def test_rag_pipeline_invalid_strategy_key_mentions_subsystem(monkeypatch):
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_parser",
        lambda key, **kwargs: (_ for _ in ()).throw(ValueError("Invalid parser key: invalid")),
    )

    with pytest.raises(ValueError, match="Invalid parser strategy key: invalid"):
        RAGPipeline({"parser_key": "invalid"})


def test_rag_pipeline_constructor_error_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_embedder",
        lambda key, **kwargs: (_ for _ in ()).throw(ValueError("Missing API key")),
    )

    with pytest.raises(ValueError, match="Failed to initialize embedder strategy 'azure_openai': Missing API key"):
        RAGPipeline({"embedder_key": "azure_openai"})


def test_rag_pipeline_requires_dict_config():
    with pytest.raises(TypeError, match="config must be a dict"):
        RAGPipeline("not-a-dict")  # type: ignore[arg-type]


def test_rag_pipeline_requires_dict_kwargs():
    with pytest.raises(TypeError, match="chunker_kwargs must be a dict"):
        RAGPipeline({"chunker_key": "document", "chunker_kwargs": "bad"})


def test_rag_pipeline_requires_single_string_strategy_key():
    with pytest.raises(TypeError, match="Only one parser strategy can be selected at a time"):
        RAGPipeline({"parser_key": ["docx"]})
