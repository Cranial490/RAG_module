from memory_module.rag_pipeline import RAGPipeline


def test_rag_pipeline_resolves_all_strategies(monkeypatch):
    parser = object()
    chunker = object()
    embedder = object()
    vector_db = object()
    retriever = object()

    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_parser",
        lambda key, **kwargs: parser,
    )
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_chunker",
        lambda key, **kwargs: chunker,
    )
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_embedder",
        lambda key, **kwargs: embedder,
    )
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_vector_db",
        lambda key, **kwargs: vector_db,
    )
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_retrieval_strategy",
        lambda key, **kwargs: retriever,
    )

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

    def parser_factory(key, **kwargs):
        captured["parser"] = (key, kwargs)
        return "parser-instance"

    def chunker_factory(key, **kwargs):
        captured["chunker"] = (key, kwargs)
        return "chunker-instance"

    def embedder_factory(key, **kwargs):
        captured["embedder"] = (key, kwargs)
        return "embedder-instance"

    def vector_db_factory(key, **kwargs):
        captured["vector_db"] = (key, kwargs)
        return "vector-db-instance"

    def retrieval_factory(key, **kwargs):
        captured["retrieval"] = (key, kwargs)
        return "retriever-instance"

    monkeypatch.setattr("memory_module.rag_pipeline.get_parser", parser_factory)
    monkeypatch.setattr("memory_module.rag_pipeline.get_chunker", chunker_factory)
    monkeypatch.setattr("memory_module.rag_pipeline.get_embedder", embedder_factory)
    monkeypatch.setattr("memory_module.rag_pipeline.get_vector_db", vector_db_factory)
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_retrieval_strategy",
        retrieval_factory,
    )

    pipeline = RAGPipeline(
        {
            "parser_key": "docx",
            "parser_kwargs": {"strict": True},
            "chunker_key": "document",
            "chunker_kwargs": {"max_tokens": 256},
            "embedder_key": "azure_openai",
            "embedder_kwargs": {"model": "text-embedding-ada-002"},
            "vector_db_key": "qdrant",
            "vector_db_kwargs": {"collection_name": "test"},
            "retrieval_key": "similarity",
            "retrieval_kwargs": {"top_k": 3},
        }
    )

    assert pipeline.parser == "parser-instance"
    assert pipeline.chunker == "chunker-instance"
    assert pipeline.embedder == "embedder-instance"
    assert pipeline.vector_db == "vector-db-instance"
    assert pipeline.retriever == "retriever-instance"
    assert captured["parser"] == ("docx", {"strict": True})
    assert captured["chunker"] == ("document", {"max_tokens": 256})
    assert captured["embedder"] == ("azure_openai", {"model": "text-embedding-ada-002"})
    assert captured["vector_db"] == ("qdrant", {"collection_name": "test"})
    assert captured["retrieval"] == (
        "similarity",
        {"top_k": 3, "vector_db": "vector-db-instance"},
    )


def test_rag_pipeline_invalid_strategy_key_mentions_subsystem(monkeypatch):
    monkeypatch.setattr(
        "memory_module.rag_pipeline.get_parser",
        lambda key, **kwargs: (_ for _ in ()).throw(ValueError("Invalid parser key")),
    )

    try:
        RAGPipeline({"parser_key": "invalid"})
        assert False, "Expected ValueError for invalid parser key"
    except ValueError as exc:
        assert "Invalid parser strategy key: invalid" in str(exc)


def test_rag_pipeline_requires_dict_config():
    try:
        RAGPipeline("not-a-dict")
        assert False, "Expected TypeError for non-dict config"
    except TypeError as exc:
        assert "config must be a dict" in str(exc)


def test_rag_pipeline_requires_dict_kwargs():
    try:
        RAGPipeline({"chunker_key": "document", "chunker_kwargs": "not-a-dict"})
        assert False, "Expected TypeError for non-dict kwargs"
    except TypeError as exc:
        assert "chunker_kwargs must be a dict" in str(exc)

    try:
        RAGPipeline({"vector_db_key": "qdrant", "vector_db_kwargs": "not-a-dict"})
        assert False, "Expected TypeError for non-dict kwargs"
    except TypeError as exc:
        assert "vector_db_kwargs must be a dict" in str(exc)


def test_rag_pipeline_import_smoke():
    pipeline = RAGPipeline({})

    assert pipeline.config == {}
