import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import main
from memory_module.chunking.data_models import Chunk, ChunkMetadata


def test_index_endpoint_success(monkeypatch):
    pipeline = MagicMock()
    pipeline.parser = object()
    pipeline.chunker = object()
    pipeline.embedder = object()
    pipeline.vector_db = object()
    pipeline.indexer.return_value = [
        Chunk(
            chunk_id="chunk-1",
            text="hello",
            embedding=[0.1],
            metadata=ChunkMetadata(document_id="doc-1"),
            token_count=1,
        )
    ]
    monkeypatch.setattr(main, "RAGPipeline", lambda config: pipeline)

    client = TestClient(main.app)
    response = client.post(
        "/index",
        data={"config": json.dumps({"parser_key": "docx"}), "metadata": json.dumps({"tags": ["a"]})},
        files={"file": ("sample.docx", b"binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 1
    assert body["chunk_ids"] == ["chunk-1"]


def test_retrieve_endpoint_success(monkeypatch):
    pipeline = MagicMock()
    pipeline.retrieve.return_value = [
        Chunk(
            chunk_id="chunk-1",
            text="hello",
            embedding=[0.1],
            metadata=ChunkMetadata(document_id="doc-1"),
            token_count=1,
        )
    ]
    monkeypatch.setattr(main, "RAGPipeline", lambda config: pipeline)

    client = TestClient(main.app)
    response = client.post(
        "/retrieve",
        params={"query": "hello", "top_k": 3},
        json={
            "config": {
                "embedder_key": "azure_openai",
                "vector_db_key": "qdrant",
                "retrieval_key": "similarity",
            },
            "filters": {"metadata.tags": "a"},
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["chunk_id"] == "chunk-1"
    pipeline.retrieve.assert_called_once_with(
        query="hello",
        top_k=3,
        filters={"metadata.tags": "a"},
    )


def test_list_strategies_returns_all_components(monkeypatch):
    monkeypatch.setattr(main, "list_parsers", lambda: ["docx"])
    monkeypatch.setattr(main, "list_chunkers", lambda: ["document"])
    monkeypatch.setattr(main, "list_embedders", lambda: ["azure_openai"])
    monkeypatch.setattr(main, "list_vector_dbs", lambda: ["qdrant"])
    main.STRATEGY_LISTERS = {
        "parsers": main.list_parsers,
        "chunkers": main.list_chunkers,
        "embedders": main.list_embedders,
        "vector_dbs": main.list_vector_dbs,
    }

    client = TestClient(main.app)
    response = client.get("/strategies")

    assert response.status_code == 200
    assert response.json() == {
        "parsers": {
            "selection_mode": "single",
            "strategies": ["docx"],
        },
        "chunkers": {
            "selection_mode": "single",
            "strategies": ["document"],
        },
        "embedders": {
            "selection_mode": "single",
            "strategies": ["azure_openai"],
        },
        "vector_dbs": {
            "selection_mode": "single",
            "strategies": ["qdrant"],
        },
    }


def test_list_strategies_for_component(monkeypatch):
    monkeypatch.setattr(main, "list_chunkers", lambda: ["document"])
    main.STRATEGY_LISTERS = {
        "parsers": main.list_parsers,
        "chunkers": main.list_chunkers,
        "embedders": main.list_embedders,
        "vector_dbs": main.list_vector_dbs,
    }

    client = TestClient(main.app)
    response = client.get("/strategies/chunkers")

    assert response.status_code == 200
    assert response.json() == {
        "component": "chunkers",
        "selection_mode": "single",
        "strategies": ["document"],
    }


def test_list_strategies_for_unknown_component():
    client = TestClient(main.app)
    response = client.get("/strategies/unknown")

    assert response.status_code == 404
    assert "Unknown strategy component" in response.json()["detail"]


def test_index_endpoint_rejects_invalid_config_json():
    client = TestClient(main.app)
    response = client.post(
        "/index",
        data={"config": "{bad-json"},
        files={"file": ("sample.docx", b"binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400
    assert "Invalid config JSON" in response.json()["detail"]


def test_index_endpoint_rejects_invalid_metadata_json():
    client = TestClient(main.app)
    response = client.post(
        "/index",
        data={"config": json.dumps({"parser_key": "docx"}), "metadata": "{bad-json"},
        files={"file": ("sample.docx", b"binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400
    assert "Invalid metadata JSON" in response.json()["detail"]


def test_index_endpoint_reports_pipeline_init_failure(monkeypatch):
    monkeypatch.setattr(main, "RAGPipeline", lambda config: (_ for _ in ()).throw(ValueError("bad config")))

    client = TestClient(main.app)
    response = client.post(
        "/index",
        data={"config": json.dumps({"parser_key": "docx"})},
        files={"file": ("sample.docx", b"binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400
    assert "Failed to initialize RAGPipeline" in response.json()["detail"]


def test_index_endpoint_reports_indexing_failure(monkeypatch):
    pipeline = MagicMock()
    pipeline.parser = object()
    pipeline.chunker = object()
    pipeline.embedder = object()
    pipeline.vector_db = object()
    pipeline.indexer.side_effect = ValueError("index failed")
    monkeypatch.setattr(main, "RAGPipeline", lambda config: pipeline)

    client = TestClient(main.app)
    response = client.post(
        "/index",
        data={"config": json.dumps({"parser_key": "docx"})},
        files={"file": ("sample.docx", b"binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400
    assert "Failed to index document" in response.json()["detail"]


def test_retrieve_endpoint_rejects_non_object_config():
    client = TestClient(main.app)
    response = client.post(
        "/retrieve",
        params={"query": "hello"},
        json={"config": ["bad"], "filters": None},
    )

    assert response.status_code == 400
    assert "Invalid config" in response.json()["detail"]


def test_retrieve_endpoint_rejects_non_object_filters():
    client = TestClient(main.app)
    response = client.post(
        "/retrieve",
        params={"query": "hello"},
        json={"config": {}, "filters": ["bad"]},
    )

    assert response.status_code == 400
    assert "Invalid filters" in response.json()["detail"]


def test_retrieve_endpoint_reports_pipeline_init_failure(monkeypatch):
    monkeypatch.setattr(main, "RAGPipeline", lambda config: (_ for _ in ()).throw(ValueError("bad config")))

    client = TestClient(main.app)
    response = client.post(
        "/retrieve",
        params={"query": "hello"},
        json={"config": {"retrieval_key": "similarity"}},
    )

    assert response.status_code == 400
    assert "Failed to initialize RAGPipeline" in response.json()["detail"]


def test_retrieve_endpoint_reports_retrieval_failure(monkeypatch):
    pipeline = MagicMock()
    pipeline.retrieve.side_effect = ValueError("bad query")
    monkeypatch.setattr(main, "RAGPipeline", lambda config: pipeline)

    client = TestClient(main.app)
    response = client.post(
        "/retrieve",
        params={"query": "hello"},
        json={"config": {"retrieval_key": "similarity"}},
    )

    assert response.status_code == 400
    assert "Failed to retrieve chunks" in response.json()["detail"]
