import json

import pipeline_main
from memory_module.chunking.data_models import Chunk, ChunkMetadata


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.parser = object()
        self.chunker = object()
        self.embedder = object()
        self.vector_db = object()

    def indexer(self, upload, metadata=None):
        return [
            Chunk(
                chunk_id="chunk-1",
                text="hello",
                embedding=[0.1],
                metadata=ChunkMetadata(document_id="doc-1"),
                token_count=1,
            )
        ]


def test_pipeline_main_success(monkeypatch, tmp_path, capsys):
    document_path = tmp_path / "sample.docx"
    document_path.write_bytes(b"docx-bytes")
    monkeypatch.setattr(pipeline_main, "RAGPipeline", FakePipeline)
    monkeypatch.setattr(
        pipeline_main.sys,
        "argv",
        ["pipeline_main.py", json.dumps({"parser_key": "docx"}), str(document_path), json.dumps({"tags": ["a"]})],
    )

    exit_code = pipeline_main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RAGPipeline initialized and document indexed" in output
    assert '"chunks_indexed": 1' in output


def test_pipeline_main_rejects_invalid_config_json(monkeypatch, capsys):
    monkeypatch.setattr(pipeline_main.sys, "argv", ["pipeline_main.py", "{bad-json", "missing.docx"])

    exit_code = pipeline_main.main()

    assert exit_code == 1
    assert "Invalid JSON" in capsys.readouterr().err


def test_pipeline_main_rejects_invalid_metadata_json(monkeypatch, tmp_path, capsys):
    document_path = tmp_path / "sample.docx"
    document_path.write_bytes(b"docx-bytes")
    monkeypatch.setattr(pipeline_main.sys, "argv", ["pipeline_main.py", "{}", str(document_path), "{bad-json"])

    exit_code = pipeline_main.main()

    assert exit_code == 1
    assert "Invalid metadata JSON" in capsys.readouterr().err


def test_pipeline_main_rejects_invalid_file_path(monkeypatch, capsys):
    monkeypatch.setattr(pipeline_main.sys, "argv", ["pipeline_main.py", "{}", "missing.docx"])

    exit_code = pipeline_main.main()

    assert exit_code == 1
    assert "Invalid document path" in capsys.readouterr().err


def test_pipeline_main_reports_pipeline_init_failure(monkeypatch, tmp_path, capsys):
    document_path = tmp_path / "sample.docx"
    document_path.write_bytes(b"docx-bytes")
    monkeypatch.setattr(
        pipeline_main,
        "RAGPipeline",
        lambda config: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(pipeline_main.sys, "argv", ["pipeline_main.py", "{}", str(document_path)])

    exit_code = pipeline_main.main()

    assert exit_code == 1
    assert "Failed to initialize RAGPipeline" in capsys.readouterr().err


def test_pipeline_main_reports_index_failure(monkeypatch, tmp_path, capsys):
    class BrokenPipeline(FakePipeline):
        def indexer(self, upload, metadata=None):
            raise ValueError("index failed")

    document_path = tmp_path / "sample.docx"
    document_path.write_bytes(b"docx-bytes")
    monkeypatch.setattr(pipeline_main, "RAGPipeline", BrokenPipeline)
    monkeypatch.setattr(pipeline_main.sys, "argv", ["pipeline_main.py", "{}", str(document_path)])

    exit_code = pipeline_main.main()

    assert exit_code == 1
    assert "Failed to index document" in capsys.readouterr().err
