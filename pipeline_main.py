import json
import mimetypes
import sys
from pathlib import Path

from fastapi import UploadFile
from starlette.datastructures import Headers

from memory_module.rag_pipeline import RAGPipeline

def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python pipeline_main.py '<json-config>' '<document-path>' "
            "['<metadata-json>']",
            file=sys.stderr,
        )
        return 1

    raw_config = sys.argv[1]
    document_path = Path(sys.argv[2]).expanduser()
    raw_metadata = sys.argv[3] if len(sys.argv) == 4 else None

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(config, dict):
        print("Invalid config: top-level JSON value must be an object.", file=sys.stderr)
        return 1

    if not document_path.exists() or not document_path.is_file():
        print(f"Invalid document path: {document_path}", file=sys.stderr)
        return 1

    metadata = None
    if raw_metadata is not None:
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            print(f"Invalid metadata JSON: {exc}", file=sys.stderr)
            return 1
        if not isinstance(metadata, dict):
            print("Invalid metadata: top-level JSON value must be an object.", file=sys.stderr)
            return 1

    try:
        pipeline = RAGPipeline(config)
    except Exception as exc:
        print(f"Failed to initialize RAGPipeline: {exc}", file=sys.stderr)
        return 1

    content_type, _ = mimetypes.guess_type(document_path.name)

    try:
        with document_path.open("rb") as document_file:
            upload = UploadFile(
                file=document_file,
                size=document_path.stat().st_size,
                filename=document_path.name,
                headers=Headers(
                    {"content-type": content_type or "application/octet-stream"}
                ),
            )
            chunks = pipeline.indexer(upload, metadata=metadata)
    except Exception as exc:
        print(f"Failed to index document: {exc}", file=sys.stderr)
        return 1

    print("RAGPipeline initialized and document indexed")
    print(
        json.dumps(
            {
                "parser": pipeline.parser is not None,
                "chunker": pipeline.chunker is not None,
                "embedder": pipeline.embedder is not None,
                "vector_db": pipeline.vector_db is not None,
                "retriever": pipeline.retriever is not None,
                "document_path": str(document_path),
                "chunks_indexed": len(chunks),
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
