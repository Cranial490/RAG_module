import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from memory_module.rag_pipeline import RAGPipeline


app = FastAPI()


@app.post("/index")
async def index_document(
    config: str = Form(...),
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
):
    try:
        parsed_config = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc

    if not isinstance(parsed_config, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid config: top-level JSON value must be an object.",
        )

    parsed_metadata = None
    if metadata is not None:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}") from exc

        if not isinstance(parsed_metadata, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid metadata: top-level JSON value must be an object.",
            )

    try:
        pipeline = RAGPipeline(parsed_config)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize RAGPipeline: {exc}",
        ) from exc

    try:
        chunks = pipeline.indexer(file, metadata=parsed_metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to index document: {exc}",
        ) from exc

    return {
        "message": "Document indexed successfully",
        "filename": file.filename,
        "parser": pipeline.parser is not None,
        "chunker": pipeline.chunker is not None,
        "embedder": pipeline.embedder is not None,
        "vector_db": pipeline.vector_db is not None,
        "retriever": pipeline.retriever is not None,
        "chunks_indexed": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }
