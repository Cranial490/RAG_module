import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memory_module.factory.chunking_factory import list_chunkers
from memory_module.factory.embedder_factory import list_embedders
from memory_module.factory.parser_factory import list_parsers
from memory_module.factory.retrieval_factory import list_retrieval_strategies
from memory_module.factory.vector_db_factory import list_vector_dbs
from memory_module.rag_pipeline import RAGPipeline


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / "dashboard"


class RetrieveRequest(BaseModel):
    config: Any
    filters: Any | None = None


STRATEGY_LISTERS = {
    "parsers": list_parsers,
    "chunkers": list_chunkers,
    "embedders": list_embedders,
    "retrievals": list_retrieval_strategies,
    "vector_dbs": list_vector_dbs,
}

app.mount(
    "/dashboard/assets",
    StaticFiles(directory=DASHBOARD_DIR),
    name="dashboard-assets",
)


@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/strategies")
async def list_strategies():
    return {
        component: {
            "selection_mode": "single",
            "strategies": lister(),
        }
        for component, lister in STRATEGY_LISTERS.items()
    }


@app.get("/strategies/{component}")
async def list_strategies_for_component(component: str):
    lister = STRATEGY_LISTERS.get(component)
    if lister is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown strategy component '{component}'. "
                f"Valid values are: {', '.join(sorted(STRATEGY_LISTERS))}."
            ),
        )

    return {
        "component": component,
        "selection_mode": "single",
        "strategies": lister(),
    }


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
        "chunks_indexed": len(chunks),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
    }


@app.post("/retrieve")
async def retrieve_chunks(
    payload: RetrieveRequest,
    query: str,
    top_k: int = 5,
):
    if not isinstance(payload.config, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid config: top-level JSON value must be an object.",
        )
    if payload.filters is not None and not isinstance(payload.filters, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid filters: top-level JSON value must be an object.",
        )

    try:
        pipeline = RAGPipeline(payload.config)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize RAGPipeline: {exc}",
        ) from exc

    try:
        return pipeline.retrieve(query=query, top_k=top_k, filters=payload.filters)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to retrieve chunks: {exc}",
        ) from exc
