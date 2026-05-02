from pydantic import BaseModel

from ..chunking.data_models import Chunk


class RetrievalRequest(BaseModel):
    query_text: str
    query_embedding: list[float]
    top_k: int = 5
    filters: dict | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
