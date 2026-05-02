from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query_text: str
    query_embedding: list[float]
    top_k: int = 5
    filters: dict | None = None
