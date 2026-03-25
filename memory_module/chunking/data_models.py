from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChunkMetadata(BaseModel):
    document_id: str                      # Unique identifier of the source document
    document_title: Optional[str] = None # Optional human-readable title
    tags: Optional[List[str]] = None     # Custom tags like "finance", "HR", etc.
    chunk_version: Optional[str] = None  # Versioning support
    created_at: datetime = Field(default_factory=datetime.utcnow) # When this chunk was created

class Chunk(BaseModel):
    chunk_id: str                         # Unique ID for the chunk
    text: str                             # Raw chunk text
    embedding: List[float] = Field(default_factory=list)  # Dense embedding vector
    metadata: ChunkMetadata               # Metadata associated with the chunk
    token_count: Optional[int] = None     # Token count (for filtering/context window fitting)
