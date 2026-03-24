from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class ChunkMetadata(BaseModel):
    document_id: str                      # Unique identifier of the source document
    document_title: Optional[str] = None # Optional human-readable title
    section: Optional[str] = None        # Section or heading of the chunk
    source_url: Optional[str] = None     # Where the document came from (if applicable)
    published_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow) # When this chunk was created
    updated_at: Optional[datetime] = None
    tags: Optional[List[str]] = None     # Custom tags like "finance", "HR", etc.
    tenant_id: Optional[str] = None      # For multi-tenant filtering
    permissions: Optional[Dict[str, str]] = None # e.g., {'role': 'admin'} for access control
    chunk_version: Optional[str] = None  # Versioning support

class Chunk(BaseModel):
    chunk_id: str                         # Unique ID for the chunk
    text: str                             # Raw chunk text
    context_summary: Optional[str] = None # Optional summary for retrieval/display
    embedding: List[float]                # Dense embedding vector
    metadata: ChunkMetadata               # Metadata associated with the chunk
    token_count: Optional[int] = None     # Token count (for filtering/context window fitting)
    overlap_with_previous: Optional[int] = None
    source_rank_score: Optional[float] = None   # Reranker score, BM25, or other relevance
    is_deleted: Optional[bool] = False    # For soft-deletion
    is_verified: Optional[bool] = None    # Whether grounding/hallucination check passed
