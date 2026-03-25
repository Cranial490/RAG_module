from typing import Optional
from pydantic import BaseModel

class FileMetadata(BaseModel):
    """Minimal document metadata required for indexing."""
    document_id: str
    document_title: str


class DocumentParserResult(BaseModel):
    """Output contract for parser strategies."""
    text: str
    file_metadata: FileMetadata
