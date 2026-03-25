from typing import Any, Literal, Optional
from pydantic import BaseModel


class ParsedSection(BaseModel):
    title: Optional[str] = None
    text: str
    level: Optional[int] = None
    metadata: dict[str, Any] | None = None


class ParsedContent(BaseModel):
    mode: Literal["text", "sections"]
    text: str
    sections: list[ParsedSection]


class FileMetadata(BaseModel):
    """Minimal document metadata required for indexing."""
    document_id: str
    document_title: str


class DocumentParserResult(BaseModel):
    """Output contract for parser strategies."""
    content: ParsedContent
    file_metadata: FileMetadata
