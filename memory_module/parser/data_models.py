from typing import Optional, Dict, Any
from pydantic import BaseModel

class FileMetadata(BaseModel):
    doc_id: str
    document_title: str
    content_type: Optional[str] = None
    size: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None

class DocumentParserResult(BaseModel):
    content: Dict
    file_metadata: Optional[FileMetadata] = None
    