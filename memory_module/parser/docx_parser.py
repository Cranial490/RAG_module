from docx import Document
from fastapi import UploadFile
from pathlib import Path

from .data_models import DocumentParserResult, FileMetadata
from .document_parser_base import DocumentParserBase
from ..utils.file_operations import FileOps

class DocxParser(DocumentParserBase):
    def __init__(self):
        self.file_ops = FileOps()
        self.last_error: str | None = None

    def accepts(self, file_stream: UploadFile) -> bool:
        try:
            self.file_ops.validate(file_stream, allowed_extensions=[".docx"])
            self.last_error = None
            return True
        except ValueError as exc:
            self.last_error = str(exc)
            return False

    def convert(self, file_stream: UploadFile) -> DocumentParserResult:
        document_id = f"doc_{self.file_ops.compute_file_hash(file_stream)}"
        file_stream.file.seek(0)

        document = Document(file_stream.file)
        text = "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
        file_stream.file.seek(0)

        return DocumentParserResult(
            text=text,
            file_metadata=FileMetadata(
                document_id=document_id,
                document_title=Path(file_stream.filename or "uploaded.docx").stem,
            ),
        )
