from docx import Document
from fastapi import UploadFile
from pathlib import Path
import uuid

from .data_models import DocumentParserResult, FileMetadata
from .document_parser_base import DocumentParserBase
from .file_operations import FileOps

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
        file_stream.file.seek(0)

        document = Document(file_stream.file)
        text = "\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
        file_stream.file.seek(0)

        return DocumentParserResult(
            content= {"text": text},
            file_metadata=FileMetadata(
                doc_id=str(uuid.uuid4()),
                document_title=Path(file_stream.filename or "uploaded.docx").stem,
                content_type=file_stream.content_type,
                size=self._size_for(file_stream),
                additional_info={"filename": file_stream.filename or "uploaded.docx"},
            ),
        )

    def _size_for(self, file_stream: UploadFile) -> int | None:
        current_position = file_stream.file.tell()
        file_stream.file.seek(0, 2)
        size = file_stream.file.tell()
        file_stream.file.seek(current_position)
        return size
