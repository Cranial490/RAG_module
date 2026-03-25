from docx import Document
from fastapi import UploadFile
from pathlib import Path

from .data_models import DocumentParserResult, FileMetadata, ParsedContent, ParsedSection
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
        full_text_parts: list[str] = []
        sections: list[ParsedSection] = []
        current_title: str | None = None
        current_level: int | None = None
        current_lines: list[str] = []

        for paragraph in document.paragraphs:
            paragraph_text = paragraph.text.strip()
            if not paragraph_text:
                continue

            full_text_parts.append(paragraph_text)
            style_name = getattr(paragraph.style, "name", "") or ""

            if style_name.startswith("Heading"):
                if current_title is not None or current_lines:
                    sections.append(
                        ParsedSection(
                            title=current_title,
                            text="\n".join(current_lines).strip(),
                            level=current_level,
                        )
                    )
                current_title = paragraph_text
                current_lines = []
                current_level = self._heading_level(style_name)
                continue

            current_lines.append(paragraph_text)

        if current_title is not None or current_lines:
            sections.append(
                ParsedSection(
                    title=current_title,
                    text="\n".join(current_lines).strip(),
                    level=current_level,
                )
            )

        text = "\n".join(full_text_parts)
        sections = [section for section in sections if section.text]
        mode = "sections"
        if not sections and text:
            sections = [ParsedSection(title=None, text=text, level=None)]
            mode = "text"
        file_stream.file.seek(0)

        return DocumentParserResult(
            content=ParsedContent(
                mode=mode,
                text=text,
                sections=sections,
            ),
            file_metadata=FileMetadata(
                document_id=document_id,
                document_title=Path(file_stream.filename or "uploaded.docx").stem,
            ),
        )

    def _heading_level(self, style_name: str) -> int | None:
        suffix = style_name.replace("Heading", "").strip()
        return int(suffix) if suffix.isdigit() else None
