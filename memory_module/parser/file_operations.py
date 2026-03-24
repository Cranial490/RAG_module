from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from fastapi import UploadFile

# Files larger than 50MB can cause issues.
# 
class FileOps:
    def __init__(self):
        self.mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }

    def validate(
        self,
        file: UploadFile,
        allowed_extensions: list[str],
        max_file_size_bytes: int = 50 * 1024 * 1024,
    ) -> bool:
        filename = file.filename or ""
        extension = Path(filename).suffix.lower()
        normalized_extensions = {ext.lower() for ext in allowed_extensions}

        if extension not in normalized_extensions:
            raise ValueError(f"File extension '{extension}' is not allowed.")

        expected_mime_type = self.mime_map.get(extension)
        content_type = file.content_type
        if expected_mime_type and content_type and content_type != expected_mime_type:
            raise ValueError(
                f"File MIME type '{content_type}' does not match expected type "
                f"'{expected_mime_type}' for extension '{extension}'."
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > max_file_size_bytes:
            raise ValueError(
                f"Uploaded file exceeds the maximum allowed size of "
                f"{max_file_size_bytes} bytes."
            )

        file_bytes = file.file.read()
        file.file.seek(0)

        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        if extension == ".docx":
            self._validate_docx(file_bytes)
        elif extension in {".txt", ".md", ".csv"}:
            self._validate_text(file_bytes)
        elif extension == ".pdf":
            self._validate_pdf(file_bytes)

        return True

    def compute_file_hash(self, file: UploadFile) -> str:
        file_bytes = file.file.read()
        file.file.seek(0)
        return hashlib.sha256(file_bytes).hexdigest()

    def _validate_docx(self, file_bytes: bytes) -> None:
        try:
            Document(BytesIO(file_bytes))
        except (BadZipFile, KeyError, ValueError) as exc:
            raise ValueError("Uploaded DOCX file is corrupt.") from exc

    def _validate_text(self, file_bytes: bytes) -> None:
        try:
            file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Uploaded text file is corrupt or not UTF-8 encoded.") from exc

    def _validate_pdf(self, file_bytes: bytes) -> None:
        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("Uploaded PDF file is corrupt.")
        if b"%%EOF" not in file_bytes[-2048:]:
            raise ValueError("Uploaded PDF file is corrupt.")
