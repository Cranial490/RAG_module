from abc import ABC, abstractmethod
from fastapi import UploadFile

from .data_models import DocumentParserResult

class DocumentParserBase(ABC):
    @abstractmethod
    def accepts(self, file_stream: UploadFile) -> bool:
        """Return whether this parser can handle the uploaded file."""
        pass

    @abstractmethod
    def convert(self, file_stream: UploadFile) -> DocumentParserResult:
        """Parse the uploaded file into structured content and file metadata."""
        pass
