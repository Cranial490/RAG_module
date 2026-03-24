from abc import ABC, abstractmethod
from fastapi import UploadFile

from .data_models import DocumentParserResult

class DocumentParserBase(ABC):
    @abstractmethod
    def accepts(self, file_stream: UploadFile) -> bool:
        pass

    @abstractmethod
    def convert(self, file_stream: UploadFile) -> DocumentParserResult:
        pass
