from abc import ABC, abstractmethod
from typing import Dict, List

from .data_models import ChunkMetadata, Chunk
from ..parser.data_models import DocumentParserResult

# --- Abstract Chunking Strategy ---
#---The chunker needs to handle the metadata formation as well---
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, parsed_document: DocumentParserResult, extra: Dict) -> List[Chunk]:
        pass

    def _generate_chunk_metadata(self, metadata: Dict) -> ChunkMetadata:
        return ChunkMetadata(**metadata)
