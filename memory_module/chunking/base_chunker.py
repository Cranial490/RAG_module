from abc import ABC, abstractmethod
from typing import List, Dict

from .data_models import ChunkMetadata, Chunk
from ..parser.data_models import DocumentParserResult

# --- Abstract Chunking Strategy ---
#---The chunker needs to handle the metadata formation as well---
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, parsed_document: DocumentParserResult, metadata: Dict) -> List[Chunk]:
        """
        Abstract method to chunk a document into smaller units.
        
        :param document_text: Full text of the document
        :param metadata: Dict containing metadata like document_id, title, etc.
        :return: List of Chunk objects
        """
        pass

    def _generate_chunk_metadata(self, metadata: Dict) -> ChunkMetadata:
        return ChunkMetadata(**metadata)
