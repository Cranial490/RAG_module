from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..chunking.data_models import Chunk

class BaseVectorMemory(ABC):
    """
    Abstract base class for memory types backed by a vector database.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add memory chunks to vector DB.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        embedded_query: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve memory chunks based on semantic similarity + metadata filters.
        """
        pass
    
    @abstractmethod
    def delete(self, chunk_id: str):
        pass
