from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..chunking.data_models import Chunk
from ..vector_db.base_vector_db import BaseVectorMemory


class BaseRetrievalStrategy(ABC):
    def __init__(self, vector_db: BaseVectorMemory | None = None):
        self.vector_db = vector_db

    @abstractmethod
    def retrieve(
        self,
        embedded_query: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        pass
