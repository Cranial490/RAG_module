from abc import ABC, abstractmethod
from typing import List

from ..vector_db.base_vector_db import BaseVectorMemory
from .data_models import RetrievalRequest, ScoredChunk


class BaseRetrievalStrategy(ABC):
    def __init__(self, vector_db: BaseVectorMemory | None = None):
        self.vector_db = vector_db

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> List[ScoredChunk]:
        pass
