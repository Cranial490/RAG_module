from typing import List

from ..chunking.data_models import Chunk
from .base_retrieval import BaseRetrievalStrategy
from .data_models import RetrievalRequest


class SimilarityRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, request: RetrievalRequest) -> List[Chunk]:
        if self.vector_db is None:
            raise RuntimeError("SimilarityRetrievalStrategy requires a vector_db.")
        return self.vector_db.retrieve(
            embedded_query=request.query_embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
