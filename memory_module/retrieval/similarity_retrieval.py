from typing import List

from .base_retrieval import BaseRetrievalStrategy
from .data_models import RetrievalRequest, ScoredChunk


class SimilarityRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, request: RetrievalRequest) -> List[ScoredChunk]:
        if self.vector_db is None:
            raise RuntimeError("SimilarityRetrievalStrategy requires a vector_db.")
        return self.vector_db.retrieve(
            embedded_query=request.query_embedding,
            top_k=request.top_k,
            filters=request.filters,
        )
