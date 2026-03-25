from typing import Any, Dict, List, Optional

from ..chunking.data_models import Chunk
from .base_retrieval import BaseRetrievalStrategy

class SimilarityRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(
        self,
        embedded_query: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        if self.vector_db is None:
            raise RuntimeError("SimilarityRetrievalStrategy requires a vector_db.")
        return self.vector_db.retrieve(
            embedded_query=embedded_query,
            top_k=top_k,
            filters=filters,
        )
