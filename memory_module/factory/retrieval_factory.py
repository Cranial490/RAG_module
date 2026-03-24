from typing import Dict, Type

from ..retrieval.base_retrieval import BaseRetrievalStrategy
from ..retrieval.similarity_retrieval import SimilarityRetrievalStrategy


RETRIEVAL_REGISTRY: Dict[str, Type[BaseRetrievalStrategy]] = {
    "similarity": SimilarityRetrievalStrategy,
}


def register_retrieval_strategy(
    key: str,
    retrieval_cls: Type[BaseRetrievalStrategy],
) -> None:
    RETRIEVAL_REGISTRY[key] = retrieval_cls


def get_retrieval_strategy(key: str, **kwargs) -> BaseRetrievalStrategy:
    retrieval_cls = RETRIEVAL_REGISTRY.get(key)
    if retrieval_cls is None:
        raise ValueError(f"Invalid retrieval key: {key}")
    return retrieval_cls(**kwargs)


def list_retrieval_strategies() -> list[str]:
    return sorted(RETRIEVAL_REGISTRY)


def register_retrieval_backend(
    key: str,
    retrieval_cls: Type[BaseRetrievalStrategy],
) -> None:
    register_retrieval_strategy(key, retrieval_cls)


def get_retrieval_backend(key: str, **kwargs) -> BaseRetrievalStrategy:
    return get_retrieval_strategy(key, **kwargs)


def list_retrieval_backends() -> list[str]:
    return list_retrieval_strategies()
