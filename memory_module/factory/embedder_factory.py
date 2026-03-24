from typing import Dict, Type

from ..embedder.azure_open_ai_embedder import AzureEmbeddingGenerator
from ..embedder.base_embedder import BaseEmbedder


EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "azure_openai": AzureEmbeddingGenerator,
}


def register_embedder(key: str, embedder_cls: Type[BaseEmbedder]) -> None:
    EMBEDDER_REGISTRY[key] = embedder_cls


def get_embedder(key: str, **kwargs) -> BaseEmbedder:
    embedder_cls = EMBEDDER_REGISTRY.get(key)
    if embedder_cls is None:
        raise ValueError(f"Invalid embedder key: {key}")
    return embedder_cls(**kwargs)


def list_embedders() -> list[str]:
    return sorted(EMBEDDER_REGISTRY)
