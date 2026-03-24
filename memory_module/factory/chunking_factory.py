from typing import Dict, Type

from ..chunking.base_chunker import BaseChunker
from ..chunking.document_chunker import DocumentChunker


CHUNKING_REGISTRY: Dict[str, Type[BaseChunker]] = {
    "document": DocumentChunker,
}


def register_chunker(key: str, chunker_cls: Type[BaseChunker]) -> None:
    CHUNKING_REGISTRY[key] = chunker_cls


def get_chunker(key: str, **kwargs) -> BaseChunker:
    chunker_cls = CHUNKING_REGISTRY.get(key)
    if chunker_cls is None:
        raise ValueError(f"Invalid chunker key: {key}")
    return chunker_cls(**kwargs)


def list_chunkers() -> list[str]:
    return sorted(CHUNKING_REGISTRY)
