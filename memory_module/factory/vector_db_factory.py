from typing import Dict, Type

from ..vector_db.base_vector_db import BaseVectorMemory
from ..vector_db.qdrant_vector_db import QdrantVectorMemory


VECTOR_DB_REGISTRY: Dict[str, Type[BaseVectorMemory]] = {
    "qdrant": QdrantVectorMemory,
}


def register_vector_db(key: str, vector_db_cls: Type[BaseVectorMemory]) -> None:
    VECTOR_DB_REGISTRY[key] = vector_db_cls


def get_vector_db(key: str, **kwargs) -> BaseVectorMemory:
    vector_db_cls = VECTOR_DB_REGISTRY.get(key)
    if vector_db_cls is None:
        raise ValueError(f"Invalid vector db key: {key}")
    return vector_db_cls(**kwargs)


def list_vector_dbs() -> list[str]:
    return sorted(VECTOR_DB_REGISTRY)
