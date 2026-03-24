"""
Memory Module for Vector Database Integration

This package provides classes for working with vector databases for memory storage and retrieval.
It includes abstract base classes and concrete implementations for specific vector databases.
"""

__version__ = "0.1.0"

# Import and expose main classes
from .rag_pipeline import RAGPipeline
from .retrieval.base_retrieval import BaseRetrievalStrategy
from .vector_db.base_vector_db import BaseVectorMemory
from .vector_db.qdrant_vector_db import QdrantVectorMemory
from .chunking.data_models import Chunk

__all__ = [
    "RAGPipeline",
    "BaseRetrievalStrategy",
    "BaseVectorMemory",
    "QdrantVectorMemory",
    "Chunk",
]
