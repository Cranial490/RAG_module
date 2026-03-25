from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from .base_vector_db import BaseVectorMemory
from ..chunking.data_models import Chunk, ChunkMetadata

load_dotenv()

class QdrantVectorMemory(BaseVectorMemory):
    """
    Implementation of BaseVectorMemory using Qdrant vector database.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "memory_chunks",
        vector_size: int = 1536,  # Default size for many embedding models
        distance: Distance = Distance.COSINE,
        create_collection_if_not_exists: bool = True
    ):
        """
        Initialize the Qdrant vector memory.
        
        Args:
            vector_client: Optional pre-initialized Qdrant client instance
            url: Qdrant URL used when vector_client is not provided
            api_key: Optional Qdrant API key used when vector_client is not provided
            collection_name: Name of the collection to store memory chunks
            vector_size: Dimension of the embedding vectors
            distance: Distance metric to use for similarity search
            create_collection_if_not_exists: Whether to create the collection if it doesn't exist
        """
        resolved_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        if not resolved_url:
            raise ValueError(
                "QDRANT URL is required. Pass it explicitly or set "
                "QDRANT_URL."
            )
        resolved_api_key = api_key or os.getenv("QDRANT_API_KEY")
        try:
            vector_client = QdrantClient(
                url=resolved_url,
                api_key=resolved_api_key,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Qdrant client for URL '{resolved_url}': {exc}"
            ) from exc

        self.vector_client = vector_client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        
        # Create collection if it doesn't exist and flag is set
        if create_collection_if_not_exists:
            self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Create the collection if it doesn't exist.
        """
        try:
            collections = self.vector_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.vector_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create or verify collection: {str(e)}")

    def add_chunks(self, chunks: List[Chunk]):
        """
        Add memory chunks to vector DB.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        points = []
        for chunk in chunks:
            # Convert Chunk to PointStruct
            point = PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "text": chunk.text,
                    "metadata": chunk.metadata.model_dump(),
                    "token_count": chunk.token_count,
                }
            )
            points.append(point)
        
        try:
            self.vector_client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add chunks to Qdrant: {str(e)}")

    def retrieve(
        self,
        embedded_query: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve memory chunks based on semantic similarity + metadata filters.
        
        Args:
            embedded_query: Vector embedding of the query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of Chunk objects sorted by relevance
        """
        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                # Handle nested metadata filters
                if key.startswith("metadata."):
                    # Extract the metadata key (after "metadata.")
                    metadata_key = key[9:]  # len("metadata.") == 9
                    filter_conditions.append(
                        FieldCondition(
                            key=f"metadata.{metadata_key}",
                            match=MatchValue(value=value)
                        )
                    )
                else:
                    # Handle top-level filters
                    filter_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            if filter_conditions:
                qdrant_filter = Filter(
                    must=filter_conditions
                )
        
        try:
            # Search for similar vectors
            search_result = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=embedded_query,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            # Convert search results to Chunk objects
            chunks = []
            for result in search_result:
                # Extract data from payload
                payload = result.payload
                text = payload.get("text", "")
                metadata_dict = payload.get("metadata", {})
                created_at_value = metadata_dict.get("created_at")
                created_at = datetime.now()
                if isinstance(created_at_value, str):
                    try:
                        created_at = datetime.fromisoformat(created_at_value)
                    except ValueError:
                        created_at = datetime.utcnow()
                
                # Create ChunkMetadata
                metadata = ChunkMetadata(
                    document_id=metadata_dict.get("document_id", ""),
                    document_title=metadata_dict.get("document_title"),
                    created_at=created_at,
                    tags=metadata_dict.get("tags"),
                    chunk_version=metadata_dict.get("chunk_version")
                )
                
                # Create Chunk
                chunk = Chunk(
                    chunk_id=str(result.id),
                    text=text,
                    embedding=result.vector or [],
                    metadata=metadata,
                    token_count=payload.get("token_count"),
                )
                chunks.append(chunk)
            
            return chunks
        
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve chunks from Qdrant: {str(e)}")
    
    def delete(self, chunk_id: str):
        """
        Delete a memory chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to delete
        """
        try:
            self.vector_client.delete(
                collection_name=self.collection_name,
                points_selector=[chunk_id],
                wait=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to delete chunk from Qdrant: {str(e)}")
