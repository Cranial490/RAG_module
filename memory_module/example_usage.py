from datetime import datetime
import uuid
import random
from qdrant_client import QdrantClient

from .data_models import Chunk, ChunkMetadata
from .QdrantVectorMemory import QdrantVectorMemory

def generate_random_embedding(size=1536):
    """Generate a random embedding vector for demonstration purposes."""
    return [random.random() for _ in range(size)]

def main():
    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    # Create a QdrantVectorMemory instance
    # Using a smaller vector size for the example
    memory = QdrantVectorMemory(
        vector_client=client,
        collection_name="example_memory",
        vector_size=4,  # Small size for demonstration
        create_collection_if_not_exists=True
    )
    
    # Create some sample memory chunks
    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text="This is a sample memory about artificial intelligence.",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata=ChunkMetadata(
                document_id="doc1",
                document_title="AI Article",
                tags=["article", "AI"],
                created_at=datetime.now()
            ),
            token_count=12
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text="Vector databases are used for semantic search applications.",
            embedding=[0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(
                document_id="doc2",
                document_title="Database Guide",
                tags=["book", "databases"],
                created_at=datetime.now()
            ),
            token_count=11
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            text="Python is a popular programming language for data science.",
            embedding=[0.3, 0.4, 0.5, 0.6],
            metadata=ChunkMetadata(
                document_id="doc3",
                document_title="Python Tutorial",
                tags=["tutorial", "programming"],
                created_at=datetime.now()
            ),
            token_count=12
        )
    ]
    
    # Add chunks to the vector database
    print("Adding memory chunks...")
    memory.add_chunks(chunks)
    
    # Retrieve chunks based on a query vector
    print("\nRetrieving similar chunks...")
    query_vector = [0.2, 0.3, 0.4, 0.5]  # Similar to the second chunk
    results = memory.retrieve(embedded_query=query_vector, top_k=2)
    
    for i, chunk in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {chunk.chunk_id}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Created At: {chunk.metadata.created_at}")
    
    # Retrieve chunks with filter
    print("\nRetrieving chunks with filter...")
    filtered_results = memory.retrieve(
        embedded_query=query_vector,
        top_k=2,
        filters={"metadata.tags": "programming"}
    )
    
    for i, chunk in enumerate(filtered_results):
        print(f"\nFiltered Result {i+1}:")
        print(f"ID: {chunk.chunk_id}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
    
    # Delete a chunk
    if chunks:
        chunk_to_delete = chunks[0].chunk_id
        print(f"\nDeleting chunk with ID: {chunk_to_delete}")
        memory.delete(chunk_to_delete)
        
        # Verify deletion
        all_results = memory.retrieve(embedded_query=query_vector, top_k=10)
        remaining_ids = [chunk.chunk_id for chunk in all_results]
        
        if chunk_to_delete not in remaining_ids:
            print(f"Successfully deleted chunk with ID: {chunk_to_delete}")
        else:
            print(f"Failed to delete chunk with ID: {chunk_to_delete}")

if __name__ == "__main__":
    main()
