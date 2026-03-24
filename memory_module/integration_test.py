"""
Integration test for QdrantVectorMemory with a real Qdrant instance.
This test requires a running Qdrant instance (e.g., via docker-compose).

Run with:
    python integration_test.py
"""

import uuid
import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from data_models import MemoryChunk
from QdrantVectorMemory import QdrantVectorMemory

def run_integration_test():
    print("Starting QdrantVectorMemory integration test...")
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(url="http://localhost:6333")
    
    # Create a unique collection name for this test
    collection_name = f"test_memory_{uuid.uuid4().hex[:8]}"
    print(f"Using collection: {collection_name}")
    
    try:
        # Create a QdrantVectorMemory instance
        memory = QdrantVectorMemory(
            vector_client=client,
            collection_name=collection_name,
            vector_size=4,  # Small size for testing
            create_collection_if_not_exists=True
        )
        
        # Create some sample memory chunks
        print("Creating sample memory chunks...")
        chunks = [
            MemoryChunk(
                id=f"test-{uuid.uuid4().hex[:8]}",
                content="Qdrant is a vector database for similarity search.",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={"source": "documentation", "topic": "vector_db"},
                timestamp=datetime.now()
            ),
            MemoryChunk(
                id=f"test-{uuid.uuid4().hex[:8]}",
                content="Vector databases are used for semantic search applications.",
                embedding=[0.2, 0.3, 0.4, 0.5],
                metadata={"source": "article", "topic": "vector_db"},
                timestamp=datetime.now()
            ),
            MemoryChunk(
                id=f"test-{uuid.uuid4().hex[:8]}",
                content="Python is a popular programming language for data science.",
                embedding=[0.3, 0.4, 0.5, 0.6],
                metadata={"source": "tutorial", "topic": "programming"},
                timestamp=datetime.now()
            )
        ]
        
        # Add chunks to the vector database
        print("Adding chunks to Qdrant...")
        memory.add_chunks(chunks)
        
        # Wait for indexing to complete
        print("Waiting for indexing...")
        time.sleep(1)
        
        # Retrieve chunks based on a query vector
        print("\nRetrieving similar chunks...")
        query_vector = [0.2, 0.3, 0.4, 0.5]  # Similar to the second chunk
        results = memory.retrieve(embedded_query=query_vector, top_k=2)
        
        print(f"Found {len(results)} results:")
        for i, chunk in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"ID: {chunk.id}")
            print(f"Content: {chunk.content}")
            print(f"Metadata: {chunk.metadata}")
        
        # Retrieve chunks with filter
        print("\nRetrieving chunks with filter...")
        filtered_results = memory.retrieve(
            embedded_query=query_vector,
            top_k=2,
            filters={"metadata.topic": "programming"}
        )
        
        print(f"Found {len(filtered_results)} filtered results:")
        for i, chunk in enumerate(filtered_results):
            print(f"\nFiltered Result {i+1}:")
            print(f"ID: {chunk.id}")
            print(f"Content: {chunk.content}")
            print(f"Metadata: {chunk.metadata}")
        
        # Delete a chunk
        if chunks:
            chunk_to_delete = chunks[0].id
            print(f"\nDeleting chunk with ID: {chunk_to_delete}")
            memory.delete(chunk_to_delete)
            
            # Verify deletion
            all_results = memory.retrieve(embedded_query=query_vector, top_k=10)
            remaining_ids = [chunk.id for chunk in all_results]
            
            if chunk_to_delete not in remaining_ids:
                print(f"Successfully deleted chunk with ID: {chunk_to_delete}")
            else:
                print(f"Failed to delete chunk with ID: {chunk_to_delete}")
        
        print("\nIntegration test completed successfully!")
        
    except Exception as e:
        print(f"Error during integration test: {str(e)}")
        raise
    
    finally:
        # Clean up: delete the test collection
        print(f"\nCleaning up: deleting collection {collection_name}...")
        try:
            client.delete_collection(collection_name=collection_name)
            print("Collection deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")

if __name__ == "__main__":
    run_integration_test()
