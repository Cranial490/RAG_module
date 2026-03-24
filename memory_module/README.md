# Qdrant Vector Memory

This module provides an implementation of `BaseVectorMemory` using [Qdrant](https://qdrant.tech/) as the vector database backend.

## Overview

`QdrantVectorMemory` is a concrete implementation of the abstract `BaseVectorMemory` class that uses Qdrant for storing and retrieving vector embeddings. It provides methods for:

- Adding memory chunks to the vector database
- Retrieving memory chunks based on semantic similarity
- Filtering results based on metadata
- Deleting memory chunks

## Requirements

- Python 3.7+
- qdrant-client
- pydantic

## Installation

Make sure you have Qdrant running. You can use the provided `docker-compose.yml` file to start a Qdrant instance:

```bash
docker-compose up -d
```

## Usage

### Basic Usage

```python
from qdrant_client import QdrantClient
from memory_module.QdrantVectorMemory import QdrantVectorMemory
from memory_module.data_models import MemoryChunk
from datetime import datetime

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Create a QdrantVectorMemory instance
memory = QdrantVectorMemory(
    vector_client=client,
    collection_name="my_memories",
    vector_size=1536,  # Adjust based on your embedding model
    create_collection_if_not_exists=True
)

# Create a memory chunk
chunk = MemoryChunk(
    id="unique-id-1",
    content="This is a memory about vector databases.",
    embedding=[0.1, 0.2, ..., 0.9],  # Your actual embedding vector
    metadata={"source": "article", "topic": "databases"},
    timestamp=datetime.now()
)

# Add the chunk to the vector database
memory.add_chunks([chunk])

# Retrieve similar chunks
query_vector = [0.15, 0.25, ..., 0.85]  # Your query embedding vector
results = memory.retrieve(embedded_query=query_vector, top_k=5)

# Retrieve with filters
filtered_results = memory.retrieve(
    embedded_query=query_vector,
    top_k=5,
    filters={"metadata.topic": "databases"}
)

# Delete a chunk
memory.delete("unique-id-1")
```

### Configuration Options

When initializing `QdrantVectorMemory`, you can configure:

- `vector_client`: The Qdrant client instance
- `collection_name`: Name of the collection to store memory chunks (default: "memory_chunks")
- `vector_size`: Dimension of the embedding vectors (default: 1536)
- `distance`: Distance metric to use for similarity search (default: Distance.COSINE)
- `create_collection_if_not_exists`: Whether to create the collection if it doesn't exist (default: True)

## Example

See `example_usage.py` for a complete working example.

## Advanced Usage

### Metadata Filtering

You can filter results based on metadata fields:

```python
# Filter by a top-level field
results = memory.retrieve(
    embedded_query=query_vector,
    filters={"content": "specific content"}
)

# Filter by a nested metadata field
results = memory.retrieve(
    embedded_query=query_vector,
    filters={"metadata.source": "article"}
)
```

### Error Handling

The implementation includes error handling for common Qdrant operations. Errors are wrapped in a `RuntimeError` with a descriptive message.

## Integration with Embedding Models

This implementation focuses on the vector database operations. You'll need to generate embeddings separately using a model of your choice (e.g., OpenAI, Hugging Face, etc.) before adding chunks to the database.
