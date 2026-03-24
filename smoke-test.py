import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from memory_module.chunking.data_models import Chunk, ChunkMetadata
from memory_module.embedder.azure_open_ai_embedder import AzureEmbeddingGenerator
from memory_module.vector_db.QdrantVectorMemory import QdrantVectorMemory


load_dotenv()


def main() -> None:
    api_key = os.getenv("AZURE_EMBEDDER_API_KEY")
    base_url = os.getenv("AZURE_EMBEDDER_URL")
    model = os.getenv("AZURE_EMBEDDER_MODEL", "text-embedding-ada-002")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "smoke_test_embeddings")

    if not api_key or not base_url:
        raise RuntimeError(
            "Missing AZURE_EMBEDDER_API_KEY or AZURE_EMBEDDER_URL in environment."
        )

    text = "Make magic happen"
    chunk_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    embedder = AzureEmbeddingGenerator(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    embedding_batch = embedder.embed(text)
    embedding = embedding_batch[0]

    qdrant_client = QdrantClient(url=qdrant_url)
    memory = QdrantVectorMemory(
        vector_client=qdrant_client,
        collection_name=collection_name,
        vector_size=len(embedding),
        create_collection_if_not_exists=True,
    )

    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            document_id=document_id,
            document_title="Smoke Test Embedding",
            tags=["smoke-test"],
        ),
    )

    memory.add_chunks([chunk])

    collection_count = qdrant_client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print(f"Stored chunk {chunk_id} in '{collection_name}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"Collection count: {collection_count}")


if __name__ == "__main__":
    main()
