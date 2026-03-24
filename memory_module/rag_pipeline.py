from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from .chunking.base_chunker import BaseChunker
from .chunking.data_models import ChunkMetadata
from .embedder.base_embedder import BaseEmbedder
from .factory.chunking_factory import get_chunker
from .factory.embedder_factory import get_embedder
from .factory.parser_factory import get_parser
from .factory.retrieval_factory import get_retrieval_strategy
from .factory.vector_db_factory import get_vector_db
from .parser.document_parser_base import DocumentParserBase
from .retrieval.base_retrieval import BaseRetrievalStrategy
from .vector_db.base_vector_db import BaseVectorMemory


class RAGPipeline:
    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("RAGPipeline config must be a dict.")

        self.config = config
        self.parser = self._resolve_component(
            component_name="parser",
            factory=get_parser,
            base_key="parser",
        )
        self.chunker = self._resolve_component(
            component_name="chunker",
            factory=get_chunker,
            base_key="chunker",
        )
        self.embedder = self._resolve_component(
            component_name="embedder",
            factory=get_embedder,
            base_key="embedder",
        )
        self.vector_db = self._resolve_component(
            component_name="vector db",
            factory=get_vector_db,
            base_key="vector_db",
        )
        self.retriever = self._resolve_component(
            component_name="retrieval",
            factory=get_retrieval_strategy,
            base_key="retrieval",
            extra_kwargs={"vector_db": self.vector_db},
        )

    parser: DocumentParserBase | None
    chunker: BaseChunker | None
    embedder: BaseEmbedder | None
    vector_db: BaseVectorMemory | None
    retriever: BaseRetrievalStrategy | None

    def _resolve_component(
        self,
        component_name: str,
        factory,
        base_key: str,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        strategy_key = self.config.get(f"{base_key}_key")
        if strategy_key is None:
            return None

        kwargs_key = f"{base_key}_kwargs"
        strategy_kwargs = self.config.get(kwargs_key, {})
        if not isinstance(strategy_kwargs, dict):
            raise TypeError(f"{kwargs_key} must be a dict.")
        if extra_kwargs:
            for key, value in extra_kwargs.items():
                strategy_kwargs.setdefault(key, value)

        try:
            return factory(strategy_key, **strategy_kwargs)
        except ValueError as exc:
            error_message = str(exc)
            if error_message == f"Invalid {component_name} key: {strategy_key}":
                raise ValueError(
                    f"Invalid {component_name} strategy key: {strategy_key}"
                ) from exc
            raise ValueError(
                f"Failed to initialize {component_name} strategy "
                f"'{strategy_key}': {error_message}"
            ) from exc

    def indexer(self, document, metadata: dict[str, Any] | None = None):
        if self.parser is None:
            raise RuntimeError("Indexer requires a parser strategy.")
        if self.chunker is None:
            raise RuntimeError("Indexer requires a chunker strategy.")
        if self.embedder is None:
            raise RuntimeError("Indexer requires an embedder strategy.")
        if self.vector_db is None:
            raise RuntimeError("Indexer requires a vector_db strategy.")

        if hasattr(document, "file") and hasattr(document.file, "seek"):
            document.file.seek(0)
        if hasattr(self.parser, "accepts") and not self.parser.accepts(document):
            parser_error = getattr(self.parser, "last_error", None)
            if parser_error:
                raise ValueError(f"Parser rejected the provided document: {parser_error}")
            raise ValueError("Parser does not accept the provided document.")
        if hasattr(document, "file") and hasattr(document.file, "seek"):
            document.file.seek(0)

        parsed_document = self.parser.convert(document)
        parser_metadata = parsed_document.file_metadata
        chunks = self.chunker.chunk(parsed_document, metadata or {})
        normalized_chunks = []

        for index, chunk in enumerate(chunks, start=1):
            text = getattr(chunk, "text", str(chunk))
            embedding = self.embedder.embed(text)
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]

            resolved_metadata = getattr(chunk, "metadata", None)
            if resolved_metadata is None:
                resolved_metadata = ChunkMetadata(
                    document_id=(
                        parser_metadata.doc_id if parser_metadata is not None else str(uuid4())
                    ),
                    document_title=(
                        parser_metadata.document_title if parser_metadata is not None else None
                    ),
                    chunk_version=(
                        f"{parser_metadata.doc_id}_chunk_{index}"
                        if parser_metadata is not None
                        else f"chunk_{index}"
                    ),
                    created_at=datetime.now(),
                )

            existing_chunk_id = getattr(chunk, "chunk_id", None)
            if not existing_chunk_id:
                chunk.chunk_id = str(uuid4())
            else:
                chunk.chunk_id = str(existing_chunk_id)

            if not resolved_metadata.chunk_version:
                resolved_metadata.chunk_version = (
                    f"{resolved_metadata.document_id}_chunk_{index}"
                )
            chunk.text = text
            chunk.embedding = embedding
            chunk.metadata = resolved_metadata
            normalized_chunks.append(chunk)

        self.vector_db.add_chunks(normalized_chunks)
        return normalized_chunks
