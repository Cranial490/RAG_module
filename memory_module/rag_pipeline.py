from typing import Any
from fastapi import UploadFile

from .chunking.base_chunker import BaseChunker
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
        if not isinstance(strategy_key, str):
            raise TypeError(
                f"{base_key}_key must be a string. Only one {component_name} strategy "
                "can be selected at a time."
            )

        #buid the params for the class
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

    def indexer(self, document: UploadFile, metadata: dict[str, Any] | None = None):
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

        parsed_document = self.parser.convert(document)
        chunks = self.chunker.chunk(parsed_document, metadata or {})

        for chunk in chunks:
            embedding = self.embedder.embed(chunk.text)
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            chunk.embedding = embedding

        self.vector_db.add_chunks(chunks)
        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ):
        if self.embedder is None:
            raise RuntimeError("Retrieve requires an embedder strategy.")
        if self.retriever is None:
            raise RuntimeError("Retrieve requires a retrieval strategy.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieve requires a non-empty query string.")

        embedded_query = self.embedder.embed(query)
        if embedded_query and isinstance(embedded_query[0], list):
            embedded_query = embedded_query[0]

        return self.retriever.retrieve(
            embedded_query=embedded_query,
            top_k=top_k,
            filters=filters,
        )
