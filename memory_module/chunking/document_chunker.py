from typing import List, Dict
from uuid import uuid4

from chonkie import TokenChunker

from .base_chunker import BaseChunker
from .data_models import ChunkMetadata, Chunk
from ..parser.data_models import DocumentParserResult

class DocumentChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 128,
        tokenizer: str = "character",
    ):
        """
        :param tokenizer: Tokenizer name understood by chonkie
        :param chunk_size: Maximum tokens per chunk
        :param chunk_overlap: Number of tokens to overlap between adjacent chunks
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, parsed_document: DocumentParserResult, metadata: Dict) -> List[Chunk]:
        chunk_metadata_input = self._build_chunk_metadata_input(parsed_document, metadata)
        chunks = self.chunker.chunk(parsed_document.content.get("text", ""))
        chunk_objects = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = getattr(chunk, "text", str(chunk))
            chunk_metadata = self._generate_chunk_metadata(chunk_metadata_input)
            chunk_metadata.chunk_version = f"{chunk_metadata.document_id}_chunk_{i}"
            chunk = Chunk(
                chunk_id=str(uuid4()),
                text=chunk_text,
                embedding=[],
                metadata=chunk_metadata,
                token_count=len(chunk_text),
                overlap_with_previous=self.chunk_overlap if i > 1 else 0,
                source_rank_score=None,
                is_deleted=False,
                is_verified=None,
                context_summary=None,
            )
            chunk_objects.append(chunk)
        return chunk_objects

    def _build_chunk_metadata_input(
        self,
        parsed_document: DocumentParserResult,
        metadata: Dict,
    ) -> Dict:
        parser_metadata = parsed_document.file_metadata
        chunk_metadata_input = dict(metadata or {})

        if parser_metadata is not None:
            chunk_metadata_input.setdefault("document_id", parser_metadata.doc_id)
            chunk_metadata_input.setdefault(
                "document_title",
                parser_metadata.document_title,
            )
            additional_info = parser_metadata.additional_info or {}
            chunk_metadata_input.setdefault("source_url", additional_info.get("source_url"))
            chunk_metadata_input.setdefault("section", additional_info.get("section"))
            chunk_metadata_input.setdefault("tags", additional_info.get("tags"))
            chunk_metadata_input.setdefault("tenant_id", additional_info.get("tenant_id"))
            chunk_metadata_input.setdefault("permissions", additional_info.get("permissions"))

        if "document_id" not in chunk_metadata_input:
            chunk_metadata_input["document_id"] = str(uuid4())
        if "document_title" not in chunk_metadata_input:
            chunk_metadata_input["document_title"] = None

        return chunk_metadata_input
