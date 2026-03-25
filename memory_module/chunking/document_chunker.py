from typing import Dict, List
from uuid import NAMESPACE_URL, uuid5

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
        chunks = self.chunker.chunk(parsed_document.text or "")
        chunk_objects = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = getattr(chunk, "text", str(chunk))
            chunk_metadata = self._generate_chunk_metadata(chunk_metadata_input)
            chunk_metadata.chunk_version = f"{chunk_metadata.document_id}_chunk_{i}"
            chunk = Chunk(
                chunk_id=self._build_chunk_id(chunk_metadata.chunk_version),
                text=chunk_text,
                embedding=[],
                metadata=chunk_metadata,
                token_count=len(chunk_text),
            )
            chunk_objects.append(chunk)
        return chunk_objects

    def _build_chunk_metadata_input(
        self,
        parsed_document: DocumentParserResult,
        metadata: Dict,
    ) -> Dict:
        parser_metadata = parsed_document.file_metadata
        if parser_metadata is None:
            raise ValueError("DocumentChunker requires parser file metadata.")

        chunk_metadata_input = {
            "document_id": parser_metadata.document_id,
            "document_title": parser_metadata.document_title,
        }

        caller_tags = (metadata or {}).get("tags")
        if caller_tags is not None:
            chunk_metadata_input["tags"] = caller_tags

        return chunk_metadata_input

    def _build_chunk_id(self, chunk_version: str) -> str:
        return str(uuid5(NAMESPACE_URL, chunk_version))
