from typing import Dict, Type

from ..parser.document_parser_base import DocumentParserBase
from ..parser.docx_parser import DocxParser


PARSER_REGISTRY: Dict[str, Type[DocumentParserBase]] = {
    "docx": DocxParser,
}


def register_parser(key: str, parser_cls: Type[DocumentParserBase]) -> None:
    PARSER_REGISTRY[key] = parser_cls


def get_parser(key: str, **kwargs) -> DocumentParserBase:
    parser_cls = PARSER_REGISTRY.get(key)
    if parser_cls is None:
        raise ValueError(f"Invalid parser key: {key}")
    return parser_cls(**kwargs)


def list_parsers() -> list[str]:
    return sorted(PARSER_REGISTRY)
