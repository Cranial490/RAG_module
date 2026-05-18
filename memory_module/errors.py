class RAGError(Exception):
    code: str = "rag_error"


class ConfigError(RAGError):
    code = "config_error"


class InvalidQuery(RAGError):
    code = "invalid_query"


class ParserRejected(RAGError):
    code = "parser_rejected"


class EmbedderFailed(RAGError):
    code = "embedder_failed"


class VectorDBFailed(RAGError):
    code = "vector_db_failed"
