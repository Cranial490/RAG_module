class RAGError(Exception):
    code: str = "rag_error"


class ConfigError(RAGError):
    code = "config_error"


class InvalidQuery(RAGError):
    code = "invalid_query"
