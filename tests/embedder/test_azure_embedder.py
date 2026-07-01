from types import SimpleNamespace

import pytest

from memory_module.embedder.azure_open_ai_embedder import AzureEmbeddingGenerator
from memory_module.errors import EmbedderFailed


class _StubOpenAIEmbeddings:
    def __init__(self, response):
        self._response = response

    def create(self, *, model, input):
        return self._response


class _StubOpenAIClient:
    def __init__(self, response):
        self.embeddings = _StubOpenAIEmbeddings(response)


def _make_embedder(response) -> AzureEmbeddingGenerator:
    embedder = AzureEmbeddingGenerator(
        api_key="test-key",
        base_url="https://example.invalid/",
        model="test-model",
    )
    embedder.client = _StubOpenAIClient(response)
    return embedder


def test_embed_returns_flat_list_of_floats():
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
    embedder = _make_embedder(response)

    result = embedder.embed("hello")

    assert result == [0.1, 0.2, 0.3]


def test_embed_raises_embedder_failed_when_response_data_empty():
    response = SimpleNamespace(data=[])
    embedder = _make_embedder(response)

    with pytest.raises(EmbedderFailed):
        embedder.embed("hello")
