from typing import List
import os

from dotenv import load_dotenv
from openai import OpenAI

from .base_embedder import BaseEmbedder

load_dotenv()


class AzureEmbeddingGenerator(BaseEmbedder):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("AZURE_EMBEDDER_API_KEY")
        self.base_url = base_url or os.getenv("AZURE_EMBEDDER_URL")
        self.model = model or os.getenv("AZURE_EMBEDDER_MODEL")

        if not self.api_key:
            raise ValueError(
                "Azure embedder api_key is required. Pass it explicitly or set "
                "AZURE_EMBEDDER_API_KEY."
            )
        if not self.base_url:
            raise ValueError(
                "Azure embedder base_url is required. Pass it explicitly or set "
                "AZURE_EMBEDDER_URL."
            )
        if not self.model:
            raise ValueError(
                "Azure embedder model is required. Pass it explicitly or set "
                "AZURE_EMBEDDER_MODEL."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def embed(self, text: str) -> List[List[float]]:
        """
        Generates embeddings for a string.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        embeddings = [d.embedding for d in response.data]
        return embeddings
