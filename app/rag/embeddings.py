from typing import List
import os

from openai import OpenAI


class EmbeddingGenerator:
    """
    Small wrapper around an embedding model.
    Converts text into numerical vectors.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """
        Converts a single text into an embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )

        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Converts multiple texts into embeddings.
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [item.embedding for item in response.data]
