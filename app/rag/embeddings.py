from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates text embeddings using a local SentenceTransformer model.
    """

    def __init__(self):
        """
        Load the embedding model once when the class is created.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text into an embedding vector.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of texts into embedding vectors.
        """
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
