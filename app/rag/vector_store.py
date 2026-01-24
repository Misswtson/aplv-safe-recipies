from typing import List, Dict
import chromadb

from app.rag.embeddings import EmbeddingGenerator


class RecipeVectorStore:
    """
    Stores and retrieves recipe embeddings using ChromaDB.
    """

    def __init__(self, collection_name: str = "recipes"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_generator = EmbeddingGenerator()

    def add_recipes(self, recipes: List[Dict]):
    """
    Stores recipes as embeddings with allergy metadata.
    """

    texts = []
    ids = []
    metadatas = []

    for recipe in recipes:
        text = f"""
        Title: {recipe['title']}
        Ingredients: {', '.join(recipe['ingredients'])}
        Instructions: {recipe['instructions']}
        """
        texts.append(text)
        ids.append(recipe["id"])
        metadatas.append(recipe.get("metadata", {}))

    embeddings = self.embedding_generator.embed_batch(texts)

    self.collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Finds the most similar recipes to a query.
        """

        query_embedding = self.embedding_generator.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"][0]
