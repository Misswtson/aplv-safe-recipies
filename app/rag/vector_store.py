from typing import List, Dict
import chromadb

from app.rag.embeddings import EmbeddingGenerator


class RecipeVectorStore:
    """
    Stores and retrieves recipe embeddings using ChromaDB.
    Applies hard safety filters for allergies.
    """

    def __init__(self, collection_name: str = "recipes"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_generator = EmbeddingGenerator()

    def add_recipes(self, recipes: List[Dict]) -> None:
        texts: List[str] = []
        ids: List[str] = []
        metadatas: List[Dict] = []

        for recipe in recipes:
            text = (
                f"Title: {recipe['title']}\n"
                f"Ingredients: {', '.join(recipe['ingredients'])}\n"
                f"Instructions: {recipe['instructions']}"
            )

            texts.append(text)
            ids.append(recipe["id"])

            raw_metadata = recipe.get("metadata", {})
            normalized_metadata: Dict = {}

            for key, value in raw_metadata.items():
                if isinstance(value, list):
                    normalized_metadata[key] = (
                        ",".join(value) if value else "none"
                    )
                else:
                    normalized_metadata[key] = value

            metadatas.append(normalized_metadata)

        embeddings = self.embedding_generator.embed_batch(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_generator.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results["documents"][0]

    def search_safe(
        self,
        query: str,
        forbidden_allergens: List[str],
        top_k: int = 5,
    ) -> List[str]:
        query_embedding = self.embedding_generator.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "allergens": {
                    "$nin": forbidden_allergens
                }
            },
        )

        return results["documents"][0]
