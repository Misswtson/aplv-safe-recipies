from typing import List, Dict
import chromadb

from app.rag.embeddings import EmbeddingGenerator


class RecipeVectorStore:
    """
    Stores and retrieves recipe embeddings using ChromaDB.
    Applies hard safety filters for allergies.
    """

    def __init__(self, collection_name: str = "recipes"):
        """
        Initialize ChromaDB client, collection, and embedding generator.
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_generator = EmbeddingGenerator()

    def add_recipes(self, recipes: List[Dict]):
        """
        Store recipes as embeddings with metadata.
        Each recipe must include:
        - id
        - title
        - ingredients
        - instructions
        - metadata (diet, allergens, etc.)
        """

        texts = []
        ids = []
        metadatas = []

        for recipe in recipes:
            text = (
                f"Title: {recipe['title']}\n"
                f"Ingredients: {', '.join(recipe['ingredients'])}\n"
                f"Instructions: {recipe['instructions']}"
            )

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
        Basic semantic search (NO safety filtering).
        """

        query_embedding = self.embedding_generator.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results["documents"][0]

    def search_safe(
        self,
        query: str,
        forbidden_allergens: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Semantic search WITH hard allergy filters.
        Recipes containing forbidden allergens are excluded at DB level.
        """

        # 1 Convert query to embedding
        query_embedding = self.embedding_generator.embed_text(query)

        # 2️ Query vector DB with metadata filters
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={
                "allergens": {
                    "$nin": forbidden_allergens
                }
            }
        )

        # 3️ Return safe documents only
        return results["documents"][0]
