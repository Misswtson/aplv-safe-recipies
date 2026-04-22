from typing import List

from app.rag.vector_store import RecipeVectorStore
from app.rag.allergy_normalizer import normalize_allergies


class RecipeRetriever:
    """
    Handles safe semantic retrieval of recipes.
    """

    def __init__(self, vector_store: RecipeVectorStore | None = None):
        self.vector_store = vector_store or RecipeVectorStore()

    def search_safe_recipes(
        self,
        query: str,
        forbidden_allergies: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """
        Perform semantic search while excluding forbidden allergens.
        """

        # 1. Normalize allergies (egg → HUEVO, etc.)
        normalized = normalize_allergies(forbidden_allergies)

        # 2. Query vector DB with hard filter
        results = self.vector_store.search_safe(
            query=query,
            forbidden_allergens=list(normalized),
            top_k=top_k,
        )

        return results