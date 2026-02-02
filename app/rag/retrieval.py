from typing import List
from app.rag.vector_store import RecipeVectorStore
from app.rag.allergy_normalizer import normalize_allergies


class SafeRecipeRetriever:
    """
    High-level retrieval layer applying safety + semantic search.
    """

    def __init__(self, store: RecipeVectorStore):
        self.store = store

    def search(
        self,
        query: str,
        user_allergies: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Perform a safe semantic search given user allergies.
        """

        normalized_allergies = normalize_allergies(user_allergies)

        return self.store.search_safe(
            query=query,
            forbidden_allergens=list(normalized_allergies),
            top_k=top_k
        )
