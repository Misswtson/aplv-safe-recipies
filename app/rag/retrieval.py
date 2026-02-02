from typing import List
from app.rag.vector_store import RecipeVectorStore
from app.rag.allergy_normalizer import normalize_allergies


class RecipeRetriever:
    """
    High-level retrieval layer for the RAG pipeline.
    Applies semantic search + hard safety constraints.
    """

    def __init__(self):
        self.store = RecipeVectorStore()

    def retrieve(
        self,
        query: str,
        forbidden_allergies: List[str],
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve safe, relevant recipe chunks for a user query.
        """

        # 1️⃣ Normalize allergies (egg → HUEVO, milk → APLV, etc.)
        normalized_allergies = normalize_allergies(forbidden_allergies)

        # 2️⃣ Query vector store with hard filters
        results = self.store.search_safe(
            query=query,
            forbidden_allergens=list(normalized_allergies),
            top_k=top_k
        )

        return results
