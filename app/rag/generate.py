from typing import List
from app.rag.retrieval import RecipeRetriever


class RecipeGenerator:
    """
    Final generation layer of the RAG pipeline.
    Combines retrieved safe context with user intent
    to produce a grounded response.
    """

    def __init__(self):
        self.retriever = RecipeRetriever()

    def generate(
        self,
        query: str,
        forbidden_allergies: List[str],
        top_k: int = 5
    ) -> str:
        """
        Generate a safe response grounded in retrieved recipes.
        """

        # 1️⃣ Retrieve safe context
        contexts = self.retriever.retrieve(
            query=query,
            forbidden_allergies=forbidden_allergies,
            top_k=top_k
        )

        if not contexts:
            return (
                "No encontré recetas seguras para esa búsqueda "
                "con las restricciones indicadas."
            )

        # 2️⃣ Build grounded prompt (LLM-ready)
        context_block = "\n\n---\n\n".join(contexts)

        response = (
            "Basado en recetas seguras para las alergias indicadas:\n\n"
            f"{context_block}\n\n"
            "Si quieres, puedo ayudarte a adaptar la receta "
            "según edad, textura o tiempo de preparación."
        )

        return response
