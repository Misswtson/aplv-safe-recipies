from typing import List, Dict, Any
import chromadb


class _DummyEmbeddingGenerator:
    """
    Embedding simple para tests.
    Devuelve vectores determinísticos y livianos.
    """

    def embed_text(self, text: str) -> List[float]:
        return [float(len(text))]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]


class RecipeVectorStore:
    def __init__(
        self,
        collection_name: str = "recipes_test",
        embedding_generator=None,
    ):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        # 👉 si no se pasa embedding generator, usamos dummy (tests)
        self.embedding_generator = (
            embedding_generator or _DummyEmbeddingGenerator()
        )

    def _normalize_ingredients(self, ingredients: Any) -> str:
        if isinstance(ingredients, list):
            if not ingredients:
                return ""
            if isinstance(ingredients[0], dict):
                return ", ".join(
                    f"{ing.get('quantity', '')} {ing.get('name', '')}".strip()
                    for ing in ingredients
                )
            if isinstance(ingredients[0], str):
                return ", ".join(ingredients)

        if isinstance(ingredients, str):
            return ingredients

        return ""

    def add_recipes(self, recipes: List[Dict[str, Any]]) -> None:
        documents = []
        metadatas = []
        ids = []

        for recipe in recipes:
            recipe_id = recipe.get("id")
            if not recipe_id:
                continue

            title = recipe.get("title", "")
            description = recipe.get("description", "")
            steps = recipe.get("steps", "")

            ingredients_text = self._normalize_ingredients(
                recipe.get("ingredients", [])
            )

            document = (
                f"title: {title}\n"
                f"description: {description}\n"
                f"ingredients: {ingredients_text}\n"
                f"steps: {steps}"
            )

            documents.append(document)
            ids.append(str(recipe_id))

            contains = recipe.get("contains", "")
            if isinstance(contains, list):
                contains = ",".join(contains)

            metadatas.append(
                {
                    "title": title,
                    "contains": contains or "",
                }
            )

        if not documents:
            return

        embeddings = self.embedding_generator.embed_texts(documents)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search_safe(
        self,
        query: str,
        forbidden_allergens: List[str],
        top_k: int = 5,
    ) -> List[str]:
        query_embedding = self.embedding_generator.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas"],
        )

        safe_docs = []
        forbidden = [a.lower() for a in forbidden_allergens]

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            text = doc.lower()

            # filtro por texto
            if any(allergen in text for allergen in forbidden):
                continue

            # filtro por metadata
            contains = str(meta.get("contains", "")).lower()
            if any(allergen in contains for allergen in forbidden):
                continue

            safe_docs.append(doc)

            if len(safe_docs) >= top_k:
                break

        return safe_docs
