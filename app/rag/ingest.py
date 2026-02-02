from app.rag.data_loader import RecipeLoader
from app.rag.vector_store import RecipeVectorStore
from app.rag.chunking import chunk_text


def build_recipe_text(recipe: dict) -> str:
    """
    Build a canonical text representation of a recipe
    used for embeddings.
    """

    ingredients = recipe.get("ingredients", [])
    ingredients_text = ", ".join(
        f"{ing.get('quantity', '')} {ing.get('name', '')}".strip()
        for ing in ingredients
        if isinstance(ing, dict)
    )

    steps_text = " ".join(recipe.get("steps", []))

    return (
        f"Title: {recipe.get('title', '')}\n"
        f"Description: {recipe.get('description', '')}\n"
        f"Ingredients: {ingredients_text}\n"
        f"Steps: {steps_text}\n"
        f"Safe for: {', '.join(recipe.get('safe_for', []))}\n"
        f"Age range: {recipe.get('age_range', '')}"
    )


def run_ingestion():
    """
    End-to-end ingestion pipeline:
    JSON → chunks → embeddings → vector DB
    """

    loader = RecipeLoader("data/recipes.json")
    recipes = loader.get_recipes()

    store = RecipeVectorStore()

    documents = []

    for recipe in recipes:
        base_text = build_recipe_text(recipe)

        chunks = chunk_text(base_text)

        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{recipe['id']}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "recipe_id": recipe["id"],
                    "safe_for": ",".join(recipe.get("safe_for", [])),
                    "contains": ",".join(recipe.get("contains", [])),
                    "age_range": recipe.get("age_range", ""),
                }
            })

    store.add_documents(documents)

    print(f"✅ Ingested {len(documents)} recipe chunks into vector store")


if __name__ == "__main__":
    run_ingestion()
