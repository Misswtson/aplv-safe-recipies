from app.rag.vector_store import RecipeVectorStore


def test_search_safe_excludes_forbidden_allergens():
    """
    GIVEN recipes with and without allergens
    WHEN performing a safe search
    THEN recipes containing forbidden allergens must be excluded
    """

    store = RecipeVectorStore()

    store.add_recipes([
        {
            "id": "safe_recipe",
            "title": "Pollo al horno",
            "ingredients": ["pollo", "sal"],
            "instructions": "Hornear 40 minutos",
            "metadata": {
                "diet": "aplv_safe",
                "allergens": []
            }
        },
        {
            "id": "dangerous_recipe",
            "title": "Torta con huevo",
            "ingredients": ["harina", "egg"],
            "instructions": "Hornear",
            "metadata": {
                "diet": "not_safe",
                "allergens": ["egg"]
            }
        }
    ])

    results = store.search_safe(
        query="comida para niños",
        forbidden_allergens=["egg"]
    )

    joined_results = " ".join(results).lower()

    assert "huevo" not in joined_results
    assert "egg" not in joined_results
