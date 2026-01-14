import json
from pathlib import Path
from typing import List, Dict, Any


DATA_PATH = Path("data/recipes.json")


class RecipeLoader:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = data_path
        self.recipes = self._load_recipes()

    def _load_recipes(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("recipes.json must contain a list of recipes")

        return data

    def filter_by_allergies(self, excluded_allergies: List[str]) -> List[Dict[str, Any]]:
        """
        Returns recipes that are safe for the given allergy constraints.
        """
        safe_recipes = []

        for recipe in self.recipes:
            recipe_safe_for = set(recipe.get("safe_for", []))
            recipe_contains = set(recipe.get("contains", []))

            # If recipe contains an excluded allergen → skip
            if recipe_contains.intersection(excluded_allergies):
                continue

            # Recipe must explicitly declare safety
            if not set(excluded_allergies).issubset(recipe_safe_for):
                continue

            safe_recipes.append(recipe)

        return safe_recipes

    def to_documents(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts recipes into documents ready for embeddings.
        """
        documents = []

        for recipe in recipes:
            content = self._recipe_to_text(recipe)

            documents.append(
                {
                    "id": recipe["id"],
                    "content": content,
                    "metadata": {
                        "title": recipe["title"],
                        "safe_for": recipe["safe_for"],
                        "age_range": recipe["age_range"],
                        "tags": recipe.get("tags", [])
                    }
                }
            )

        return documents

    def _recipe_to_text(self, recipe: Dict[str, Any]) -> str:
        """
        Creates a human-readable text representation for LLM context.
        """
        ingredients = "\n".join(
            f"- {i['quantity']} {i['name']} ({i.get('notes', '')})"
            for i in recipe["ingredients"]
        )

        steps = "\n".join(
            f"{idx+1}. {step}"
            for idx, step in enumerate(recipe["steps"])
        )

        return f"""
Title: {recipe['title']}

Description:
{recipe['description']}

Ingredients:
{ingredients}

Preparation Steps:
{steps}

Nutritional Notes:
{recipe.get('nutritional_notes', 'N/A')}
""".strip()
