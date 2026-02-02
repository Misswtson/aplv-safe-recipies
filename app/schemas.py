from typing import List, TypedDict, Optional


class IngredientSchema(TypedDict):
    name: str
    quantity: str
    notes: Optional[str]


class RecipeSchema(TypedDict):
    id: str
    title: str
    description: str
    ingredients: List[IngredientSchema]
    steps: List[str]
    safe_for: List[str]
    contains: List[str]
    age_range: str
    tags: List[str]
