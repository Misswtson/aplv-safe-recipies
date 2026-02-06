from typing import List, Optional
from pydantic import BaseModel, Field


# =========================
# Request Schemas
# =========================

class RecipeSearchRequest(BaseModel):
    """
    Input schema for recipe search / generation.
    """
    query: str = Field(
        ...,
        description="User natural language query"
    )

    forbidden_allergies: List[str] = Field(
        default_factory=list,
        description="List of allergies to strictly exclude"
    )

    age_months: Optional[int] = Field(
        None,
        description="Age of the child in months (optional)"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of candidate recipes to retrieve"
    )


# =========================
# Internal / Domain Schemas
# =========================

class Ingredient(BaseModel):
    name: str
    quantity: Optional[str] = None
    notes: Optional[str] = None


class Recipe(BaseModel):
    """
    Canonical recipe representation inside the system.
    """
    id: str
    title: str
    description: Optional[str] = None
    ingredients: List[Ingredient]
    steps: List[str]

    safe_for: List[str] = Field(
        default_factory=list,
        description="Canonical allergy labels the recipe is safe for"
    )

    contains: List[str] = Field(
        default_factory=list,
        description="Canonical allergy labels present in the recipe"
    )

    age_range: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# =========================
# Response Schemas
# =========================

class RecipeResult(BaseModel):
    """
    Individual retrieved recipe (textual form).
    """
    content: str


class RecipeSearchResponse(BaseModel):
    """
    Final API response returned to the user.
    """
    query: str
    forbidden_allergies: List[str]
    results: List[RecipeResult]
    message: Optional[str] = None
