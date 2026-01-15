from fastapi import FastAPI, Query
from typing import List

from app.rag.data_loader import RecipeLoader

app = FastAPI(
    title="APLV Safe Recipes API",
    description="API to retrieve safe recipes for children with food allergies",
    version="0.1.0",
)

loader = RecipeLoader()


@app.get("/")
def root():
    return {"status": "ok", "message": "APLV Recipes API running"}


@app.get("/recipes")
def get_recipes(allergies: str = Query(..., description="Comma-separated allergies")):
    """
    Example:
    /recipes?allergies=leche,egg
    """
    allergy_list = [a.strip() for a in allergies.split(",")]

    recipes = loader.get_safe_recipes_for_user(allergy_list)

    return recipes
