from app.rag.data_loader import RecipeLoader
from app.rag.vector_store import RecipeVectorStore


def run_ingestion():
    """
    Full ingestion pipeline:
    Load recipes -> embed -> store in vector DB
    """

    loader = RecipeLoader("data/recipes.json")
    recipes = loader.load()

    store = RecipeVectorStore()
    store.add_recipes(recipes)

    print(f"✅ Ingested {len(recipes)} recipes into vector store")


if __name__ == "__main__":
    run_ingestion()
