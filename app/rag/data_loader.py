from pathlib import Path
import json
from typing import List, Dict


class RecipeLoader:
    """
    Loads recipes from a JSON file.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load(self) -> List[Dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Recipe file not found: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as f:
            return json.load(f)
