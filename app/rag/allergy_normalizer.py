from typing import List, Set
import re


# -------------------------------------------------
# 1️⃣ Categorías canónicas de alergias (nivel médico)
# -------------------------------------------------

ALLERGY_SYNONYMS = {
    "APLV": {
        "leche",
        "lácteos",
        "proteína de leche",
        "caseína",
        "suero de leche",
        "milk",
        "dairy",
    },
    "HUEVO": {
        "huevo",
        "egg",
        "albúmina",
    },
    "SOYA": {
        "soya",
        "soy",
        "lecitina de soya",
    },
    "FRUTOS_SECOS": {
        "nueces",
        "almendras",
        "avellanas",
        "maní",
        "peanuts",
        "tree nuts",
    },
}


# -------------------------------------------------
# 2️⃣ Normalización semántica de ingredientes
# -------------------------------------------------

INGREDIENT_NORMALIZATION = {
    # huevo
    "egg": "huevo",
    "eggs": "huevo",
    "albúmina": "huevo",
    "huevo": "huevo",

    # leche
    "milk": "leche",
    "dairy": "leche",
    "caseína": "leche",
    "leche": "leche",

    # soya
    "soy": "soya",
    "soja": "soya",
    "soya": "soya",
}


# -------------------------------------------------
# 3️⃣ API PÚBLICA
# -------------------------------------------------

def normalize_allergies(raw_allergies: List[str]) -> Set[str]:
    """
    Maps user-provided allergy terms to canonical allergy labels.
    Example: ["egg", "leche"] → {"HUEVO", "APLV"}
    """
    normalized = set()
    raw_lower = [a.lower().strip() for a in raw_allergies if a]

    for canonical, synonyms in ALLERGY_SYNONYMS.items():
        for term in raw_lower:
            if term in synonyms or term.upper() == canonical:
                normalized.add(canonical)

    return normalized


def normalize_ingredients(raw_ingredients: List[str]) -> Set[str]:
    """
    Normalizes ingredient names to a canonical semantic form.
    Example: ["egg", "milk"] → {"huevo", "leche"}
    """
    normalized = set()

    for ingredient in raw_ingredients:
        if not ingredient:
            continue

        key = ingredient.lower().strip()
        normalized.add(INGREDIENT_NORMALIZATION.get(key, key))

    return normalized


# -------------------------------------------------
# 4️⃣ 🔥 DETECCIÓN AUTOMÁTICA DESDE TEXTO LIBRE
# -------------------------------------------------

def detect_allergens_from_text(text: str) -> Set[str]:
    """
    Detects canonical allergens from free text.
    Example:
        "Bizcocho con egg y milk"
        → {"HUEVO", "APLV"}
    """
    if not text:
        return set()

    detected = set()
    text_lower = text.lower()

    # tokenización simple y segura
    tokens = set(re.findall(r"\b\w+\b", text_lower))

    # normalizamos tokens semánticamente
    normalized_tokens = {
        INGREDIENT_NORMALIZATION.get(token, token)
        for token in tokens
    }

    # mapeamos a categorías médicas
    for canonical, synonyms in ALLERGY_SYNONYMS.items():
        if normalized_tokens & synonyms:
            detected.add(canonical)

    return detected
