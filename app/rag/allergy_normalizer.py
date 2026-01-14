from typing import List, Set


ALLERGY_SYNONYMS = {
    "APLV": {
        "leche",
        "lácteos",
        "proteína de leche",
        "caseína",
        "suero de leche",
        "milk",
        "dairy"
    },
    "HUEVO": {
        "huevo",
        "egg",
        "albúmina"
    },
    "SOYA": {
        "soya",
        "soy",
        "lecitina de soya"
    },
    "FRUTOS_SECOS": {
        "nueces",
        "almendras",
        "avellanas",
        "maní",
        "peanuts",
        "tree nuts"
    }
}


def normalize_allergies(raw_allergies: List[str]) -> Set[str]:
    """
    Maps user-provided allergy terms to canonical allergy labels.
    """
    normalized = set()
    raw_lower = [a.lower() for a in raw_allergies]

    for canonical, synonyms in ALLERGY_SYNONYMS.items():
        for term in raw_lower:
            if term in synonyms or term.upper() == canonical:
                normalized.add(canonical)

    return normalized
