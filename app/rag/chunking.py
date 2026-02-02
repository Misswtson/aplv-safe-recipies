from typing import List


def chunk_text(
    text: str,
    max_tokens: int = 200,
    overlap: int = 40
) -> List[str]:
    """
    Splits text into overlapping chunks.
    Token-agnostic simple chunker (word-based).
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += max_tokens - overlap

    return chunks
