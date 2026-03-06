"""
similarity.py
─────────────
Functions to measure how similar two items are.
Supports cosine similarity, Jaccard similarity, and dot-product similarity.
All vectors are represented as plain Python dicts (sparse) or lists (dense).
"""

import math


# ── Dense-vector helpers ──────────────────────────────────────────────────────

def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Return the dot product of two equal-length vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length.")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def magnitude(vec: list[float]) -> float:
    """Return the Euclidean magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x ** 2 for x in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two dense vectors.
    Returns a value in [-1, 1]; higher means more similar.
    Returns 0.0 if either vector is the zero vector.
    """
    mag_a, mag_b = magnitude(vec_a), magnitude(vec_b)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product(vec_a, vec_b) / (mag_a * mag_b)


# ── Sparse-vector (dict) helpers ──────────────────────────────────────────────

def cosine_similarity_sparse(vec_a: dict, vec_b: dict) -> float:
    """
    Cosine similarity for sparse vectors represented as {feature: value} dicts.
    Efficient: only iterates over non-zero features.
    """
    # Dot product over shared keys
    shared_keys = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in shared_keys)

    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Set-based similarity ──────────────────────────────────────────────────────

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Useful for comparing tags, genres, or keyword sets.
    Returns 0.0 when both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


# ── Text-based similarity ─────────────────────────────────────────────────────

def text_overlap_similarity(text_a: str, text_b: str) -> float:
    """
    Simple word-overlap similarity (bag-of-words Jaccard).
    Lowercases and tokenises on whitespace before comparing.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    return jaccard_similarity(words_a, words_b)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_similarity(item_a: dict, item_b: dict, method: str = "cosine") -> float:
    """
    High-level similarity between two item dicts.

    Each item dict should have at least ONE of:
        'vector'  – list[float]  (dense embedding)
        'tags'    – set[str]     (genre / keyword tags)
        'title'   – str          (text description)

    method: 'cosine' | 'jaccard' | 'text'
    """
    if method == "cosine":
        va = item_a.get("vector")
        vb = item_b.get("vector")
        if va is None or vb is None:
            raise ValueError("Both items need a 'vector' key for cosine similarity.")
        return cosine_similarity(va, vb)

    elif method == "jaccard":
        ta = set(item_a.get("tags", []))
        tb = set(item_b.get("tags", []))
        return jaccard_similarity(ta, tb)

    elif method == "text":
        return text_overlap_similarity(
            item_a.get("title", ""),
            item_b.get("title", ""),
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'cosine', 'jaccard', or 'text'.")
