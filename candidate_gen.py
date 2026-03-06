"""
candidate_gen.py
────────────────
Generates a shortlist of candidate items for a given user or query.
Uses fast, lightweight filters before the expensive ranking step.

Catalogue format (dict of item_id → item dict):
    {
        "item_001": {
            "title":    "The Matrix",
            "tags":     {"sci-fi", "action", "cyberpunk"},
            "vector":   [0.1, 0.8, 0.3, ...],
            "rating":   8.7,
            "year":     1999,
        },
        ...
    }
"""

from similarity import cosine_similarity, jaccard_similarity


# ── Tag-based filtering ───────────────────────────────────────────────────────

def candidates_by_tags(
    catalogue: dict,
    query_tags: set[str],
    min_overlap: int = 1,
    limit: int = 50,
) -> list[str]:
    """
    Return item IDs whose tag sets share at least `min_overlap` tags
    with `query_tags`.

    Args:
        catalogue:   Full item catalogue.
        query_tags:  Tags we want to match (e.g. from a liked item or user profile).
        min_overlap: Minimum number of shared tags required.
        limit:       Maximum number of candidates to return.

    Returns:
        Sorted list of item IDs (descending tag overlap count).
    """
    scored = []
    for item_id, item in catalogue.items():
        item_tags = set(item.get("tags", []))
        overlap   = len(query_tags & item_tags)
        if overlap >= min_overlap:
            scored.append((item_id, overlap))

    # Sort by overlap descending, then by item_id for determinism
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [item_id for item_id, _ in scored[:limit]]


# ── Vector-based (ANN-lite) filtering ────────────────────────────────────────

def candidates_by_vector(
    catalogue: dict,
    query_vector: list[float],
    threshold: float = 0.5,
    limit: int = 50,
) -> list[str]:
    """
    Return item IDs whose embedding cosine-similarity to `query_vector`
    exceeds `threshold`.

    In production this would be an ANN index (FAISS, ScaNN, etc.).
    Here we do a brute-force scan — fine for small catalogues.

    Returns:
        Sorted list of item IDs (descending similarity).
    """
    scored = []
    for item_id, item in catalogue.items():
        vec = item.get("vector")
        if vec is None:
            continue
        sim = cosine_similarity(query_vector, vec)
        if sim >= threshold:
            scored.append((item_id, sim))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [item_id for item_id, _ in scored[:limit]]


# ── History-based filtering ───────────────────────────────────────────────────

def candidates_from_history(
    catalogue: dict,
    liked_item_ids: list[str],
    limit: int = 50,
) -> list[str]:
    """
    Collect candidates that share tags with any item the user has liked.
    Union of tag-based candidates for each liked item.

    Returns:
        Deduplicated list of candidate IDs (excludes already-liked items).
    """
    seen_candidates: set[str] = set()
    already_liked   = set(liked_item_ids)

    for liked_id in liked_item_ids:
        if liked_id not in catalogue:
            continue
        tags = set(catalogue[liked_id].get("tags", []))
        new_candidates = candidates_by_tags(catalogue, tags, min_overlap=1, limit=limit)
        seen_candidates.update(new_candidates)

    # Remove items the user already interacted with
    candidates = [c for c in seen_candidates if c not in already_liked]
    return candidates[:limit]


# ── Popularity-based fallback ─────────────────────────────────────────────────

def candidates_by_popularity(
    catalogue: dict,
    limit: int = 20,
    rating_key: str = "rating",
) -> list[str]:
    """
    Fallback for cold-start: return the top-rated items overall.
    Useful when there's no user history or query vector yet.
    """
    rated = [
        (item_id, item.get(rating_key, 0.0))
        for item_id, item in catalogue.items()
    ]
    rated.sort(key=lambda x: (-x[1], x[0]))
    return [item_id for item_id, _ in rated[:limit]]


# ── Combined candidate generation ─────────────────────────────────────────────

def generate_candidates(
    catalogue: dict,
    query_tags:    set[str]   | None = None,
    query_vector:  list[float]| None = None,
    liked_item_ids: list[str] | None = None,
    limit: int = 50,
) -> list[str]:
    """
    Master function: combine all candidate sources and deduplicate.

    Priority order: vector → tags → history → popularity fallback.
    Returns at most `limit` unique candidate IDs.
    """
    candidates: set[str] = set()

    liked_set = set(liked_item_ids or [])

    if query_vector is not None:
        candidates.update(
            candidates_by_vector(catalogue, query_vector, threshold=0.3, limit=limit)
        )

    if query_tags:
        candidates.update(
            candidates_by_tags(catalogue, query_tags, min_overlap=1, limit=limit)
        )

    if liked_item_ids:
        candidates.update(
            candidates_from_history(catalogue, liked_item_ids, limit=limit)
        )

    # Fall back to popularity if still empty
    if not candidates:
        candidates.update(candidates_by_popularity(catalogue, limit=limit))

    # Always exclude already-liked items
    candidates -= liked_set

    return list(candidates)[:limit]
