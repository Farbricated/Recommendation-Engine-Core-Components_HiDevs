"""
scorer.py
─────────
Scores and ranks candidate items for a given user/query context.
Combines multiple signals (similarity, popularity, recency, diversity)
into a single final score, then returns a ranked list.
"""

import math
from similarity import cosine_similarity, jaccard_similarity


# ── Individual scoring signals ────────────────────────────────────────────────

def score_similarity(
    item: dict,
    query_vector: list[float] | None = None,
    query_tags:   set[str]   | None = None,
) -> float:
    """
    Similarity score in [0, 1].
    Uses cosine similarity if vectors are available; falls back to Jaccard tags.
    """
    if query_vector and item.get("vector"):
        return max(0.0, cosine_similarity(query_vector, item["vector"]))

    if query_tags and item.get("tags"):
        return jaccard_similarity(query_tags, set(item["tags"]))

    return 0.0


def score_popularity(item: dict, max_rating: float = 10.0) -> float:
    """
    Normalised popularity score in [0, 1] based on item rating.
    """
    rating = item.get("rating", 0.0)
    return min(max(rating / max_rating, 0.0), 1.0)


def score_recency(item: dict, current_year: int = 2024, decay: float = 0.05) -> float:
    """
    Recency score in (0, 1] using exponential decay.
    Newer items score closer to 1; very old items approach 0.

    decay: how fast older items fade (higher = faster decay).
    """
    year = item.get("year", current_year)
    age  = max(current_year - year, 0)
    return math.exp(-decay * age)


def score_diversity_penalty(
    item_id: str,
    item: dict,
    already_selected: list[dict],
    penalty: float = 0.3,
) -> float:
    """
    Returns a penalty in [0, 1] if this item is too similar to items already
    in the result list. 1.0 means no penalty; lower values penalise redundancy.

    Uses tag Jaccard similarity against already-selected items.
    """
    if not already_selected:
        return 1.0   # no penalty when list is empty

    item_tags = set(item.get("tags", []))
    max_sim   = max(
        jaccard_similarity(item_tags, set(sel.get("tags", [])))
        for sel in already_selected
    )
    # Convert similarity → diversity weight: high similarity → lower weight
    return 1.0 - (penalty * max_sim)


# ── Combined scoring ──────────────────────────────────────────────────────────

def compute_score(
    item: dict,
    query_vector:     list[float] | None = None,
    query_tags:       set[str]   | None = None,
    weights:          dict       | None = None,
    already_selected: list[dict] | None = None,
) -> float:
    """
    Weighted combination of all scoring signals.

    Default weights:
        similarity  0.50
        popularity  0.25
        recency     0.15
        diversity   0.10

    Args:
        item:             The item dict to score.
        query_vector:     Optional dense query embedding.
        query_tags:       Optional set of query tags.
        weights:          Override default signal weights.
        already_selected: Items already picked (for diversity penalty).

    Returns:
        Float score in [0, 1].
    """
    default_weights = {
        "similarity": 0.50,
        "popularity": 0.25,
        "recency":    0.15,
        "diversity":  0.10,
    }
    w = {**default_weights, **(weights or {})}

    sim_score  = score_similarity(item, query_vector, query_tags)
    pop_score  = score_popularity(item)
    rec_score  = score_recency(item)
    div_factor = score_diversity_penalty(
        item.get("id", ""), item, already_selected or []
    )

    raw = (
        w["similarity"] * sim_score
        + w["popularity"] * pop_score
        + w["recency"]    * rec_score
    )
    # Diversity is a multiplicative penalty so it never inflates the score
    return raw * (w["diversity"] + (1 - w["diversity"]) * div_factor)


# ── Ranking ───────────────────────────────────────────────────────────────────

def rank_candidates(
    catalogue:    dict,
    candidate_ids: list[str],
    query_vector:  list[float] | None = None,
    query_tags:    set[str]   | None = None,
    weights:       dict       | None = None,
    top_k:         int = 10,
    diversity:     bool = True,
) -> list[dict]:
    """
    Score and rank all candidate items, return the top-k results.

    Args:
        catalogue:     Full item catalogue {item_id: item_dict}.
        candidate_ids: Pre-filtered candidate IDs from candidate_gen.
        query_vector:  Optional dense query embedding.
        query_tags:    Optional tag set for the query.
        weights:       Custom signal weights (see compute_score).
        top_k:         Number of results to return.
        diversity:     If True, apply incremental diversity penalty.

    Returns:
        List of item dicts (with 'id' and '_score' injected), sorted best-first.
    """
    scored_items: list[tuple[float, dict]] = []
    selected: list[dict] = []    # grows as we pick items (for diversity)

    # First pass: score all candidates
    all_scored = []
    for item_id in candidate_ids:
        item = catalogue.get(item_id)
        if item is None:
            continue
        item_with_id = {**item, "id": item_id}
        score = compute_score(item_with_id, query_vector, query_tags, weights, [])
        all_scored.append((score, item_with_id))

    # Sort by score descending
    all_scored.sort(key=lambda x: -x[0])

    if not diversity:
        results = []
        for score, item in all_scored[:top_k]:
            results.append({**item, "_score": round(score, 4)})
        return results

    # Second pass: greedy diversity-aware selection
    remaining = list(all_scored)
    while len(selected) < top_k and remaining:
        best_score, best_item = -1.0, None
        for score, item in remaining:
            adjusted = compute_score(
                item, query_vector, query_tags, weights, selected
            )
            if adjusted > best_score:
                best_score = adjusted
                best_item  = item

        if best_item is None:
            break

        selected.append({**best_item, "_score": round(best_score, 4)})
        remaining = [(s, it) for s, it in remaining if it["id"] != best_item["id"]]

    return selected


# ── Utility ───────────────────────────────────────────────────────────────────

def explain_score(
    item: dict,
    query_vector: list[float] | None = None,
    query_tags:   set[str]   | None = None,
) -> dict:
    """
    Return a breakdown of every scoring signal for transparency / debugging.
    """
    return {
        "similarity": round(score_similarity(item, query_vector, query_tags), 4),
        "popularity":  round(score_popularity(item), 4),
        "recency":     round(score_recency(item), 4),
        "final":       round(compute_score(item, query_vector, query_tags), 4),
    }
