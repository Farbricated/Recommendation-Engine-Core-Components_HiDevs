"""
evaluator.py
────────────
Evaluation metrics for recommendation / ranking systems.

Implements:
  - Precision@K
  - Recall@K
  - Average Precision (AP) → Mean Average Precision (MAP)
  - NDCG@K (Normalised Discounted Cumulative Gain)
  - Hit Rate@K
  - Coverage
  - Diversity (intra-list)
  - Full evaluation report
"""

import math
from similarity import jaccard_similarity


# ── Basic retrieval metrics ───────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of the top-k recommendations that are relevant.

    precision@k = |recommended[:k] ∩ relevant| / k
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of all relevant items that appear in top-k recommendations.

    recall@k = |recommended[:k] ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision(recommended: list, relevant: set) -> float:
    """
    Average Precision (AP): area under the precision-recall curve.
    Rewards relevant items appearing earlier in the ranking.
    """
    if not relevant:
        return 0.0

    hits, total_precision = 0, 0.0
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            total_precision += hits / i

    return total_precision / len(relevant)


def mean_average_precision(
    all_recommended: list[list],
    all_relevant:    list[set],
) -> float:
    """
    MAP over multiple users/queries.

    Args:
        all_recommended: List of ranked recommendation lists (one per user).
        all_relevant:    List of relevant-item sets (one per user).
    """
    if not all_recommended:
        return 0.0
    ap_scores = [
        average_precision(rec, rel)
        for rec, rel in zip(all_recommended, all_relevant)
    ]
    return sum(ap_scores) / len(ap_scores)


# ── Discounted Cumulative Gain ────────────────────────────────────────────────

def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Discounted Cumulative Gain @k.
    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    top_k = recommended[:k]
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant
    )


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised DCG@k: DCG divided by the ideal DCG (IDCG).
    Returns a value in [0, 1]; 1.0 = perfect ranking.
    """
    actual_dcg = dcg_at_k(recommended, relevant, k)
    # IDCG: imagine top-min(|relevant|,k) positions are all relevant
    ideal_hits  = min(len(relevant), k)
    ideal_dcg   = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── Hit Rate ──────────────────────────────────────────────────────────────────

def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    1.0 if at least one relevant item appears in top-k, else 0.0.
    (Also called Recall@K binarised.)
    """
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# ── Catalogue-level metrics ───────────────────────────────────────────────────

def coverage(
    all_recommended: list[list],
    catalogue_size:  int,
) -> float:
    """
    Fraction of the catalogue that appears in at least one recommendation list.
    Higher coverage → less popularity bias.
    """
    if catalogue_size == 0:
        return 0.0
    all_items = set(item for rec in all_recommended for item in rec)
    return len(all_items) / catalogue_size


def intra_list_diversity(
    recommended:  list[str],
    catalogue:    dict,
    sim_key:      str = "tags",
) -> float:
    """
    Average pairwise dissimilarity within a single recommendation list.
    1.0 = completely diverse, 0.0 = all items identical.

    Uses tag-Jaccard similarity (1 - similarity = dissimilarity).
    """
    n = len(recommended)
    if n < 2:
        return 0.0

    total_dissim = 0.0
    pairs        = 0
    for i in range(n):
        for j in range(i + 1, n):
            item_a = catalogue.get(recommended[i], {})
            item_b = catalogue.get(recommended[j], {})
            tags_a = set(item_a.get("tags", []))
            tags_b = set(item_b.get("tags", []))
            sim    = jaccard_similarity(tags_a, tags_b)
            total_dissim += (1.0 - sim)
            pairs += 1

    return total_dissim / pairs


# ── Full evaluation report ────────────────────────────────────────────────────

def evaluate(
    recommended:  list[str],
    relevant:     set[str],
    catalogue:    dict,
    k_values:     list[int] = None,
) -> dict:
    """
    Compute a full suite of metrics for a single user/query.

    Returns a dict of metric_name → value.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    report: dict = {}

    for k in k_values:
        report[f"precision@{k}"]  = round(precision_at_k(recommended, relevant, k), 4)
        report[f"recall@{k}"]     = round(recall_at_k(recommended, relevant, k), 4)
        report[f"ndcg@{k}"]       = round(ndcg_at_k(recommended, relevant, k), 4)
        report[f"hit_rate@{k}"]   = round(hit_rate_at_k(recommended, relevant, k), 4)

    report["avg_precision"] = round(average_precision(recommended, relevant), 4)
    report["diversity"]     = round(intra_list_diversity(recommended, catalogue), 4)

    return report


def print_report(report: dict, title: str = "Evaluation Report") -> None:
    """Pretty-print an evaluation report dict."""
    width = 40
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)
    for metric, value in report.items():
        print(f"  {metric:<22} {value:.4f}")
    print("─" * width + "\n")
