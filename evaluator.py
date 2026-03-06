"""
evaluator.py
────────────
Component 4: RecommendationEvaluator
Measures how good the recommendations actually are.
Implements Precision@K, Recall@K, NDCG@K, and an aggregate evaluate_all().
"""

import math


class RecommendationEvaluator:
    """
    Offline evaluation of recommendation quality.

    All methods accept:
        recommendations: ordered list of recommended item IDs
        relevant_items:  set (or list) of actually relevant item IDs
        k:               cutoff position to evaluate at

    Metrics returned are floats in [0, 1]; higher = better.
    """

    # ── Precision@K ──────────────────────────────────────────────────────

    def precision_at_k(
        self,
        recommendations: list[str],
        relevant_items:  set | list,
        k: int,
    ) -> float:
        """
        % of the top-k recommendations that are relevant.

        precision@k = |recommended[:k] ∩ relevant| / k

        Handles empty lists and k=0 gracefully (returns 0.0).
        """
        if k <= 0 or not recommendations:
            return 0.0

        relevant = set(relevant_items)
        top_k    = recommendations[:k]
        hits     = sum(1 for item in top_k if item in relevant)
        return hits / k

    # ── Recall@K ─────────────────────────────────────────────────────────

    def recall_at_k(
        self,
        recommendations: list[str],
        relevant_items:  set | list,
        k: int,
    ) -> float:
        """
        % of all relevant items that appear in the top-k recommendations.

        recall@k = |recommended[:k] ∩ relevant| / |relevant|

        Returns 0.0 if relevant_items is empty.
        """
        relevant = set(relevant_items)
        if not relevant:
            return 0.0
        if k <= 0 or not recommendations:
            return 0.0

        top_k = recommendations[:k]
        hits  = sum(1 for item in top_k if item in relevant)
        return hits / len(relevant)

    # ── NDCG@K ───────────────────────────────────────────────────────────

    def ndcg_at_k(
        self,
        recommendations: list[str],
        relevant_items:  set | list,
        k: int,
    ) -> float:
        """
        Normalised Discounted Cumulative Gain @ k.
        Rewards relevant items appearing earlier in the list.
        Returns value in [0, 1]; 1.0 = perfect ranking.

        Uses binary relevance (relevant=1, not-relevant=0).
        """
        relevant = set(relevant_items)
        if not relevant or k <= 0:
            return 0.0

        def dcg(ranked_list: list[str]) -> float:
            return sum(
                1.0 / math.log2(rank + 1)
                for rank, item in enumerate(ranked_list[:k], start=1)
                if item in relevant
            )

        actual_dcg = dcg(recommendations)

        # Ideal DCG: all relevant items at the top positions
        ideal_list = list(relevant) + [None] * k   # pad so slicing is safe
        ideal_dcg  = dcg(list(relevant)[:k])

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # ── Aggregate Evaluation ──────────────────────────────────────────────

    def evaluate_all(
        self,
        recommendations_dict: dict[str, list[str]],
        ground_truth_dict:    dict[str, set | list],
        k: int = 10,
    ) -> dict[str, float]:
        """
        Average all metrics across multiple users / queries.

        Args:
            recommendations_dict: {user_id: [ranked item IDs]}
            ground_truth_dict:    {user_id: {relevant item IDs}}
            k:                    Cutoff position for all metrics.

        Returns:
            {
                "precision@k":  float,
                "recall@k":     float,
                "ndcg@k":       float,
                "num_users":    int,   # how many users were evaluated
                "num_skipped":  int,   # users with missing ground truth
            }
        """
        precision_scores, recall_scores, ndcg_scores = [], [], []
        skipped = 0

        for user_id, recs in recommendations_dict.items():
            if user_id not in ground_truth_dict:
                skipped += 1
                continue

            relevant = ground_truth_dict[user_id]

            precision_scores.append(self.precision_at_k(recs, relevant, k))
            recall_scores.append(   self.recall_at_k(   recs, relevant, k))
            ndcg_scores.append(     self.ndcg_at_k(     recs, relevant, k))

        n = len(precision_scores)

        def avg(lst: list[float]) -> float:
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        return {
            f"precision@{k}": avg(precision_scores),
            f"recall@{k}":    avg(recall_scores),
            f"ndcg@{k}":      avg(ndcg_scores),
            "num_users":      n,
            "num_skipped":    skipped,
        }

    # ── Pretty Printer ────────────────────────────────────────────────────

    def print_report(self, metrics: dict, title: str = "Evaluation Report") -> None:
        """Pretty-print a metrics dictionary."""
        width = 42
        print("\n" + "─" * width)
        print(f"  {title}")
        print("─" * width)
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:<24} {val:.4f}")
            else:
                print(f"  {key:<24} {val}")
        print("─" * width + "\n")