"""
scorer.py
─────────
Component 3: RecommendationScorer
Scores candidate items and produces a final ranked list.
Supports pluggable, weighted scoring functions with explanations.
"""

from candidate_gen import ITEM_CATALOGUE, USER_HISTORY


# ── Sample rating data (replaces a database for now) ──────────────────────────

USER_RATINGS: dict[str, dict[str, float]] = {
    "user_A": {"item_1": 4.5, "item_2": 3.0, "item_3": 5.0},
    "user_B": {"item_2": 4.0, "item_3": 4.5, "item_4": 2.0},
    "user_C": {"item_3": 3.5, "item_4": 4.0, "item_5": 4.8},
    "user_D": {"item_1": 5.0, "item_5": 3.0, "item_6": 4.2},
}


class RecommendationScorer:
    """
    Pluggable, weighted scorer for recommendation candidates.

    Usage:
        scorer = RecommendationScorer()
        scorer.add_scorer("popularity", my_popularity_fn, weight=0.4)
        scorer.add_scorer("recency",    my_recency_fn,    weight=0.6)
        ranked = scorer.rank_candidates("user_A", ["item_4", "item_5"], limit=5)
    """

    def __init__(
        self,
        item_catalogue: dict | None = None,
        user_history:   dict | None = None,
        user_ratings:   dict | None = None,
    ):
        self.item_catalogue = item_catalogue or ITEM_CATALOGUE
        self.user_history   = user_history   or USER_HISTORY
        self.user_ratings   = user_ratings   or USER_RATINGS

        # Registry: {name: {"fn": callable, "weight": float}}
        self._scorers: dict[str, dict] = {}

        # Register default built-in scorers
        self.add_scorer("relevance",  self._score_relevance,  weight=0.50)
        self.add_scorer("popularity", self._score_popularity, weight=0.30)
        self.add_scorer("recency",    self._score_recency,    weight=0.20)

    # ── Scorer Registration ───────────────────────────────────────────────

    def add_scorer(self, name: str, function, weight: float) -> None:
        """
        Register a scoring function.

        Args:
            name:     Unique label for this scorer (used in explanations).
            function: Callable(user_id, item_id, context) → float in [0, 1].
            weight:   Relative importance. Weights are normalised automatically.
        """
        if weight < 0:
            raise ValueError(f"Weight for '{name}' must be >= 0.")
        self._scorers[name] = {"fn": function, "weight": weight}

    # ── Built-in Scoring Functions ────────────────────────────────────────

    def _score_relevance(self, user_id: str, item_id: str, context: dict) -> float:
        """
        Tag-overlap relevance: fraction of item's tags matching user's taste profile.
        """
        user_items = self.user_history.get(user_id, set())
        liked_tags: set[str] = set()
        for uid in user_items:
            liked_tags |= self.item_catalogue.get(uid, {}).get("tags", set())

        item_tags = self.item_catalogue.get(item_id, {}).get("tags", set())
        if not item_tags:
            return 0.0
        return len(liked_tags & item_tags) / len(item_tags)

    def _score_popularity(self, user_id: str, item_id: str, context: dict) -> float:
        """Normalised popularity score (catalogue max = 10.0)."""
        raw = self.item_catalogue.get(item_id, {}).get("popularity", 0.0)
        return min(max(raw / 10.0, 0.0), 1.0)

    def _score_recency(self, user_id: str, item_id: str, context: dict) -> float:
        """
        Recency proxy: uses item_id numeric suffix — higher number = newer.
        In a real system this would use release or interaction timestamps.
        """
        try:
            num = int(item_id.split("_")[-1])
            max_num = max(
                int(i.split("_")[-1])
                for i in self.item_catalogue
                if i.split("_")[-1].isdigit()
            )
            return num / max_num if max_num > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.5   # neutral default

    # ── Core Scoring ──────────────────────────────────────────────────────

    def calculate_score(
        self,
        user_id: str,
        item_id: str,
        context: dict | None = None,
    ) -> dict:
        """
        Score a single (user, item) pair using all registered scorers.

        Returns:
            {
                "total":   float,          # weighted combined score in [0, 1]
                "signals": {name: score},  # individual signal breakdown
                "reason":  str,            # human-readable explanation
            }
        """
        if context is None:
            context = {}

        total_weight = sum(s["weight"] for s in self._scorers.values())
        if total_weight == 0:
            return {"total": 0.0, "signals": {}, "reason": "No scorers registered."}

        signals: dict[str, float] = {}
        weighted_sum = 0.0

        for name, scorer in self._scorers.items():
            try:
                raw_score = scorer["fn"](user_id, item_id, context)
            except Exception:
                raw_score = 0.0

            # Clamp each signal to [0, 1]
            clamped = min(max(float(raw_score), 0.0), 1.0)
            signals[name] = round(clamped, 4)
            weighted_sum += clamped * scorer["weight"]

        total = weighted_sum / total_weight
        total = round(min(max(total, 0.0), 1.0), 4)

        # Build a human-readable reason from top signals
        top_signal = max(signals, key=signals.get)
        reason = (
            f"Recommended because of high {top_signal} "
            f"({signals[top_signal]:.0%}). "
            f"Overall score: {total:.0%}."
        )

        return {"total": total, "signals": signals, "reason": reason}

    # ── Ranking ───────────────────────────────────────────────────────────

    def rank_candidates(
        self,
        user_id:    str,
        candidates: list[str],
        limit:      int = 10,
        context:    dict | None = None,
    ) -> list[dict]:
        """
        Score all candidates and return top `limit` results, sorted best-first.

        Args:
            user_id:    Target user.
            candidates: List of item IDs to evaluate.
            limit:      Maximum results to return.
            context:    Optional extra context passed to scorer functions.

        Returns:
            List of dicts: [{"item_id", "total", "signals", "reason"}, ...]
        """
        if context is None:
            context = {}

        results = []
        for item_id in candidates:
            score_info = self.calculate_score(user_id, item_id, context)
            results.append({"item_id": item_id, **score_info})

        results.sort(key=lambda r: -r["total"])
        return results[:limit]