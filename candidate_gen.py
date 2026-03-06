"""
candidate_gen.py
────────────────
Component 2: CandidateGenerator
Generates a pool of candidate items to evaluate before scoring.
Uses three strategies: collaborative, content-based, and popularity.
"""


# ── Sample in-memory data store (replaces a database for now) ─────────────────

# User interaction history: {user_id: set of item_ids they liked}
USER_HISTORY: dict[str, set[str]] = {
    "user_A": {"item_1", "item_2", "item_3"},
    "user_B": {"item_2", "item_3", "item_4"},
    "user_C": {"item_3", "item_4", "item_5"},
    "user_D": {"item_1", "item_5", "item_6"},
}

# Item metadata: {item_id: {tags, category, popularity_score}}
ITEM_CATALOGUE: dict[str, dict] = {
    "item_1": {"tags": {"sci-fi", "action"},   "category": "movie", "popularity": 9.2},
    "item_2": {"tags": {"sci-fi", "drama"},    "category": "movie", "popularity": 8.5},
    "item_3": {"tags": {"sci-fi", "thriller"}, "category": "movie", "popularity": 8.8},
    "item_4": {"tags": {"drama", "romance"},   "category": "movie", "popularity": 7.9},
    "item_5": {"tags": {"action", "comedy"},   "category": "movie", "popularity": 8.1},
    "item_6": {"tags": {"animation", "family"},"category": "movie", "popularity": 8.3},
    "item_7": {"tags": {"sci-fi", "comedy"},   "category": "movie", "popularity": 7.5},
    "item_8": {"tags": {"thriller", "horror"}, "category": "movie", "popularity": 7.2},
}


class CandidateGenerator:
    """
    Generates candidate item pools using three strategies:
      1. Collaborative : items liked by users similar to the target user
      2. Content-based : items sharing tags with the user's history
      3. Popularity    : top-rated items overall (cold-start fallback)
      4. Hybrid        : union of all three strategies
    """

    def __init__(
        self,
        user_history:   dict[str, set[str]] | None = None,
        item_catalogue: dict[str, dict]     | None = None,
        limit: int = 20,
    ):
        self.user_history   = user_history   or USER_HISTORY
        self.item_catalogue = item_catalogue or ITEM_CATALOGUE
        self.limit          = limit

    # ── Helper ────────────────────────────────────────────────────────────

    def _user_similarity(self, user_a: str, user_b: str) -> float:
        """Jaccard similarity between two users' liked-item sets."""
        hist_a = self.user_history.get(user_a, set())
        hist_b = self.user_history.get(user_b, set())
        if not hist_a and not hist_b:
            return 0.0
        return len(hist_a & hist_b) / len(hist_a | hist_b)

    # ── Strategy 1: Collaborative Filtering ──────────────────────────────

    def collaborative_candidates(self, user_id: str) -> list[str]:
        """
        Items liked by users who are similar to user_id.
        Excludes items the user has already seen.
        Falls back to popularity candidates for cold-start users.

        Returns: list of item IDs (up to self.limit)
        """
        user_items = self.user_history.get(user_id, set())

        # Cold-start: no history → fall back to popularity
        if not user_items:
            return self.popularity_candidates()

        # Find similar users (anyone who shares at least 1 liked item)
        similar_users = [
            other_id
            for other_id in self.user_history
            if other_id != user_id and self._user_similarity(user_id, other_id) > 0
        ]

        # Gather items from similar users, weighted by similarity
        item_votes: dict[str, float] = {}
        for other_id in similar_users:
            sim   = self._user_similarity(user_id, other_id)
            items = self.user_history[other_id] - user_items   # exclude seen
            for item in items:
                item_votes[item] = item_votes.get(item, 0.0) + sim

        # Sort by vote weight descending
        sorted_items = sorted(item_votes, key=lambda i: -item_votes[i])
        return sorted_items[: self.limit]

    # ── Strategy 2: Content-Based Filtering ──────────────────────────────

    def content_based_candidates(self, user_id: str) -> list[str]:
        """
        Items that share tags with items the user has previously liked.
        Excludes items the user has already seen.
        Falls back to popularity candidates for cold-start users.

        Returns: list of item IDs (up to self.limit)
        """
        user_items = self.user_history.get(user_id, set())

        # Cold-start fallback
        if not user_items:
            return self.popularity_candidates()

        # Build the user's 'taste profile' as a union of all liked-item tags
        liked_tags: set[str] = set()
        for item_id in user_items:
            liked_tags |= self.item_catalogue.get(item_id, {}).get("tags", set())

        # Score unseen items by tag overlap with the taste profile
        item_scores: dict[str, int] = {}
        for item_id, meta in self.item_catalogue.items():
            if item_id in user_items:
                continue   # already seen
            overlap = len(liked_tags & meta.get("tags", set()))
            if overlap > 0:
                item_scores[item_id] = overlap

        sorted_items = sorted(item_scores, key=lambda i: -item_scores[i])
        return sorted_items[: self.limit]

    # ── Strategy 3: Popularity (cold-start fallback) ──────────────────────

    def popularity_candidates(self) -> list[str]:
        """
        Top items by overall popularity score.
        Used when there is no user history available.

        Returns: list of item IDs (up to self.limit)
        """
        sorted_items = sorted(
            self.item_catalogue,
            key=lambda i: -self.item_catalogue[i].get("popularity", 0.0),
        )
        return sorted_items[: self.limit]

    # ── Strategy 4: Hybrid ────────────────────────────────────────────────

    def hybrid_candidates(self, user_id: str) -> list[str]:
        """
        Combines collaborative, content-based, and popularity candidates.
        Deduplicates and returns up to self.limit items.
        Gives priority to items appearing in multiple strategies.

        Returns: list of item IDs (up to self.limit)
        """
        collab   = set(self.collaborative_candidates(user_id))
        content  = set(self.content_based_candidates(user_id))
        popular  = set(self.popularity_candidates())

        # Remove items the user already liked
        user_items = self.user_history.get(user_id, set())

        # Items appearing in more strategies rank first
        vote_count: dict[str, int] = {}
        for item in (collab | content | popular) - user_items:
            vote_count[item] = (
                (1 if item in collab  else 0)
                + (1 if item in content else 0)
                + (1 if item in popular else 0)
            )

        sorted_items = sorted(vote_count, key=lambda i: -vote_count[i])
        return sorted_items[: self.limit]