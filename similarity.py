"""
similarity.py
─────────────
Component 1: SimilarityCalculator
Measures how similar two users, items, or skill sets are.
Foundation of all recommendation logic.
"""

import math


class SimilarityCalculator:
    """
    Calculates similarity between users or items using three metrics:
      - Cosine Similarity  : compare dense vectors (user/item embeddings)
      - Jaccard Similarity : compare sets (skills, tags, genres)
      - Pearson Correlation: compare rating patterns
    """

    # ── Cosine Similarity ─────────────────────────────────────────────────

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Cosine similarity between two dense vectors.
        Returns value in [-1, 1]. Higher = more similar.
        Returns 0.0 for zero vectors.

        Use case: 'Users/items with similar feature vectors.'
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must be the same length.")

        dot   = sum(a * b for a, b in zip(vec1, vec2))
        mag1  = math.sqrt(sum(a ** 2 for a in vec1))
        mag2  = math.sqrt(sum(b ** 2 for b in vec2))

        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0

        return dot / (mag1 * mag2)

    # ── Jaccard Similarity ────────────────────────────────────────────────

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """
        Jaccard similarity: |A ∩ B| / |A ∪ B|.
        Returns value in [0, 1]. Returns 0.0 for two empty sets.

        Use case: 'Users with overlapping skill sets or genre preferences.'
        """
        set1, set2 = set(set1), set(set2)

        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union        = len(set1 | set2)
        return intersection / union

    # ── Pearson Correlation ───────────────────────────────────────────────

    def pearson_correlation(
        self,
        ratings1: dict[str, float],
        ratings2: dict[str, float],
    ) -> float:
        """
        Pearson correlation between two users' rating dictionaries.
        Only considers items rated by BOTH users (co-rated items).
        Returns value in [-1, 1]. Returns 0.0 if < 2 co-rated items.

        Use case: 'Users who rate items the same way have similar taste.'

        Args:
            ratings1: {item_id: rating}  e.g. {"movie_1": 4.5, "movie_2": 3.0}
            ratings2: {item_id: rating}
        """
        # Find commonly rated items
        common = set(ratings1.keys()) & set(ratings2.keys())

        if len(common) < 2:
            return 0.0   # Not enough data to correlate

        n    = len(common)
        r1   = [ratings1[i] for i in common]
        r2   = [ratings2[i] for i in common]

        mean1 = sum(r1) / n
        mean2 = sum(r2) / n

        # Deviations from mean
        d1 = [x - mean1 for x in r1]
        d2 = [x - mean2 for x in r2]

        numerator   = sum(a * b for a, b in zip(d1, d2))
        denominator = math.sqrt(sum(a**2 for a in d1)) * math.sqrt(sum(b**2 for b in d2))

        if denominator == 0.0:
            return 0.0

        # Clamp to [-1, 1] to handle floating point edge cases
        return max(-1.0, min(1.0, numerator / denominator))

    # ── Convenience wrapper ───────────────────────────────────────────────

    def most_similar(
        self,
        target_id: str,
        all_vectors: dict[str, list[float]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find the top-k most similar items/users to target_id using cosine similarity.

        Args:
            target_id:   ID of the item/user to compare against.
            all_vectors: {id: vector} for every item/user.
            top_k:       How many neighbours to return.

        Returns:
            List of (id, similarity_score) sorted descending.
        """
        if target_id not in all_vectors:
            return []

        target_vec = all_vectors[target_id]
        scores = [
            (other_id, self.cosine_similarity(target_vec, vec))
            for other_id, vec in all_vectors.items()
            if other_id != target_id
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]