"""
test.py
───────
Test suite for all four class-based components.
Run with:   python test.py
No external dependencies required.
"""

import sys
import math

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS, FAIL = 0, 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  ✅  {name}")
        PASS += 1
    else:
        print(f"  ❌  {name}" + (f"  →  {detail}" if detail else ""))
        FAIL += 1

def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) < tol

def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

# ── Sample data shared across tests ──────────────────────────────────────────

CATALOGUE = {
    "item_1": {"tags": {"sci-fi", "action"},    "category": "movie", "popularity": 9.2},
    "item_2": {"tags": {"sci-fi", "drama"},     "category": "movie", "popularity": 8.5},
    "item_3": {"tags": {"sci-fi", "thriller"},  "category": "movie", "popularity": 8.8},
    "item_4": {"tags": {"drama", "romance"},    "category": "movie", "popularity": 7.9},
    "item_5": {"tags": {"action", "comedy"},    "category": "movie", "popularity": 8.1},
    "item_6": {"tags": {"animation", "family"}, "category": "movie", "popularity": 8.3},
    "item_7": {"tags": {"sci-fi", "comedy"},    "category": "movie", "popularity": 7.5},
    "item_8": {"tags": {"thriller", "horror"},  "category": "movie", "popularity": 7.2},
}

HISTORY = {
    "user_A": {"item_1", "item_2", "item_3"},
    "user_B": {"item_2", "item_3", "item_4"},
    "user_C": {"item_3", "item_4", "item_5"},
    "user_D": {"item_1", "item_5", "item_6"},
    "new_user": set(),   # cold-start
}

# ══════════════════════════════════════════════════════
#  Component 1 — SimilarityCalculator
# ══════════════════════════════════════════════════════

def test_similarity():
    section("Component 1: SimilarityCalculator")
    from similarity import SimilarityCalculator
    sc = SimilarityCalculator()

    # ── cosine_similarity ─────────────────────────────────────────────────
    check("cosine: identical vectors = 1.0",
          close(sc.cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0))

    check("cosine: orthogonal vectors = 0.0",
          close(sc.cosine_similarity([1, 0], [0, 1]), 0.0))

    check("cosine: opposite vectors = -1.0",
          close(sc.cosine_similarity([1, 0], [-1, 0]), -1.0))

    check("cosine: zero vector = 0.0",
          close(sc.cosine_similarity([0, 0], [1, 2]), 0.0))

    check("cosine: [1,1] vs [1,0] ≈ 0.707",
          close(sc.cosine_similarity([1, 1], [1, 0]), 1 / math.sqrt(2)))

    check("cosine: result in [-1, 1]",
          -1.0 <= sc.cosine_similarity([3, 4], [1, 2]) <= 1.0)

    # ── jaccard_similarity ────────────────────────────────────────────────
    check("jaccard: identical sets = 1.0",
          close(sc.jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0))

    check("jaccard: disjoint sets = 0.0",
          close(sc.jaccard_similarity({"a"}, {"b"}), 0.0))

    check("jaccard: partial overlap = 1/3",
          close(sc.jaccard_similarity({"a", "b"}, {"b", "c"}), 1/3))

    check("jaccard: both empty sets = 0.0",
          close(sc.jaccard_similarity(set(), set()), 0.0))

    check("jaccard: list inputs work too",
          close(sc.jaccard_similarity(["x", "y"], ["y", "z"]), 1/3))

    # ── pearson_correlation ───────────────────────────────────────────────
    r1 = {"movie_1": 5.0, "movie_2": 4.0, "movie_3": 3.0}
    r2 = {"movie_1": 5.0, "movie_2": 4.0, "movie_3": 3.0}
    check("pearson: identical ratings = 1.0",
          close(sc.pearson_correlation(r1, r2), 1.0))

    r3 = {"movie_1": 1.0, "movie_2": 2.0, "movie_3": 3.0}
    check("pearson: reversed ratings = -1.0",
          close(sc.pearson_correlation(r1, r3), -1.0))

    r4 = {"movie_X": 5.0, "movie_Y": 3.0}   # no overlap with r1
    check("pearson: no common items = 0.0",
          close(sc.pearson_correlation(r1, r4), 0.0))

    check("pearson: one common item = 0.0 (not enough)",
          close(sc.pearson_correlation({"a": 4.0}, {"a": 4.0}), 0.0))

    check("pearson: result in [-1, 1]",
          -1.0 <= sc.pearson_correlation(r1, r3) <= 1.0)

    r5 = {"m1": 4.0, "m2": 5.0, "m3": 2.0}
    r6 = {"m1": 3.5, "m2": 4.5, "m3": 1.5}
    check("pearson: similar patterns → high positive",
          sc.pearson_correlation(r5, r6) > 0.9)

    # ── most_similar ──────────────────────────────────────────────────────
    vectors = {
        "A": [1.0, 0.0],
        "B": [0.9, 0.1],
        "C": [0.0, 1.0],
    }
    results = sc.most_similar("A", vectors, top_k=2)
    check("most_similar: returns top-k neighbours",
          len(results) == 2)
    check("most_similar: B is closest to A",
          results[0][0] == "B")
    check("most_similar: unknown ID returns empty",
          sc.most_similar("Z", vectors) == [])


# ══════════════════════════════════════════════════════
#  Component 2 — CandidateGenerator
# ══════════════════════════════════════════════════════

def test_candidate_gen():
    section("Component 2: CandidateGenerator")
    from candidate_gen import CandidateGenerator
    gen = CandidateGenerator(user_history=HISTORY, item_catalogue=CATALOGUE, limit=20)

    # ── collaborative_candidates ──────────────────────────────────────────
    collab = gen.collaborative_candidates("user_A")
    check("collaborative: returns a list",
          isinstance(collab, list))
    check("collaborative: excludes already-liked items",
          all(item not in HISTORY["user_A"] for item in collab))
    check("collaborative: within limit",
          len(collab) <= 20)

    collab_new = gen.collaborative_candidates("new_user")
    check("collaborative: cold-start fallback returns items",
          len(collab_new) > 0)

    # ── content_based_candidates ──────────────────────────────────────────
    content = gen.content_based_candidates("user_A")
    check("content-based: returns a list",
          isinstance(content, list))
    check("content-based: excludes already-liked items",
          all(item not in HISTORY["user_A"] for item in content))
    check("content-based: within limit",
          len(content) <= 20)

    # user_A likes sci-fi items → item_7 (sci-fi) should be in candidates
    check("content-based: finds sci-fi matches for user_A",
          "item_7" in content)

    content_new = gen.content_based_candidates("new_user")
    check("content-based: cold-start fallback returns items",
          len(content_new) > 0)

    # ── popularity_candidates ─────────────────────────────────────────────
    popular = gen.popularity_candidates()
    check("popularity: returns a list",
          isinstance(popular, list))
    check("popularity: within limit",
          len(popular) <= 20)
    check("popularity: item_1 (9.2) ranked first",
          popular[0] == "item_1")
    check("popularity: item_8 (7.2) ranked last",
          popular[-1] == "item_8")

    # ── hybrid_candidates ─────────────────────────────────────────────────
    hybrid = gen.hybrid_candidates("user_A")
    check("hybrid: returns a list",
          isinstance(hybrid, list))
    check("hybrid: within limit",
          len(hybrid) <= 20)
    check("hybrid: excludes already-liked items for user_A",
          all(item not in HISTORY["user_A"] for item in hybrid))

    hybrid_new = gen.hybrid_candidates("new_user")
    check("hybrid: cold-start returns items",
          len(hybrid_new) > 0)


# ══════════════════════════════════════════════════════
#  Component 3 — RecommendationScorer
# ══════════════════════════════════════════════════════

def test_scorer():
    section("Component 3: RecommendationScorer")
    from scorer import RecommendationScorer
    scorer = RecommendationScorer(item_catalogue=CATALOGUE, user_history=HISTORY)

    # ── add_scorer ────────────────────────────────────────────────────────
    scorer.add_scorer("custom", lambda u, i, c: 0.5, weight=0.1)
    check("add_scorer: custom scorer registered",
          "custom" in scorer._scorers)

    try:
        scorer.add_scorer("bad", lambda u, i, c: 0.0, weight=-1)
        check("add_scorer: rejects negative weight", False)
    except ValueError:
        check("add_scorer: rejects negative weight", True)

    # ── calculate_score ───────────────────────────────────────────────────
    result = scorer.calculate_score("user_A", "item_7")
    check("calculate_score: returns dict",
          isinstance(result, dict))
    check("calculate_score: has 'total' key",
          "total" in result)
    check("calculate_score: has 'signals' key",
          "signals" in result)
    check("calculate_score: has 'reason' key",
          "reason" in result)
    check("calculate_score: total in [0, 1]",
          0.0 <= result["total"] <= 1.0)
    check("calculate_score: signals dict is non-empty",
          len(result["signals"]) > 0)
    check("calculate_score: all signal values in [0, 1]",
          all(0.0 <= v <= 1.0 for v in result["signals"].values()))
    check("calculate_score: reason is a non-empty string",
          isinstance(result["reason"], str) and len(result["reason"]) > 0)

    # user_A is a sci-fi fan; item_7 is sci-fi → should score well
    sci_fi_score  = scorer.calculate_score("user_A", "item_7")["total"]
    romance_score = scorer.calculate_score("user_A", "item_4")["total"]
    check("calculate_score: sci-fi scores higher than romance for sci-fi user",
          sci_fi_score > romance_score)

    # ── rank_candidates ───────────────────────────────────────────────────
    candidates = ["item_4", "item_5", "item_6", "item_7", "item_8"]
    ranked = scorer.rank_candidates("user_A", candidates, limit=3)
    check("rank_candidates: returns list",
          isinstance(ranked, list))
    check("rank_candidates: respects limit",
          len(ranked) <= 3)
    check("rank_candidates: each result has item_id",
          all("item_id" in r for r in ranked))
    check("rank_candidates: sorted descending by total",
          all(ranked[i]["total"] >= ranked[i+1]["total"] for i in range(len(ranked)-1)))

    ranked_all = scorer.rank_candidates("user_A", candidates, limit=100)
    check("rank_candidates: cannot exceed candidate count",
          len(ranked_all) <= len(candidates))

    # Cold-start user
    ranked_new = scorer.rank_candidates("new_user", candidates, limit=5)
    check("rank_candidates: works for cold-start user",
          isinstance(ranked_new, list))


# ══════════════════════════════════════════════════════
#  Component 4 — RecommendationEvaluator
# ══════════════════════════════════════════════════════

def test_evaluator():
    section("Component 4: RecommendationEvaluator")
    from evaluator import RecommendationEvaluator
    ev = RecommendationEvaluator()

    recs     = ["item_1", "item_2", "item_3", "item_4", "item_5"]
    relevant = {"item_1", "item_3", "item_5"}   # positions 1, 3, 5

    # ── precision_at_k ────────────────────────────────────────────────────
    check("precision@1 = 1.0  (item_1 is relevant)",
          close(ev.precision_at_k(recs, relevant, 1), 1.0))
    check("precision@2 = 0.5  (1 of 2 relevant)",
          close(ev.precision_at_k(recs, relevant, 2), 0.5))
    check("precision@5 = 0.6  (3 of 5 relevant)",
          close(ev.precision_at_k(recs, relevant, 5), 0.6))
    check("precision@k: k=0 returns 0.0",
          close(ev.precision_at_k(recs, relevant, 0), 0.0))
    check("precision@k: empty recs returns 0.0",
          close(ev.precision_at_k([], relevant, 5), 0.0))
    check("precision@k: empty relevant = 0.0",
          close(ev.precision_at_k(recs, set(), 5), 0.0))

    # ── recall_at_k ───────────────────────────────────────────────────────
    check("recall@1 = 0.333  (1 of 3 found)",
          close(ev.recall_at_k(recs, relevant, 1), 1/3))
    check("recall@5 = 1.0   (all 3 found)",
          close(ev.recall_at_k(recs, relevant, 5), 1.0))
    check("recall@k: empty relevant returns 0.0",
          close(ev.recall_at_k(recs, set(), 5), 0.0))
    check("recall@k: empty recs returns 0.0",
          close(ev.recall_at_k([], relevant, 5), 0.0))

    # ── ndcg_at_k ─────────────────────────────────────────────────────────
    check("ndcg@5: value in [0, 1]",
          0.0 <= ev.ndcg_at_k(recs, relevant, 5) <= 1.0)
    check("ndcg@5: imperfect ranking < 1.0",
          ev.ndcg_at_k(recs, relevant, 5) < 1.0)

    perfect_recs = ["item_1", "item_3", "item_5", "item_2", "item_4"]
    check("ndcg@3: perfect ranking = 1.0",
          close(ev.ndcg_at_k(perfect_recs, relevant, 3), 1.0))

    check("ndcg@k: empty relevant = 0.0",
          close(ev.ndcg_at_k(recs, set(), 5), 0.0))
    check("ndcg@k: k=0 returns 0.0",
          close(ev.ndcg_at_k(recs, relevant, 0), 0.0))

    # ── evaluate_all ──────────────────────────────────────────────────────
    recs_dict = {
        "user_A": ["item_4", "item_5", "item_6", "item_7"],
        "user_B": ["item_1", "item_5", "item_6"],
        "user_C": ["item_1", "item_2"],           # no ground truth → skipped
    }
    ground_truth = {
        "user_A": {"item_5", "item_6"},
        "user_B": {"item_1"},
    }

    report = ev.evaluate_all(recs_dict, ground_truth, k=3)
    check("evaluate_all: returns dict",
          isinstance(report, dict))
    check("evaluate_all: has precision@3 key",
          "precision@3" in report)
    check("evaluate_all: has recall@3 key",
          "recall@3" in report)
    check("evaluate_all: has ndcg@3 key",
          "ndcg@3" in report)
    check("evaluate_all: has num_users key",
          "num_users" in report)
    check("evaluate_all: has num_skipped key",
          "num_skipped" in report)
    check("evaluate_all: 2 users evaluated",
          report["num_users"] == 2)
    check("evaluate_all: 1 user skipped (no ground truth)",
          report["num_skipped"] == 1)
    check("evaluate_all: precision in [0, 1]",
          0.0 <= report["precision@3"] <= 1.0)
    check("evaluate_all: empty dicts return 0 users",
          ev.evaluate_all({}, {}, k=5)["num_users"] == 0)

    ev.print_report(report, "Sample Evaluation Report")


# ══════════════════════════════════════════════════════
#  End-to-End Integration
# ══════════════════════════════════════════════════════

def test_end_to_end():
    section("End-to-End Integration")
    from candidate_gen import CandidateGenerator
    from scorer        import RecommendationScorer
    from evaluator     import RecommendationEvaluator

    gen    = CandidateGenerator(user_history=HISTORY, item_catalogue=CATALOGUE)
    scorer = RecommendationScorer(item_catalogue=CATALOGUE, user_history=HISTORY)
    ev     = RecommendationEvaluator()

    # Step 1: generate candidates for user_A
    candidates = gen.hybrid_candidates("user_A")
    check("E2E: candidates generated", len(candidates) > 0)

    # Step 2: rank them
    ranked = scorer.rank_candidates("user_A", candidates, limit=5)
    check("E2E: ranked list ≤ 5 items", len(ranked) <= 5)

    recommended_ids = [r["item_id"] for r in ranked]
    check("E2E: ranked items have scores", all("total" in r for r in ranked))
    check("E2E: ranked items have reasons", all("reason" in r for r in ranked))

    # Step 3: evaluate
    ground_truth = {"user_A": {"item_4", "item_5", "item_6", "item_7"}}
    recs_dict    = {"user_A": recommended_ids}
    report       = ev.evaluate_all(recs_dict, ground_truth, k=5)

    check("E2E: evaluation report produced", len(report) > 0)
    check("E2E: precision in [0, 1]", 0.0 <= report["precision@5"] <= 1.0)

    print("\n  Top 5 recommendations for user_A:")
    for i, r in enumerate(ranked, 1):
        print(f"    {i}. {r['item_id']}  score={r['total']:.3f}  | {r['reason']}")

    ev.print_report(report, "End-to-End: user_A Evaluation")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  Day 29 Project — Test Suite")
    print("═" * 50)

    test_similarity()
    test_candidate_gen()
    test_scorer()
    test_evaluator()
    test_end_to_end()

    total = PASS + FAIL
    print(f"\n{'═'*50}")
    print(f"  Results: {PASS}/{total} passed", end="")
    if FAIL == 0:
        print("  🎉 All tests passed!")
    else:
        print(f"  ⚠️  {FAIL} test(s) failed.")
    print("═" * 50 + "\n")

    sys.exit(0 if FAIL == 0 else 1)