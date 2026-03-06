"""
test.py
───────
Test suite for all four modules.
Run with:   python test.py
All tests use simple, hand-verifiable data — no external dependencies.
"""

import math
import sys

# ── Shared sample data ────────────────────────────────────────────────────────

CATALOGUE = {
    "item_001": {
        "title":  "The Matrix",
        "tags":   {"sci-fi", "action", "cyberpunk"},
        "vector": [0.9, 0.1, 0.8, 0.2],
        "rating": 8.7,
        "year":   1999,
    },
    "item_002": {
        "title":  "Blade Runner 2049",
        "tags":   {"sci-fi", "drama", "cyberpunk"},
        "vector": [0.8, 0.2, 0.7, 0.3],
        "rating": 8.0,
        "year":   2017,
    },
    "item_003": {
        "title":  "Toy Story",
        "tags":   {"animation", "family", "comedy"},
        "vector": [0.1, 0.9, 0.1, 0.8],
        "rating": 8.3,
        "year":   1995,
    },
    "item_004": {
        "title":  "Inception",
        "tags":   {"sci-fi", "thriller", "action"},
        "vector": [0.7, 0.3, 0.6, 0.4],
        "rating": 8.8,
        "year":   2010,
    },
    "item_005": {
        "title":  "Finding Nemo",
        "tags":   {"animation", "family", "adventure"},
        "vector": [0.2, 0.8, 0.2, 0.7],
        "rating": 8.1,
        "year":   2003,
    },
}


# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  ✅  {name}")
        PASS += 1
    else:
        print(f"  ❌  {name}" + (f"  →  {detail}" if detail else ""))
        FAIL += 1

def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol

def section(title: str) -> None:
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════
#  MODULE 1 — similarity.py
# ══════════════════════════════════════════════════════

def test_similarity():
    section("similarity.py")
    from similarity import (
        dot_product,
        magnitude,
        cosine_similarity,
        cosine_similarity_sparse,
        jaccard_similarity,
        text_overlap_similarity,
        compute_similarity,
    )

    # dot_product
    check("dot_product basic",
          close(dot_product([1, 2, 3], [4, 5, 6]), 32.0))

    check("dot_product zeros",
          close(dot_product([0, 0], [1, 2]), 0.0))

    # magnitude
    check("magnitude [3, 4] = 5",
          close(magnitude([3.0, 4.0]), 5.0))

    check("magnitude zero vector",
          close(magnitude([0.0, 0.0]), 0.0))

    # cosine_similarity
    check("cosine_similarity identical vectors = 1",
          close(cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0))

    check("cosine_similarity orthogonal vectors = 0",
          close(cosine_similarity([1, 0], [0, 1]), 0.0))

    check("cosine_similarity opposite vectors = -1",
          close(cosine_similarity([1, 0], [-1, 0]), -1.0))

    check("cosine_similarity zero vector = 0",
          close(cosine_similarity([0, 0], [1, 2]), 0.0))

    sim = cosine_similarity([1, 1], [1, 0])
    check("cosine_similarity [1,1] vs [1,0] ≈ 0.707",
          close(sim, 1 / math.sqrt(2), tol=1e-5))

    # cosine_similarity_sparse
    sim_sparse = cosine_similarity_sparse({"a": 1.0, "b": 1.0}, {"a": 1.0})
    check("cosine_similarity_sparse partial overlap ≈ 0.707",
          close(sim_sparse, 1 / math.sqrt(2), tol=1e-5))

    check("cosine_similarity_sparse no overlap = 0",
          close(cosine_similarity_sparse({"a": 1}, {"b": 1}), 0.0))

    # jaccard_similarity
    check("jaccard identical sets = 1",
          close(jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0))

    check("jaccard disjoint sets = 0",
          close(jaccard_similarity({"a"}, {"b"}), 0.0))

    check("jaccard partial overlap",
          close(jaccard_similarity({"a", "b"}, {"b", "c"}), 1/3, tol=1e-5))

    check("jaccard both empty = 0",
          close(jaccard_similarity(set(), set()), 0.0))

    # text_overlap_similarity
    check("text_overlap identical sentences = 1",
          close(text_overlap_similarity("hello world", "hello world"), 1.0))

    check("text_overlap no common words = 0",
          close(text_overlap_similarity("cat dog", "fish bird"), 0.0))

    tov = text_overlap_similarity("the quick brown fox", "the slow brown dog")
    check("text_overlap partial (the, brown) = 2/6 = 0.333",
          close(tov, 2/6, tol=1e-5))

    # compute_similarity (high-level)
    item_a = {"vector": [1.0, 0.0], "tags": {"sci-fi", "action"}, "title": "Movie A"}
    item_b = {"vector": [1.0, 0.0], "tags": {"sci-fi", "drama"},  "title": "Movie A"}

    check("compute_similarity cosine identical = 1",
          close(compute_similarity(item_a, item_b, method="cosine"), 1.0))

    check("compute_similarity jaccard partial",
          compute_similarity(item_a, item_b, method="jaccard") > 0)

    check("compute_similarity text identical = 1",
          close(compute_similarity(item_a, item_b, method="text"), 1.0))


# ══════════════════════════════════════════════════════
#  MODULE 2 — candidate_gen.py
# ══════════════════════════════════════════════════════

def test_candidate_gen():
    section("candidate_gen.py")
    from candidate_gen import (
        candidates_by_tags,
        candidates_by_vector,
        candidates_from_history,
        candidates_by_popularity,
        generate_candidates,
    )

    # candidates_by_tags
    sci_fi_candidates = candidates_by_tags(CATALOGUE, {"sci-fi"}, min_overlap=1)
    check("candidates_by_tags finds sci-fi movies",
          all(c in sci_fi_candidates for c in ["item_001", "item_002", "item_004"]))

    check("candidates_by_tags excludes non-matching",
          "item_003" not in sci_fi_candidates)

    check("candidates_by_tags respects limit",
          len(candidates_by_tags(CATALOGUE, {"sci-fi"}, limit=2)) <= 2)

    no_match = candidates_by_tags(CATALOGUE, {"fantasy"}, min_overlap=1)
    check("candidates_by_tags returns empty when no match",
          len(no_match) == 0)

    # candidates_by_vector — query similar to Matrix/Inception
    query_vec = [0.85, 0.15, 0.75, 0.25]
    vec_candidates = candidates_by_vector(CATALOGUE, query_vec, threshold=0.9)
    check("candidates_by_vector finds similar items",
          len(vec_candidates) >= 1)

    very_diff_vec = [0.0, 1.0, 0.0, 1.0]
    check("candidates_by_vector filters by threshold",
          "item_001" not in candidates_by_vector(CATALOGUE, very_diff_vec, threshold=0.95))

    # candidates_from_history
    history_cands = candidates_from_history(CATALOGUE, ["item_001"])
    check("candidates_from_history excludes liked items",
          "item_001" not in history_cands)

    check("candidates_from_history finds related sci-fi items",
          any(c in history_cands for c in ["item_002", "item_004"]))

    # candidates_by_popularity
    popular = candidates_by_popularity(CATALOGUE, limit=3)
    check("candidates_by_popularity returns 3 items",
          len(popular) == 3)

    check("candidates_by_popularity: Inception (8.8) ranked first",
          popular[0] == "item_004")

    # generate_candidates (combined)
    combined = generate_candidates(
        CATALOGUE,
        query_tags={"sci-fi"},
        liked_item_ids=["item_001"],
        limit=10,
    )
    check("generate_candidates returns results",
          len(combined) > 0)

    check("generate_candidates excludes liked items",
          "item_001" not in combined)

    # Cold-start fallback
    cold_start = generate_candidates(CATALOGUE, limit=5)
    check("generate_candidates cold-start fallback returns items",
          len(cold_start) > 0)


# ══════════════════════════════════════════════════════
#  MODULE 3 — scorer.py
# ══════════════════════════════════════════════════════

def test_scorer():
    section("scorer.py")
    from scorer import (
        score_similarity,
        score_popularity,
        score_recency,
        score_diversity_penalty,
        compute_score,
        rank_candidates,
        explain_score,
    )

    matrix = CATALOGUE["item_001"]
    toy    = CATALOGUE["item_003"]

    # score_similarity
    sim = score_similarity(matrix, query_vector=[0.9, 0.1, 0.8, 0.2])
    check("score_similarity with identical vector ≈ 1",
          sim > 0.99)

    sim_tag = score_similarity(matrix, query_tags={"sci-fi", "action"})
    check("score_similarity tag-based > 0",
          sim_tag > 0)

    check("score_similarity no query = 0",
          close(score_similarity(matrix), 0.0))

    # score_popularity
    check("score_popularity 8.7/10 = 0.87",
          close(score_popularity(matrix), 0.87))

    check("score_popularity 10.0 clamps to 1.0",
          close(score_popularity({"rating": 10.0}), 1.0))

    check("score_popularity 0 = 0.0",
          close(score_popularity({"rating": 0.0}), 0.0))

    # score_recency
    rec_2024 = score_recency({"year": 2024}, current_year=2024)
    check("score_recency this year = 1.0",
          close(rec_2024, 1.0))

    rec_old = score_recency({"year": 1924}, current_year=2024, decay=0.05)
    check("score_recency 100 years old ≈ 0 (well below 0.01)",
          rec_old < 0.01)

    rec_1999 = score_recency(matrix, current_year=2024, decay=0.05)
    check("score_recency 1999 film is between 0 and 1",
          0 < rec_1999 < 1)

    # score_diversity_penalty
    check("diversity_penalty empty list = 1.0",
          close(score_diversity_penalty("x", matrix, []), 1.0))

    # Very similar item → penalty applies
    similar_item = {"tags": {"sci-fi", "action", "cyberpunk"}}
    penalty = score_diversity_penalty("x", matrix, [similar_item])
    check("diversity_penalty high similarity → < 1.0",
          penalty < 1.0)

    # compute_score
    score = compute_score(
        matrix,
        query_vector=[0.9, 0.1, 0.8, 0.2],
        query_tags={"sci-fi"},
    )
    check("compute_score returns value in [0, 1]",
          0.0 <= score <= 1.0)

    score_matrix = compute_score(matrix, query_vector=[0.9, 0.1, 0.8, 0.2])
    score_toy    = compute_score(toy,    query_vector=[0.9, 0.1, 0.8, 0.2])
    check("compute_score: Matrix scores higher than Toy Story for sci-fi query",
          score_matrix > score_toy)

    # rank_candidates
    candidates   = list(CATALOGUE.keys())
    ranked       = rank_candidates(
        CATALOGUE, candidates,
        query_vector=[0.9, 0.1, 0.8, 0.2],
        top_k=3,
    )
    check("rank_candidates returns 3 results", len(ranked) == 3)
    check("rank_candidates results have '_score' field",
          all("_score" in item for item in ranked))

    top_ids = [r["id"] for r in ranked]
    check("rank_candidates: Matrix or Inception in top 2 results",
          any(i in top_ids[:2] for i in ("item_001", "item_004")))

    # explain_score
    explanation = explain_score(matrix, query_vector=[0.9, 0.1, 0.8, 0.2])
    check("explain_score returns all signal keys",
          all(k in explanation for k in ["similarity", "popularity", "recency", "final"]))

    check("explain_score values in [0,1]",
          all(0.0 <= v <= 1.0 for v in explanation.values()))


# ══════════════════════════════════════════════════════
#  MODULE 4 — evaluator.py
# ══════════════════════════════════════════════════════

def test_evaluator():
    section("evaluator.py")
    from evaluator import (
        precision_at_k,
        recall_at_k,
        average_precision,
        mean_average_precision,
        dcg_at_k,
        ndcg_at_k,
        hit_rate_at_k,
        coverage,
        intra_list_diversity,
        evaluate,
        print_report,
    )

    recommended = ["item_001", "item_002", "item_003", "item_004", "item_005"]
    relevant    = {"item_001", "item_003", "item_005"}   # positions 1, 3, 5

    # precision@k
    check("precision@1 = 1.0 (first is relevant)",
          close(precision_at_k(recommended, relevant, 1), 1.0))

    check("precision@2 = 0.5 (1 of 2 relevant)",
          close(precision_at_k(recommended, relevant, 2), 0.5))

    check("precision@5 = 0.6 (3 of 5 relevant)",
          close(precision_at_k(recommended, relevant, 5), 0.6))

    check("precision@k = 0 when k=0",
          close(precision_at_k(recommended, relevant, 0), 0.0))

    # recall@k
    check("recall@1 = 1/3 (1 of 3 relevant found)",
          close(recall_at_k(recommended, relevant, 1), 1/3, tol=1e-5))

    check("recall@5 = 1.0 (all 3 relevant found)",
          close(recall_at_k(recommended, relevant, 5), 1.0))

    check("recall@k = 0 with empty relevant set",
          close(recall_at_k(recommended, set(), 5), 0.0))

    # average_precision
    ap = average_precision(recommended, relevant)
    check("average_precision > 0",
          ap > 0)

    check("average_precision ≤ 1",
          ap <= 1.0)

    perfect_ap = average_precision(["a", "b", "c"], {"a", "b", "c"})
    check("average_precision perfect ranking = 1.0",
          close(perfect_ap, 1.0))

    # mean_average_precision
    map_score = mean_average_precision(
        [recommended, ["item_003", "item_004"]],
        [relevant, {"item_003"}],
    )
    check("MAP is between 0 and 1",
          0.0 <= map_score <= 1.0)

    check("MAP empty input = 0",
          close(mean_average_precision([], []), 0.0))

    # dcg / ndcg
    dcg = dcg_at_k(recommended, relevant, 5)
    check("dcg@5 > 0", dcg > 0)

    ndcg = ndcg_at_k(recommended, relevant, 5)
    check("ndcg@5 in [0, 1]",
          0.0 <= ndcg <= 1.0)

    perfect_ndcg = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3)
    check("ndcg perfect ranking = 1.0",
          close(perfect_ndcg, 1.0))

    # hit_rate
    check("hit_rate@1 = 1 (first item relevant)",
          close(hit_rate_at_k(recommended, relevant, 1), 1.0))

    check("hit_rate@1 = 0 when first item not relevant",
          close(hit_rate_at_k(["item_002"], {"item_001"}, 1), 0.0))

    # coverage
    cov = coverage([["item_001", "item_002"], ["item_003"]], len(CATALOGUE))
    check("coverage 3 of 5 items = 0.6",
          close(cov, 0.6))

    check("coverage zero catalogue = 0",
          close(coverage([["item_001"]], 0), 0.0))

    # intra_list_diversity
    sci_fi_list = ["item_001", "item_002", "item_004"]   # all sci-fi → low diversity
    mixed_list  = ["item_001", "item_003", "item_005"]   # mixed genres → higher diversity

    div_sci_fi = intra_list_diversity(sci_fi_list, CATALOGUE)
    div_mixed  = intra_list_diversity(mixed_list,  CATALOGUE)
    check("diversity: mixed genres > similar genres",
          div_mixed > div_sci_fi)

    check("diversity single item = 0",
          close(intra_list_diversity(["item_001"], CATALOGUE), 0.0))

    # full evaluate report
    report = evaluate(recommended, relevant, CATALOGUE, k_values=[1, 5])
    check("evaluate report has precision@5",
          "precision@5" in report)

    check("evaluate report has ndcg@5",
          "ndcg@5" in report)

    check("evaluate report has diversity",
          "diversity" in report)

    print_report(report, title="Sample Evaluation Report")


# ══════════════════════════════════════════════════════
#  END-TO-END integration smoke test
# ══════════════════════════════════════════════════════

def test_end_to_end():
    section("End-to-End Integration")
    from candidate_gen import generate_candidates
    from scorer        import rank_candidates
    from evaluator     import evaluate, print_report

    # Simulate a user who liked The Matrix
    liked    = ["item_001"]
    query_v  = CATALOGUE["item_001"]["vector"]
    query_t  = CATALOGUE["item_001"]["tags"]

    # Step 1: generate candidates
    candidates = generate_candidates(
        CATALOGUE,
        query_vector=query_v,
        query_tags=query_t,
        liked_item_ids=liked,
        limit=10,
    )
    check("E2E: candidates generated",
          len(candidates) > 0)

    # Step 2: rank
    ranked = rank_candidates(
        CATALOGUE, candidates,
        query_vector=query_v,
        query_tags=query_t,
        top_k=3,
    )
    check("E2E: ranked list has ≤ 3 items",
          len(ranked) <= 3)

    # Step 3: evaluate
    recommended_ids = [r["id"] for r in ranked]
    relevant        = {"item_002", "item_004"}    # Ground truth: similar sci-fi films
    report          = evaluate(recommended_ids, relevant, CATALOGUE)
    check("E2E: evaluation report produced",
          len(report) > 0)

    print_report(report, "End-to-End Report (liked: The Matrix)")


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
