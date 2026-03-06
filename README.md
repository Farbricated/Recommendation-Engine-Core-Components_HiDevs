# Day 29 Project — Recommendation Engine Core Components

A modular recommendation engine built with **pure Python** (standard library only — no pip install needed).  
Implements four class-based components that mirror how platforms like Netflix and Amazon suggest items.

---

## Project Structure

```
day29_project/
├── similarity.py      # SimilarityCalculator  — cosine, jaccard, pearson
├── candidate_gen.py   # CandidateGenerator    — collaborative, content-based, popularity, hybrid
├── scorer.py          # RecommendationScorer  — pluggable weighted scoring + ranking
├── evaluator.py       # RecommendationEvaluator — precision, recall, ndcg, evaluate_all
└── test.py            # 85 test cases across all components + end-to-end
```

---

## Requirements

- **Python 3.10+**
- **Zero external libraries** — uses only `math` and `sys` from the standard library

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/day29_project.git
cd day29_project
```

### 2. (Optional) Virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Run the tests

```bash
python test.py
```

Expected output:
```
══════════════════════════════════════════════════
  Day 29 Project — Test Suite
══════════════════════════════════════════════════
  Results: 85/85 passed 🎉 All tests passed!
══════════════════════════════════════════════════
```

---

## Component Overview

### Component 1 — `SimilarityCalculator` (`similarity.py`)

Measures how similar two users or items are.

```python
from similarity import SimilarityCalculator
sc = SimilarityCalculator()

sc.cosine_similarity([1, 0, 0], [1, 0, 0])          # → 1.0
sc.jaccard_similarity({"sci-fi", "action"}, {"sci-fi", "drama"})  # → 0.333
sc.pearson_correlation({"m1": 5, "m2": 3}, {"m1": 5, "m2": 3})   # → 1.0
sc.most_similar("user_A", all_user_vectors, top_k=5)
```

| Method | Input | Output | Use Case |
|---|---|---|---|
| `cosine_similarity` | two float lists | [-1, 1] | User/item embeddings |
| `jaccard_similarity` | two sets | [0, 1] | Tags, skills, genres |
| `pearson_correlation` | two rating dicts | [-1, 1] | Rating patterns |
| `most_similar` | id + vector dict | ranked list | Finding neighbours |

---

### Component 2 — `CandidateGenerator` (`candidate_gen.py`)

Generates a pool of candidate items before scoring.

```python
from candidate_gen import CandidateGenerator
gen = CandidateGenerator()

gen.collaborative_candidates("user_A")   # items liked by similar users
gen.content_based_candidates("user_A")  # items matching user's taste tags
gen.popularity_candidates()             # top-rated items (cold-start)
gen.hybrid_candidates("user_A")         # best of all three
```

All methods handle **cold-start** (new users with no history) by falling back to popularity.

---

### Component 3 — `RecommendationScorer` (`scorer.py`)

Scores candidates and returns a ranked list with explanations.

```python
from scorer import RecommendationScorer
scorer = RecommendationScorer()

# Add a custom scorer
scorer.add_scorer("my_signal", lambda u, i, ctx: 0.9, weight=0.3)

# Score one item
result = scorer.calculate_score("user_A", "item_7")
# → {"total": 0.82, "signals": {"relevance": 1.0, ...}, "reason": "..."}

# Rank all candidates
ranked = scorer.rank_candidates("user_A", ["item_4", "item_5", "item_7"], limit=5)
```

**Default signals:**

| Signal | Weight | Description |
|---|---|---|
| Relevance | 0.50 | Tag overlap with user's liked items |
| Popularity | 0.30 | Normalised item rating |
| Recency | 0.20 | How new the item is |

---

### Component 4 — `RecommendationEvaluator` (`evaluator.py`)

Measures recommendation quality with offline metrics.

```python
from evaluator import RecommendationEvaluator
ev = RecommendationEvaluator()

ev.precision_at_k(recommendations, relevant_items, k=5)  # % relevant in top-k
ev.recall_at_k(recommendations, relevant_items, k=5)     # % of relevant found
ev.ndcg_at_k(recommendations, relevant_items, k=5)       # position-weighted score

# Evaluate across all users at once
report = ev.evaluate_all(recs_dict, ground_truth_dict, k=10)
ev.print_report(report)
```

---

## End-to-End Example

```python
from candidate_gen import CandidateGenerator
from scorer        import RecommendationScorer
from evaluator     import RecommendationEvaluator

gen    = CandidateGenerator()
scorer = RecommendationScorer()
ev     = RecommendationEvaluator()

# 1. Generate candidates
candidates = gen.hybrid_candidates("user_A")

# 2. Rank them
ranked = scorer.rank_candidates("user_A", candidates, limit=10)
for r in ranked:
    print(f"{r['item_id']}  {r['total']:.2f}  {r['reason']}")

# 3. Evaluate
recs_dict    = {"user_A": [r["item_id"] for r in ranked]}
ground_truth = {"user_A": {"item_4", "item_5", "item_7"}}
report       = ev.evaluate_all(recs_dict, ground_truth, k=5)
ev.print_report(report)
```

---

## License

MIT — free to use and modify.