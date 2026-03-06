# Day 29 Project — Recommendation System Pipeline

A modular recommendation engine built with pure Python (standard library only).
Covers similarity calculation, candidate generation, scoring/ranking, and evaluation metrics.

---

## Project Structure

```
day29_project/
├── similarity.py      # Cosine, Jaccard, and text similarity functions
├── candidate_gen.py   # Fast candidate filtering (tags, vectors, history, popularity)
├── scorer.py          # Multi-signal scoring and diversity-aware ranking
├── evaluator.py       # Precision@K, Recall@K, NDCG, MAP, Coverage, Diversity
└── test.py            # 79 test cases covering all modules + end-to-end
```

---

## Requirements

- **Python 3.10+** (uses `list[str]` and `set[str]` type hints)
- **No external libraries** — only `math`, `sys`, and `re` from the standard library

---

## How to Run

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/day29_project.git
cd day29_project
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Activate on Mac/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install dependencies

No packages needed. But if you want optional future extensions:

```bash
pip install -r requirements.txt
```

### 4. Run the tests

```bash
python test.py
```

Expected output:

```
══════════════════════════════════════════════════
  Day 29 Project — Test Suite
══════════════════════════════════════════════════
  Results: 79/79 passed 🎉 All tests passed!
══════════════════════════════════════════════════
```

### 5. Use the modules in your own script

```python
from similarity    import cosine_similarity, jaccard_similarity
from candidate_gen import generate_candidates
from scorer        import rank_candidates
from evaluator     import evaluate, print_report

catalogue = {
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
}

# Step 1 — generate candidates
candidates = generate_candidates(
    catalogue,
    query_tags={"sci-fi", "action"},
    liked_item_ids=["item_001"],
)

# Step 2 — rank them
ranked = rank_candidates(catalogue, candidates, query_tags={"sci-fi"}, top_k=5)
for item in ranked:
    print(f"{item['title']:30s}  score={item['_score']}")

# Step 3 — evaluate
recommended_ids = [r["id"] for r in ranked]
report = evaluate(recommended_ids, relevant={"item_002"}, catalogue=catalogue)
print_report(report)
```

---

## Module Overview

| Module | Key Functions |
|---|---|
| `similarity.py` | `cosine_similarity`, `jaccard_similarity`, `text_overlap_similarity`, `compute_similarity` |
| `candidate_gen.py` | `generate_candidates`, `candidates_by_tags`, `candidates_by_vector`, `candidates_by_popularity` |
| `scorer.py` | `rank_candidates`, `compute_score`, `explain_score` |
| `evaluator.py` | `evaluate`, `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `coverage`, `intra_list_diversity` |

---

## Scoring Signals (scorer.py)

| Signal | Default Weight | Description |
|---|---|---|
| Similarity | 0.50 | Cosine (vector) or Jaccard (tags) match to query |
| Popularity | 0.25 | Normalised item rating (0–10 scale) |
| Recency | 0.15 | Exponential decay based on item age |
| Diversity | 0.10 | Penalty for redundancy with already-ranked items |

Override weights per call:

```python
rank_candidates(
    catalogue, candidates,
    weights={"similarity": 0.7, "popularity": 0.3, "recency": 0.0, "diversity": 0.0}
)
```

---

## Quick One-liners

```bash
# Test cosine similarity
python -c "from similarity import cosine_similarity; print(cosine_similarity([1,0],[0,1]))"
# → 0.0

# Score breakdown for one item
python -c "
from scorer import explain_score
item = {'vector': [0.9, 0.1], 'tags': {'sci-fi'}, 'rating': 8.5, 'year': 2020}
print(explain_score(item, query_vector=[0.9, 0.1]))
"
```

---

## License

MIT — free to use and modify.