# Naive Bayes MapReduce — Spark Project

## Project Context
University project for Database Management / Big Data at École Polytechnique (Prof. Dario Colazzo).
Due: **April 3, 2026**. Group of 3.

## Goal
Implement a Naive Bayes classifier using the MapReduce paradigm in PySpark, in two flavours:
- **RDD API** — low-level, explicit map/reduce operations (baseline + optimized)
- **DataFrame API** — high-level SQL-style operations (baseline + optimized)

Then **compare all four experimentally** on Google Dataproc (execution time, scalability).

## File Ownership
| Person | Files |
|--------|-------|
| Person 1 (you) | `core/naive_bayes.py`, `rdd/naive_bayes_rdd_baseline.ipynb`, `rdd/naive_bayes_rdd_optimized.ipynb` |
| Person 2 | `dataframe/naive_bayes_df_baseline.ipynb`, `dataframe/naive_bayes_df_optimized.ipynb` |
| Person 3 | `experiments/benchmark.ipynb`, report lead |

**Person 2 needs**: `core/naive_bayes.py` committed to GitHub before they can start.
**Person 3 needs**: any working notebook + `data/loader.py` before they can start.

## Repo Structure
```
core/naive_bayes.py                      ← shared pure-Python logic (no Spark)
rdd/naive_bayes_rdd_baseline.ipynb       ← RDD baseline (no optimisations)
rdd/naive_bayes_rdd_optimized.ipynb      ← RDD with 4 optimisations
dataframe/naive_bayes_df_baseline.ipynb  ← DataFrame baseline (UDFs, no persist)
dataframe/naive_bayes_df_optimized.ipynb ← DataFrame with 3 optimisations
experiments/benchmark.ipynb             ← timing & scalability across all 4
data/loader.py                           ← dataset loading utilities
CLAUDE.md                                ← this file
```

## Dataset
**UCI Car Evaluation** — all categorical features, no discretisation needed.
- 1728 instances, 6 features, 4 classes
- Columns: `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety` → `class`
- Classes: `unacc`, `acc`, `good`, `vgood`
- Expected accuracy: **~87.39%** (per Zheng 2014 reference paper)
- Class distribution: heavily skewed (~70% unacc) — note this in the report

## Algorithm Summary

### Training Phase (MapReduce)
1. **Map**: each row emits key-value pairs using string keys:
   - `"feat_{i}_{value}_{class}"` → 1  (feature-value-per-class count)
   - `"class_{label}"` → 1  (class count)
   - `"total"` → 1  (total row count)
   Key format matches Zheng (2014) mapper convention.
2. **Reduce**: sum all counts globally
3. **Probability computation** (in `core/naive_bayes.py`):
   - `log P(class)` = log((count + 1) / (total + num_classes))  ← Laplace smoothed
   - `log P(val | class)` = log((count + 1) / (class_count + |V_i|))  ← Laplace smoothed

### Prediction Phase
1. Look up stored log probabilities (broadcast in optimized versions)
2. For each row: `score(class) = log P(class) + Σ log P(feat_i | class)`
3. Return `argmax` over all classes

### Why log probabilities?
Multiplying many small probabilities underflows to 0.0 in floating point.
`log(a × b × c) = log(a) + log(b) + log(c)` — same argmax, numerically safe.

### Why Laplace smoothing?
Zero count → zero probability → kills the entire product.
Adding pseudocount of 1 ensures P > 0 for all combos including unseen ones.

## Symmetric Optimization Strategy
The baseline/optimized pairs are designed symmetrically so the RDD vs DataFrame
comparison is apples-to-apples:

| Optimization | RDD version | DataFrame version |
|---|---|---|
| Local pre-aggregation | `flatMap` → `mapPartitions` | UDF key builder → built-in `F.concat` |
| Efficient shuffle | `groupByKey` → `reduceByKey` | UDF log-prob → `F.log()` (Catalyst-friendly) |
| Broadcast model | no broadcast → `sc.broadcast()` | — |
| Avoid recomputation | no persist → `.persist()` | no persist → `.persist()` |
| Partition tuning | — | no repartition → `.repartition(n)` |

## Augmentation Strategy (Scalability Experiments)
To simulate scalability without a truly large dataset, we repeat the base dataset:
- **Small**: × 1 (1,728 rows)
- **Medium**: × 10 (17,280 rows)
- **Large**: × 50 (86,400 rows)

This matches the augmentation approach from Zheng (2014).

## Grading Rubric (19 pts total)
| # | Criterion | Points |
|---|-----------|--------|
| 1 | **Solution description** — clear explanation of what the solution does | 4 pts |
| 2 | **Algorithm design + global comments** — overall structure, design choices explained | 4 pts |
| 3 | **Code comments on main fragments** — inline comments on key code blocks | 3 pts |
| 4 | **Scalability experiments** — timing results across dataset sizes, Google Dataproc cluster | 3 pts |
| 5 | **Weak/strong points analysis** — honest comparison of RDD vs DataFrame, bottlenecks | 3 pts |
| 6 | **Code appendix** — clean, readable code attached to the report | 2 pts |

## What Good Comments Look Like
- Explain **WHY**, not what — the code already says what
- One comment per logical block, not per line
- For RDD transformations: show what the data looks like before and after with example tuples:
  ```
  # Input:  ("unacc", ["low", "med", "2", "2", "small", "low"])
  # Output: [("feat_0_low_unacc", 1), ("class_unacc", 1), ("total", 1), ...]
  ```
- For DataFrame transformations: show the schema at each major step:
  ```
  # Schema after groupBy: [label, feature_name, feature_value, count]
  ```

## Baselines Must Remain True Baselines
Do **not** optimise the baseline notebooks even if the improvement is obvious.
The whole point of having a baseline is to measure what the optimizations actually gain.

## Timing Pattern (use consistently across all notebooks)
```python
t_start = time.time()
# ... code being timed ...
duration = time.time() - t_start
print(f"[TIMING] Training: {duration:.4f}s")
```

## Reference Paper
Zheng, S. (2014). *Naïve Bayes Classifier: A MapReduce Approach*. NDSU Master's Thesis.
- Key mapper output format: `"feat_i_value_class"` → 1
- Expected accuracy on car evaluation dataset: ~87.39%

## Running Locally
All notebooks include a local SparkSession setup so they can be tested without a cluster.
Use the dummy dataset in `data/loader.py` (5 hardcoded rows) for a quick smoke test.
To switch to real data: replace `DATA_PATH = None` with `"gs://your-bucket/car.data"`.
