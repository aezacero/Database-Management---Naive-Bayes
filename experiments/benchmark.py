"""
experiments/benchmark.py
=========================
Scalability benchmark comparing the RDD and DataFrame Naive Bayes implementations.

What this script measures:
  - Training time for each implementation across multiple dataset sizes
  - Prediction time for each implementation across multiple dataset sizes
  - (Optional) Accuracy to verify both implementations agree

Experiment design:
  We create datasets of increasing size by repeating/sampling the base dataset.
  This simulates "small", "medium", and "large" workloads so we can plot
  how execution time scales with data volume (weak scaling experiment).

  A "strong scaling" experiment would fix the dataset size and vary the number
  of cluster nodes — that is left as a TODO for the Databricks cluster runs.

Why is this comparison useful for the report?
  - The DataFrame API benefits from Spark's Catalyst query optimiser and
    Tungsten execution engine (columnar memory, code generation).
  - The RDD API executes Python closures directly, with more serialisation
    overhead but finer control over the computation graph.
  - Comparing wall-clock times at different scales reveals at which point
    (if any) the DataFrame API's optimisations overcome its higher setup cost.
"""

import time
import sys
import os

# Make sure the parent directory is on the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pyspark.sql import SparkSession
from data.loader import load_car_rdd, load_car_dataframe, CAR_FEATURE_COLS, CAR_LABEL_COL
from rdd.naive_bayes_rdd import train as rdd_train, predict as rdd_predict, evaluate as rdd_evaluate
from dataframe.naive_bayes_df import (
    train as df_train, predict as df_predict, evaluate as df_evaluate
)


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
def get_spark():
    """
    Create a SparkSession for the benchmark.

    TODO (Databricks): When running on a cluster, remove .master("local[*]")
    and paste your cluster configuration below instead:
      .config("spark.executor.cores", "4")
      .config("spark.executor.memory", "8g")
      .config("spark.executor.instances", "4")
    The cluster config affects both implementations equally, so the relative
    comparison remains valid.
    """
    return (
        SparkSession.builder
        .appName("NaiveBayes-Benchmark")
        .master("local[*]")   # TODO: remove for Databricks cluster
        .config("spark.sql.shuffle.partitions", "8")  # TODO: tune for cluster size
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Dataset size factory
# ---------------------------------------------------------------------------
def make_scaled_rdd(base_rdd, scale_factor: int, spark):
    """
    Create a larger RDD by repeating the base dataset `scale_factor` times.

    This is the simplest way to create datasets of predictable sizes without
    needing a real large dataset.  Each repetition is a union of the original,
    so the class distribution is preserved.

    Args:
        base_rdd:     the original training RDD
        scale_factor: how many copies to stack (1 = original size)
        spark:        active SparkSession

    Returns:
        A new RDD with approximately (base_rows * scale_factor) rows.
    """
    scaled = base_rdd
    for _ in range(scale_factor - 1):
        # sc.union joins multiple RDDs into one without shuffling
        scaled = spark.sparkContext.union([scaled, base_rdd])
    return scaled


def make_scaled_df(base_df, scale_factor: int):
    """
    Create a larger DataFrame by repeating the base dataset `scale_factor` times.
    Uses DataFrame.union() which stacks rows without deduplication.

    Args:
        base_df:      the original training DataFrame
        scale_factor: how many copies to stack

    Returns:
        A new DataFrame with approximately (base_rows * scale_factor) rows.
    """
    scaled = base_df
    for _ in range(scale_factor - 1):
        scaled = scaled.union(base_df)
    return scaled


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_one_experiment(spark, scale_factor: int, filepath=None):
    """
    Run one benchmark experiment at a given scale factor.
    Returns a dict with timing results for both implementations.

    Args:
        spark:        active SparkSession
        scale_factor: dataset size multiplier (1 = base, 10 = 10x base, …)
        filepath:     path to real data CSV; None = use dummy data

    Returns:
        dict with keys: scale, n_train_rows, rdd_train_s, rdd_predict_s,
                        df_train_s, df_predict_s, rdd_acc, df_acc
    """
    print(f"\n--- Scale factor: {scale_factor}x ---")

    # Load base datasets (both RDD and DataFrame versions)
    # TODO: Replace None with your real dataset path for meaningful experiments.
    #       Dummy data has only 5 rows; results on it are not meaningful.
    base_train_rdd, test_rdd = load_car_rdd(spark, filepath=filepath)
    base_train_df,  test_df  = load_car_dataframe(spark, filepath=filepath)

    # Scale up the training data
    train_rdd = make_scaled_rdd(base_train_rdd, scale_factor, spark)
    train_df  = make_scaled_df(base_train_df,  scale_factor)

    # Materialise the scaled RDD so timing is fair (lazy evaluation would
    # otherwise include dataset creation time in the first operation)
    train_rdd.cache()
    train_df.cache()
    n_train = train_rdd.count()
    print(f"  Training rows: {n_train}")

    result = {"scale": scale_factor, "n_train_rows": n_train}

    # ---- RDD benchmark ----
    print("  [RDD] Training...")
    t0 = time.time()
    rdd_model = rdd_train(train_rdd, num_features=len(CAR_FEATURE_COLS), num_bins=3)
    result["rdd_train_s"] = time.time() - t0
    print(f"    Train time: {result['rdd_train_s']:.3f}s")

    print("  [RDD] Predicting...")
    t0 = time.time()
    rdd_preds = rdd_predict(test_rdd, rdd_model, spark)
    rdd_preds.cache()
    result["rdd_predict_s"] = time.time() - t0
    result["rdd_acc"] = rdd_evaluate(rdd_preds)
    print(f"    Predict time: {result['rdd_predict_s']:.3f}s  |  Accuracy: {result['rdd_acc']:.3f}")

    # ---- DataFrame benchmark ----
    print("  [DF] Training...")
    t0 = time.time()
    class_probs, feat_probs, classes = df_train(
        train_df, feature_cols=CAR_FEATURE_COLS, label_col=CAR_LABEL_COL, num_bins=3
    )
    result["df_train_s"] = time.time() - t0
    print(f"    Train time: {result['df_train_s']:.3f}s")

    print("  [DF] Predicting...")
    t0 = time.time()
    df_preds = df_predict(
        test_df, class_probs, feat_probs,
        feature_cols=CAR_FEATURE_COLS, label_col=CAR_LABEL_COL,
        num_bins=3, spark=spark
    )
    df_preds.cache()
    result["df_predict_s"] = time.time() - t0
    result["df_acc"] = df_evaluate(df_preds, label_col=CAR_LABEL_COL)
    print(f"    Predict time: {result['df_predict_s']:.3f}s  |  Accuracy: {result['df_acc']:.3f}")

    # Clean up cached data to free memory before next experiment
    train_rdd.unpersist()
    train_df.unpersist()
    rdd_preds.unpersist()
    df_preds.unpersist()

    return result


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------
def print_results_table(results: list):
    """
    Print a formatted table summarising all benchmark results.

    Args:
        results: list of dicts returned by run_one_experiment()
    """
    print("\n" + "=" * 85)
    print(f"{'BENCHMARK RESULTS':^85}")
    print("=" * 85)
    header = (
        f"{'Scale':>6}  {'Rows':>8}  "
        f"{'RDD Train':>10}  {'RDD Pred':>9}  "
        f"{'DF Train':>9}  {'DF Pred':>8}  "
        f"{'RDD Acc':>8}  {'DF Acc':>7}"
    )
    print(header)
    print("-" * 85)
    for r in results:
        row = (
            f"{r['scale']:>6}x  {r['n_train_rows']:>8,}  "
            f"{r['rdd_train_s']:>9.3f}s  {r['rdd_predict_s']:>8.3f}s  "
            f"{r['df_train_s']:>8.3f}s  {r['df_predict_s']:>7.3f}s  "
            f"{r['rdd_acc']:>8.3f}  {r['df_acc']:>7.3f}"
        )
        print(row)
    print("=" * 85)

    # Summary: which is faster on average?
    avg_rdd = sum(r["rdd_train_s"] + r["rdd_predict_s"] for r in results) / len(results)
    avg_df  = sum(r["df_train_s"]  + r["df_predict_s"]  for r in results) / len(results)
    winner  = "RDD" if avg_rdd < avg_df else "DataFrame"
    print(f"\nAverage total time — RDD: {avg_rdd:.3f}s | DataFrame: {avg_df:.3f}s")
    print(f"Faster on average: {winner}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """
    Run scalability experiments across three dataset sizes and print a results table.

    TODO: Adjust SCALE_FACTORS to cover the range that makes sense for your
          cluster.  On a laptop with dummy data, use [1, 2, 3].
          On Databricks with real data, use [1, 10, 100] or larger.
    TODO: Replace DATA_PATH with the path to your real CSV file on DBFS:
          DATA_PATH = "dbfs:/FileStore/iris.csv"
          Set to None to use the hardcoded dummy dataset (local testing only).
    """
    # TODO: Set DATA_PATH to your dataset file.
    #       None → 5-row dummy dataset (local smoke test only)
    #       "dbfs:/FileStore/iris.csv" → real Iris data on Databricks DBFS
    DATA_PATH = None

    # TODO: Adjust these scale factors.
    #       Small / medium / large should produce meaningfully different runtimes.
    #       With real Iris (150 rows): try [1, 50, 200] to get 150 / 7500 / 30000 rows.
    SCALE_FACTORS = [1, 2, 3]  # TODO: replace with [1, 50, 200] for real experiments

    spark = get_spark()

    all_results = []
    for sf in SCALE_FACTORS:
        result = run_one_experiment(spark, scale_factor=sf, filepath=DATA_PATH)
        all_results.append(result)

    print_results_table(all_results)

    spark.stop()


if __name__ == "__main__":
    main()
