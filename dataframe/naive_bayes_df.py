"""
dataframe/naive_bayes_df.py
============================
Naive Bayes classifier implemented using the PySpark DataFrame / SQL API.

This file implements the same algorithm as rdd/naive_bayes_rdd.py but
expresses every operation as DataFrame transformations (groupBy, agg, join,
Window) instead of low-level RDD map/reduceByKey calls.

Why compare the two implementations?
  - DataFrame operations are compiled by Spark's Catalyst optimizer, which
    can apply query optimisations (predicate pushdown, column pruning, etc.)
    invisible to the RDD programmer.
  - For structured tabular data, the DataFrame API is often faster because
    Spark can avoid Python serialisation overhead via Tungsten's binary format.
  - However, the RDD API gives finer-grained control and is conceptually
    closer to "classic" MapReduce — important for this course.

Algorithm (same as RDD version):
  Training:
    1. Count class occurrences → P(class) with Laplace smoothing
    2. For each (class, feature, bin_value) → count occurrences → P(bin | class)
    3. Take log of all probabilities to avoid underflow
  Prediction:
    1. Broadcast log-probability tables
    2. For each test row, sum log priors + log likelihoods per class
    3. Return argmax
"""

import math
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
def get_spark():
    """
    Create or retrieve a SparkSession.
    On Databricks this returns the already-running session.
    """
    return (
        SparkSession.builder
        .appName("NaiveBayes-DataFrame")
        .master("local[*]")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_df, feature_cols: list, label_col: str = "label", num_bins: int = 3):
    """
    Train a Naive Bayes model using the DataFrame API.

    The output is a pair of DataFrames:
      - class_probs_df:   columns [label, log_class_prob]
      - feature_probs_df: columns [label, feature, bin_val, log_feat_prob]

    These DataFrames can be joined against test data during prediction,
    or collected to the driver and broadcast (see predict()).

    Args:
        train_df:     Spark DataFrame with feature columns (integer bins) + label column
        feature_cols: list of feature column names, e.g. ["sepal_length", …]
        label_col:    name of the column containing class labels
        num_bins:     number of discrete bins per feature (used in Laplace smoothing)

    Returns:
        (class_probs_df, feature_probs_df, classes)
        classes: Python list of unique class label strings
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Count class occurrences (equivalent to the MAP + REDUCE for class counts)
    # -----------------------------------------------------------------------
    # groupBy collapses all rows with the same label into one group;
    # agg(count) sums the rows in each group.
    # Result schema: [label, class_count]
    class_counts_df = (
        train_df
        .groupBy(label_col)
        .agg(F.count("*").alias("class_count"))
    )

    # Total number of training rows — needed for P(class) = count / total
    total_rows = train_df.count()

    # Number of distinct classes — used in Laplace smoothing for the prior:
    #   P(class) = (count + 1) / (total + num_classes)
    num_classes = class_counts_df.count()
    classes = [row[label_col] for row in class_counts_df.select(label_col).collect()]

    # ---- Compute log P(class) with Laplace smoothing ----
    # We add the smoothing constants as literal columns so the expression
    # can be evaluated as a single Spark SQL expression (no Python UDF needed).
    class_probs_df = class_counts_df.withColumn(
        "log_class_prob",
        F.log(
            (F.col("class_count") + F.lit(1)) /
            F.lit(float(total_rows + num_classes))
        )
    ).select(label_col, "log_class_prob")
    # After this step, class_probs_df looks like:
    #   | label      | log_class_prob |
    #   |------------|----------------|
    #   | setosa     | -1.098         |
    #   | virginica  | -1.099         |

    # -----------------------------------------------------------------------
    # STEP 2 — Count (class, feature, bin_value) occurrences
    #          (equivalent to the MAP + REDUCE for feature counts)
    # -----------------------------------------------------------------------
    # We need to "unpivot" the wide DataFrame (one column per feature) into a
    # tall DataFrame (one row per feature value per training row).
    # This is necessary because groupBy only works row-wise; we need to group
    # by (label, feature_name, bin_value) across all features at once.

    # Build a list of SELECT expressions that produce (label, feature_name, bin_val) rows.
    # We use F.lit(col_name) to embed the feature name as a column value.
    unpivot_exprs = [
        F.struct(
            F.col(label_col).alias("label"),
            F.lit(col_name).alias("feature"),
            F.col(col_name).alias("bin_val")
        )
        for col_name in feature_cols
    ]

    # F.explode on an array of structs creates one row per struct element.
    # This is the DataFrame equivalent of flatMap in the RDD version.
    tall_df = (
        train_df
        .select(F.explode(F.array(*unpivot_exprs)).alias("entry"))
        .select(
            F.col("entry.label").alias(label_col),
            F.col("entry.feature"),
            F.col("entry.bin_val")
        )
    )
    # After explode, tall_df looks like:
    #   | label   | feature      | bin_val |
    #   |---------|--------------|---------|
    #   | setosa  | sepal_length | 2       |
    #   | setosa  | sepal_width  | 1       |
    #   | setosa  | sepal_length | 2       |
    #   ...

    # Count occurrences of each (label, feature, bin_val) combination
    # This is the REDUCE step: summing counts per key
    feat_counts_df = (
        tall_df
        .groupBy(label_col, "feature", "bin_val")
        .agg(F.count("*").alias("feat_count"))
    )

    # ---- Join with class counts to get the Laplace smoothing denominator ----
    # We need class_count (total rows per class) to compute:
    #   P(bin | class) = (feat_count + 1) / (class_count + num_bins)
    feat_counts_df = feat_counts_df.join(
        class_counts_df.select(label_col, "class_count"),
        on=label_col,
        how="left"
    )

    # ---- Compute log P(bin | class) with Laplace smoothing ----
    feature_probs_df = feat_counts_df.withColumn(
        "log_feat_prob",
        F.log(
            (F.col("feat_count") + F.lit(1)) /
            (F.col("class_count") + F.lit(float(num_bins)))
        )
    ).select(label_col, "feature", "bin_val", "log_feat_prob")
    # After this step, feature_probs_df looks like:
    #   | label   | feature      | bin_val | log_feat_prob |
    #   |---------|--------------|---------|---------------|
    #   | setosa  | sepal_length | 2       | -0.693        |
    #   ...

    return class_probs_df, feature_probs_df, classes


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(test_df, class_probs_df, feature_probs_df, feature_cols: list,
            label_col: str = "label", num_bins: int = 3, spark=None):
    """
    Predict class labels for all rows in test_df.

    Strategy:
      We collect the probability tables to the driver as Python dicts, then
      broadcast them to workers and apply a UDF row-by-row.

      An alternative (pure DataFrame) approach would join test_df against
      feature_probs_df multiple times (once per feature), but that is complex
      and often slower due to the many joins.  The broadcast + UDF approach
      is simpler and equivalent to what the RDD version does.

    NOTE (UDF explanation):
      A UDF (User Defined Function) is required here because the classification
      logic involves iterating over classes and summing log-probabilities, which
      is Python procedural logic and cannot be expressed as a single Spark SQL
      expression.  The trade-off is that UDFs bypass Catalyst optimisation and
      involve Python serialisation overhead.

    Args:
        test_df:          Spark DataFrame with same schema as train_df
        class_probs_df:   output of train()
        feature_probs_df: output of train()
        feature_cols:     list of feature column names
        label_col:        name of the label column
        num_bins:         number of discrete bins (for Laplace fallback)
        spark:            active SparkSession (needed for broadcast)

    Returns:
        Spark DataFrame with an added column "prediction"
    """
    sc = spark.sparkContext

    # ---- Collect probability tables to driver ----
    # These tables are small (O(classes × features × bins)) so collecting them
    # to the driver and broadcasting is much cheaper than repeated DataFrame joins.
    log_class_probs = {
        row["label"]: row["log_class_prob"]
        for row in class_probs_df.collect()
    }

    log_feature_probs = {}
    for row in feature_probs_df.collect():
        key = (row["label"], row["feature"], row["bin_val"])
        log_feature_probs[key] = row["log_feat_prob"]

    classes = list(log_class_probs.keys())

    # Precompute the Laplace fallback log-probability for unseen combos
    # P_fallback(bin | class) = 1 / (class_count + num_bins) ≈ very small number
    # We approximate using the class prior denominator stored implicitly.
    # For simplicity, use log(1 / num_bins) as the fallback.
    # TODO: For a more precise fallback, collect class_counts_df and compute
    #       log(1 / (class_count + num_bins)) per class.
    fallback_log_prob = math.log(1.0 / (num_bins + 1))

    # Broadcast the dicts — sent once per executor, not once per task
    bc_class_probs   = sc.broadcast(log_class_probs)
    bc_feature_probs = sc.broadcast(log_feature_probs)
    bc_classes       = sc.broadcast(classes)
    bc_fallback      = sc.broadcast(fallback_log_prob)

    # ---- Define the classification UDF ----
    # Why a UDF and not a SQL expression?
    #   The argmax over classes requires iterating over a variable-length list
    #   of classes and comparing scores — this is not expressible in Spark SQL.
    def classify(*feature_values):
        """
        Classify a single row given its (already-binned) feature values.
        This function runs on worker nodes (inside Spark executors).

        Args:
            *feature_values: one integer bin value per feature, in the same
                             order as feature_cols

        Returns:
            The predicted class label (string).
        """
        clp  = bc_class_probs.value    # { label → log P(class) }
        flp  = bc_feature_probs.value  # { (label, feature, bin) → log P(bin|class) }
        clss = bc_classes.value        # [label1, label2, …]
        fb   = bc_fallback.value       # fallback log-prob for unseen combos

        best_class = None
        best_score = float("-inf")

        for cls in clss:
            score = clp[cls]  # log prior

            for feat_name, bin_val in zip(feature_cols, feature_values):
                key = (cls, feat_name, bin_val)
                # Add log likelihood; use fallback if this combo wasn't in training
                score += flp.get(key, fb)

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    # Register the UDF: takes N integer columns, returns a string label
    classify_udf = F.udf(classify, StringType())

    # ---- Apply UDF to every row ----
    # F.col(*feature_cols) expands to one Column argument per feature
    prediction_df = test_df.withColumn(
        "prediction",
        classify_udf(*[F.col(c) for c in feature_cols])
    )

    return prediction_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(prediction_df, label_col: str = "label"):
    """
    Compute accuracy from a DataFrame that has both 'label' and 'prediction' columns.

    Accuracy = fraction of rows where prediction == true label

    Args:
        prediction_df: output of predict(), with columns [label, …, prediction]
        label_col:     name of the true label column

    Returns:
        accuracy as a float in [0.0, 1.0]
    """
    total = prediction_df.count()
    if total == 0:
        return 0.0

    correct = (
        prediction_df
        .filter(F.col(label_col) == F.col("prediction"))
        .count()
    )
    return correct / total


# ---------------------------------------------------------------------------
# Main — end-to-end pipeline
# ---------------------------------------------------------------------------
def main():
    """
    Run the full training and prediction pipeline using the DataFrame implementation.
    """
    spark = get_spark()

    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import load_dataframe, FEATURE_NAMES, LABEL_COL, NUM_BINS

    # TODO: Replace None with your actual file path to use real data.
    #       Example (local):      "/Users/you/data/iris.csv"
    #       Example (Databricks): "dbfs:/FileStore/iris.csv"
    train_df, test_df, feature_stats = load_dataframe(spark, filepath=None)

    # ---- Train ----
    print("Training (DataFrame)...")
    t_start = time.time()
    class_probs_df, feature_probs_df, classes = train(
        train_df,
        feature_cols=FEATURE_NAMES,
        label_col=LABEL_COL,
        num_bins=NUM_BINS,
    )
    t_train = time.time() - t_start
    print(f"  Training time: {t_train:.3f}s")
    print(f"  Classes found: {classes}")

    # ---- Predict ----
    print("Predicting (DataFrame)...")
    t_start = time.time()
    prediction_df = predict(
        test_df,
        class_probs_df,
        feature_probs_df,
        feature_cols=FEATURE_NAMES,
        label_col=LABEL_COL,
        num_bins=NUM_BINS,
        spark=spark,
    )
    prediction_df.cache()
    t_predict = time.time() - t_start
    print(f"  Prediction time: {t_predict:.3f}s")

    # ---- Evaluate ----
    accuracy = evaluate(prediction_df, label_col=LABEL_COL)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    prediction_df.select(LABEL_COL, "prediction").show(10)

    spark.stop()


if __name__ == "__main__":
    main()
