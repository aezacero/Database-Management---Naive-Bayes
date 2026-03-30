"""
rdd/naive_bayes_rdd.py
======================
Naive Bayes classifier implemented using the PySpark RDD API.

Algorithm overview:
  Training is a classic MapReduce job:
    - MAP phase:  each data row emits key-value pairs that record what
                  we need to count (class occurrences, feature-value-per-class
                  occurrences, total row count).
    - REDUCE phase: sum all values for identical keys across the whole dataset.
    - POST-PROCESSING: convert raw counts into log-probabilities with
                       Laplace smoothing.

  Prediction does NOT use MapReduce:
    - The probability tables are broadcast to every worker.
    - Each row independently computes the log-probability sum for every class
      and returns the class with the highest score (argmax).

Why RDD and not DataFrame?
  RDDs give us fine-grained control over exactly what gets shuffled across
  the network.  For a MapReduce implementation this transparency is important:
  we can see every (key, value) pair being emitted, grouped, and reduced.

Design choices:
  - Multinomial Naive Bayes with discrete (binned) features.
    Continuous features must be binned before calling train().
    See data/loader.py for the binning logic.
  - Laplace smoothing avoids zero-probability for unseen feature values.
  - Log probabilities avoid floating-point underflow when multiplying
    many small probabilities together.
"""

import math
import time
from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# SparkSession — local mode for development/testing
# ---------------------------------------------------------------------------
def get_spark():
    """
    Create or retrieve a SparkSession.
    On Databricks this returns the pre-existing session; locally it starts one.
    """
    return (
        SparkSession.builder
        .appName("NaiveBayes-RDD")
        .master("local[*]")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_rdd, num_features: int, num_bins: int = 3):
    """
    Train a Naive Bayes model using the MapReduce paradigm on an RDD.

    The model is a set of log-probability tables:
      - log_class_probs:   dict { class_label → log P(class) }
      - log_feature_probs: dict { (class_label, feature_idx, bin_value) → log P(bin | class) }

    These tables are returned as plain Python dicts so they can be broadcast
    to workers during prediction.

    Args:
        train_rdd:    RDD of (label, [bin_0, bin_1, …, bin_{n-1}])
                      Labels are strings; bin values are integers in [0, num_bins-1].
        num_features: number of feature columns (needed for Laplace smoothing denominator)
        num_bins:     number of discrete bins per feature (needed for Laplace smoothing)

    Returns:
        (log_class_probs, log_feature_probs)
    """

    # -----------------------------------------------------------------------
    # MAP PHASE
    # Each row emits three categories of key-value pairs that capture all
    # the counts we need for Naive Bayes.
    # -----------------------------------------------------------------------
    def map_row(row):
        """
        For a single training row (label, [bins]), emit:

          1. Class count key:
             ("class_count", label) → 1
             Example: ("class_count", "setosa") → 1

          2. Feature-value-per-class count keys (one per feature):
             ("feat_count", label, feature_idx, bin_value) → 1
             Example: ("feat_count", "setosa", 0, 2) → 1
               meaning: feature 0 had bin value 2 for a row labelled "setosa"

          3. Total row count:
             ("total",) → 1
             Used to compute P(class) = count(class) / total

        Why emit a single flat stream instead of separate RDDs?
          Combining everything into one RDD lets us do a single pass over
          the data and a single shuffle — more efficient than three separate jobs.
        """
        label, bins = row
        pairs = []

        # Emit 1 for this class occurrence
        pairs.append((("class_count", label), 1))

        # Emit 1 for each (class, feature_index, bin_value) combination
        for feature_idx, bin_val in enumerate(bins):
            pairs.append((("feat_count", label, feature_idx, bin_val), 1))

        # Emit 1 toward the total row count
        pairs.append((("total",), 1))

        return pairs

    # flatMap: apply map_row to every row and flatten the list of lists into
    # a single RDD of (key, 1) pairs.
    # After flatMap, data looks like:
    #   (("class_count", "setosa"), 1)
    #   (("feat_count", "setosa", 0, 2), 1)
    #   (("total",), 1)
    #   (("class_count", "virginica"), 1)
    #   ...
    mapped_rdd = train_rdd.flatMap(map_row)

    # -----------------------------------------------------------------------
    # REDUCE PHASE
    # Sum all values for identical keys across the entire dataset.
    # reduceByKey triggers a shuffle: Spark groups all pairs with the same key
    # onto the same worker, then applies the lambda to accumulate values.
    # -----------------------------------------------------------------------
    # After reduceByKey, data looks like:
    #   (("class_count", "setosa"), 50)
    #   (("feat_count", "setosa", 0, 2), 18)
    #   (("total",), 150)
    #   ...
    counts_rdd = mapped_rdd.reduceByKey(lambda a, b: a + b)

    # Collect all counts to the driver — the result fits in memory because
    # it is O(classes × features × bins), not O(dataset rows).
    all_counts = dict(counts_rdd.collect())

    # -----------------------------------------------------------------------
    # POST-PROCESSING: convert counts → log-probabilities
    # -----------------------------------------------------------------------
    total_rows = all_counts.get(("total",), 0)

    # Extract class counts: {"setosa": 50, "virginica": 50, "versicolor": 50}
    class_counts = {
        key[1]: count
        for key, count in all_counts.items()
        if key[0] == "class_count"
    }
    num_classes = len(class_counts)

    # ---- Log prior: log P(class) ----
    # P(class) = count(class) / total_rows
    # We add Laplace smoothing to the class prior as well:
    #   P(class) = (count(class) + 1) / (total_rows + num_classes)
    # This handles the (unlikely) case where a class has zero training examples.
    log_class_probs = {
        label: math.log((count + 1) / (total_rows + num_classes))
        for label, count in class_counts.items()
    }

    # ---- Log likelihood: log P(feature_value | class) with Laplace smoothing ----
    # For each (class, feature, bin_value) combination:
    #   P(bin | class) = (count(class, feature, bin) + 1)
    #                    / (count(class) + num_bins)
    #
    # The "+1" in the numerator and "+num_bins" in the denominator is Laplace
    # smoothing.  num_bins is the number of possible values for this feature.
    # This ensures P > 0 even for (class, feature, bin) combos not in training.
    log_feature_probs = {}
    for key, count in all_counts.items():
        if key[0] != "feat_count":
            continue
        _, label, feature_idx, bin_val = key
        class_count = class_counts.get(label, 0)
        smoothed_prob = (count + 1) / (class_count + num_bins)
        log_feature_probs[(label, feature_idx, bin_val)] = math.log(smoothed_prob)

    # For any (class, feature, bin) combo that had zero occurrences in training,
    # we compute the Laplace-smoothed log probability on the fly during prediction.
    # Store the denominator so prediction can reconstruct it.
    # (This avoids storing O(classes × features × bins) entries for unseen combos.)
    smoothing_denominators = {
        label: class_counts.get(label, 0) + num_bins
        for label in class_counts
    }

    model = {
        "log_class_probs": log_class_probs,
        "log_feature_probs": log_feature_probs,
        "smoothing_denominators": smoothing_denominators,
        "num_bins": num_bins,
        "num_features": num_features,
        "classes": list(class_counts.keys()),
    }
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(test_rdd, model: dict, spark):
    """
    Predict class labels for all rows in test_rdd.

    Strategy:
      1. Broadcast the model (probability tables) to all workers so each
         worker can look up probabilities without network communication.
      2. For each test row, compute the log-probability score for every class:
           score(class) = log P(class)
                        + sum over features of log P(feature_value | class)
      3. Return the class with the highest score (argmax).

    Why broadcast?
      Without broadcasting, Spark would serialise the model dict and ship it
      with every task.  Broadcasting sends it once per worker (executor),
      which is much more efficient when the model is large.

    Args:
        test_rdd: RDD of (true_label, [bin_0, …, bin_{n-1}])
        model:    dict returned by train()
        spark:    active SparkSession

    Returns:
        RDD of (true_label, predicted_label)
    """
    sc = spark.sparkContext

    # Broadcast model to all workers — sent once per executor, not per task
    bc_model = sc.broadcast(model)

    def classify_row(row):
        """
        Score every class for a single test row using log-probability sums.
        Returns (true_label, predicted_label).
        """
        true_label, bins = row
        m = bc_model.value  # access the broadcast copy on this worker

        best_class = None
        best_score = float("-inf")

        for cls in m["classes"]:
            # Start with the log prior for this class
            score = m["log_class_probs"][cls]

            # Add log likelihood for each feature value
            for feature_idx, bin_val in enumerate(bins):
                key = (cls, feature_idx, bin_val)
                if key in m["log_feature_probs"]:
                    # We saw this (class, feature, bin) combo in training
                    score += m["log_feature_probs"][key]
                else:
                    # Unseen combo: apply Laplace smoothing on the fly
                    # P = 1 / (class_count + num_bins)  [numerator count = 0, +1 smoothing]
                    denom = m["smoothing_denominators"][cls]
                    score += math.log(1.0 / denom)

            if score > best_score:
                best_score = score
                best_class = cls

        return (true_label, best_class)

    # map: apply classify_row to every test row independently (no shuffle needed)
    # After map, data looks like:
    #   ("setosa", "setosa")
    #   ("virginica", "versicolor")   ← misclassification example
    #   ...
    predictions_rdd = test_rdd.map(classify_row)
    return predictions_rdd


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(predictions_rdd):
    """
    Compute classification accuracy from (true_label, predicted_label) pairs.

    Accuracy = number of correct predictions / total predictions

    Args:
        predictions_rdd: RDD of (true_label, predicted_label)

    Returns:
        accuracy as a float in [0.0, 1.0]
    """
    # map each pair to 1 (correct) or 0 (wrong), then compute the mean
    # This is a single pass over the data — efficient for large datasets.
    total   = predictions_rdd.count()
    correct = predictions_rdd.filter(lambda pair: pair[0] == pair[1]).count()

    if total == 0:
        return 0.0
    return correct / total


# ---------------------------------------------------------------------------
# Main — end-to-end pipeline
# ---------------------------------------------------------------------------
def main():
    """
    Run the full training and prediction pipeline using the RDD implementation.
    Prints accuracy and total wall-clock time.
    """
    spark = get_spark()

    # ---- Load data ----
    # We import here (not at top-level) to avoid circular imports when
    # benchmark.py imports from both rdd/ and dataframe/.
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from data.loader import load_car_rdd, CAR_FEATURE_COLS

    # TODO: Replace None with your actual file path to use real data.
    #       Example (local):      "/Users/you/data/car.data"
    #       Example (Databricks): "dbfs:/FileStore/car.data"
    train_rdd, test_rdd = load_car_rdd(spark, filepath=None)

    num_features = len(CAR_FEATURE_COLS)
    num_bins     = 3  # approximate vocab size per categorical feature for Laplace smoothing

    # ---- Train ----
    print("Training (RDD)...")
    t_start = time.time()
    model = train(train_rdd, num_features=num_features, num_bins=num_bins)
    t_train = time.time() - t_start
    print(f"  Training time: {t_train:.3f}s")
    print(f"  Classes found: {model['classes']}")

    # ---- Predict ----
    print("Predicting (RDD)...")
    t_start = time.time()
    predictions_rdd = predict(test_rdd, model, spark)
    predictions_rdd.cache()  # cache so evaluate() doesn't recompute predictions
    t_predict = time.time() - t_start
    print(f"  Prediction time: {t_predict:.3f}s")

    # ---- Evaluate ----
    accuracy = evaluate(predictions_rdd)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    spark.stop()


if __name__ == "__main__":
    main()
