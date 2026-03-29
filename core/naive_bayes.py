"""
core/naive_bayes.py
===================
Pure Python logic shared by both RDD and DataFrame implementations.
No PySpark imports — this module is intentionally framework-agnostic
so it can be unit-tested independently and reused across all four notebooks.

Role in the pipeline:
  1. Training notebooks produce raw counts via MapReduce (flatMap + groupByKey, etc.)
  2. Those counts are passed into compute_log_probs() here to get a probability table.
  3. predict() uses that table to classify individual rows.
  4. evaluate() measures classification accuracy.

Key format used throughout:
  "feat_{feature_index}_{value}_{class_label}"
  Example: "feat_0_low_unacc" means feature 0 had value "low" for class "unacc".
  This format matches the mapper convention in Zheng (2014).
"""

import math
from collections import defaultdict


def compute_log_probs(class_counts, feature_counts, class_totals):
    """
    Convert raw training counts into a log-probability lookup table.

    Why log probabilities?
        Naive Bayes multiplies P(class) * P(f1|class) * P(f2|class) * ...
        With even 6 features, each probability ~0.3, the product is ~0.001.
        With hundreds of features it underflows to 0.0 in floating point,
        making it impossible to compare classes. Using log transforms the
        product into a sum: log(a*b*c) = log(a)+log(b)+log(c), which stays
        in a numerically safe range. The argmax is identical either way.

    Why Laplace smoothing?
        If a (feature, value, class) combo was never seen in training, its
        count is 0 so P = 0, which zeros out the entire product regardless
        of all other features. Adding pseudocount +1 to every count ensures
        P > 0 everywhere. The denominator gains +|V_i| (number of unique
        values for feature i) to keep the distribution normalised.

    Args:
        class_counts  (dict): {class_label: int}
            Number of training rows belonging to each class.

        feature_counts (dict): {"feat_{i}_{value}_{class}": int}
            Number of times feature i had a given value for a given class.
            Key format must match exactly what the training notebooks emit.

        class_totals  (dict): {class_label: int}
            Total training rows per class, used as the Laplace denominator.
            For standard Naive Bayes this is identical to class_counts.

    Returns:
        dict with four keys:
            "classes"           : list[str]  — all class labels seen in training
            "log_class_probs"   : dict        — log P(class) for each class
            "log_feature_probs" : dict        — log P(value | class) per key string
            "fallback_log_probs": dict        — Laplace fallback for unseen combos,
                                               keyed by (feature_idx, class_label)
    """
    total_samples = sum(class_counts.values())
    num_classes   = len(class_counts)

    # --- Log prior: log P(class) ---
    # Standard formula: P(class) = count(class) / total_samples.
    # We apply mild Laplace smoothing (+1 / +num_classes) here too, so that
    # a class with zero training examples doesn't produce log(0).
    log_class_probs = {
        label: math.log((count + 1) / (total_samples + num_classes))
        for label, count in class_counts.items()
    }

    # --- Derive unique values per feature index from the key strings ---
    # We need |V_i| (the vocabulary size for feature i) to compute the
    # Laplace-smoothed denominator: P(val|class) = (count+1) / (class_count + |V_i|).
    # Key format: "feat_{feat_idx}_{value}_{class_label}"
    # We parse every seen key to collect all values per feature.
    unique_vals_per_feature = defaultdict(set)
    for key in feature_counts:
        parts    = key.split("_")
        feat_idx = int(parts[1])
        # Value occupies parts[2:-1]; handles values that contain underscores.
        value    = "_".join(parts[2:-1])
        unique_vals_per_feature[feat_idx].add(value)

    num_unique_vals = {idx: len(vals) for idx, vals in unique_vals_per_feature.items()}

    # --- Log likelihood: log P(feature_value | class) ---
    # Applied for every (feature, value, class) combination seen in training.
    log_feature_probs = {}
    for key, count in feature_counts.items():
        parts       = key.split("_")
        feat_idx    = int(parts[1])
        class_label = parts[-1]
        num_vals    = num_unique_vals.get(feat_idx, 1)
        class_total = class_totals.get(class_label, 0)

        smoothed_prob = (count + 1) / (class_total + num_vals)
        log_feature_probs[key] = math.log(smoothed_prob)

    # --- Fallback log-prob for unseen (feature, value, class) combos ---
    # At prediction time, a test row may contain a feature value never seen
    # with a particular class during training (count = 0). The Laplace-smoothed
    # probability in that case is 1 / (class_total + |V_i|).
    # We precompute this per (feat_idx, class_label) to avoid recomputing in predict().
    fallback_log_probs = {}
    for feat_idx, num_vals in num_unique_vals.items():
        for class_label, class_total in class_totals.items():
            fallback_log_probs[(feat_idx, class_label)] = math.log(
                1 / (class_total + num_vals)
            )

    return {
        "classes":            list(class_counts.keys()),
        "log_class_probs":    log_class_probs,
        "log_feature_probs":  log_feature_probs,
        "fallback_log_probs": fallback_log_probs,
    }


def predict(log_prob_table, test_point):
    """
    Classify a single test row using the log-probability table from compute_log_probs().

    Why sum logs instead of multiplying probabilities?
        Multiplying many small floats underflows to 0.0, making all classes
        equally (im)probable. Summing their logs is mathematically equivalent
        — log(P_a * P_b * ... * P_n) = log(P_a) + log(P_b) + ... + log(P_n)
        — and stays in a numerically stable range. The argmax class is the same.

    Args:
        log_prob_table (dict): output of compute_log_probs().
        test_point     (list): feature values for one row, in feature-index order.
            Example: ["low", "med", "2", "2", "small", "low"]

    Returns:
        str: the predicted class label (class with the highest log-prob score).
    """
    classes            = log_prob_table["classes"]
    log_class_probs    = log_prob_table["log_class_probs"]
    log_feature_probs  = log_prob_table["log_feature_probs"]
    fallback_log_probs = log_prob_table["fallback_log_probs"]

    best_class = None
    best_score = float("-inf")

    for class_label in classes:
        # Start with the log prior for this class
        score = log_class_probs[class_label]

        # Add log likelihood for each feature value
        for feat_idx, value in enumerate(test_point):
            key = f"feat_{feat_idx}_{value}_{class_label}"

            if key in log_feature_probs:
                score += log_feature_probs[key]
            else:
                # (feature, value, class) was never seen in training.
                # Use the precomputed Laplace fallback instead of returning -inf.
                fallback_key = (feat_idx, class_label)
                score += fallback_log_probs.get(fallback_key, math.log(1e-10))

        if score > best_score:
            best_score = score
            best_class = class_label

    return best_class


def evaluate(predictions, true_labels):
    """
    Compute classification accuracy given parallel lists of predictions and true labels.

    Accuracy = (number of correct predictions) / (total predictions)

    When accuracy can mislead you:
        The car evaluation dataset is heavily imbalanced (~70% "unacc"). A naive
        classifier that always predicts "unacc" achieves ~70% accuracy without
        having learned anything. Our ~87% target is meaningfully above this
        majority-class baseline, but it's worth noting in the report.
        For a more complete picture, a per-class confusion matrix would show
        whether the model struggles on the minority classes (good, vgood).

    Args:
        predictions  (list): predicted class labels, one per test row.
        true_labels  (list): ground-truth labels, same order as predictions.

    Returns:
        float: accuracy in [0.0, 1.0].
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    return correct / len(predictions)
