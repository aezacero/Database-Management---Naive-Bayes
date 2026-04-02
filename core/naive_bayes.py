import math
from collections import defaultdict


def compute_log_probs(class_counts, feature_counts, class_totals):
    '''
    Compute log probabilities for the Naive Bayes classifier based on training counts.
    '''

    total_samples = sum(class_counts.values())
    num_classes   = len(class_counts)

    # Log prior 
    log_class_probs = {
        label: math.log((count + 1) / (total_samples + num_classes))
        for label, count in class_counts.items()
    }

    # To compute log likelihoods, we need to know how many unique values each feature takes on in the training data.
    unique_vals_per_feature = defaultdict(set)
    for key in feature_counts:
        parts    = key.split("_")
        feat_idx = int(parts[1])
        # The value may contain underscores, so we join all parts between the feature index and the class label.
        value    = "_".join(parts[2:-1])
        unique_vals_per_feature[feat_idx].add(value)

    num_unique_vals = {idx: len(vals) for idx, vals in unique_vals_per_feature.items()}

    # Log likelihoods with Laplace smoothing: P(feature=value | class) = (count + 1) / (class_total + |V_i|)
    log_feature_probs = {}
    for key, count in feature_counts.items():
        parts       = key.split("_")
        feat_idx    = int(parts[1])
        class_label = parts[-1]
        num_vals    = num_unique_vals.get(feat_idx, 1)
        class_total = class_totals.get(class_label, 0)

        smoothed_prob = (count + 1) / (class_total + num_vals)
        log_feature_probs[key] = math.log(smoothed_prob)

    # Precompute fallback log probabilities for unseen feature values: log(1 / (class_total + num_vals))
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
    '''
    Predict the class label for a single test point using the precomputed log probabilities.
    '''

    classes            = log_prob_table["classes"]
    log_class_probs    = log_prob_table["log_class_probs"]
    log_feature_probs  = log_prob_table["log_feature_probs"]
    fallback_log_probs = log_prob_table["fallback_log_probs"]

    best_class = None
    best_score = float("-inf")

    for class_label in classes:

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
    '''
    Compute accuracy of predictions against true labels.
    '''

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    return correct / len(predictions)
