"""Set, cardinality, pair-risk, and temporal metrics."""

from collections import Counter


def set_routing_metrics(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("prediction/target length mismatch")
    canonical_pred = [tuple(sorted(map(int, row))) for row in predictions]
    canonical_true = [tuple(sorted(map(int, row))) for row in targets]
    exact = sum(a == b for a, b in zip(canonical_pred, canonical_true)) / max(1, len(targets))
    cardinality = sum(len(a) == len(b) for a, b in zip(canonical_pred, canonical_true)) / max(1, len(targets))
    output = {"exact_set_accuracy": exact, "cardinality_accuracy": cardinality}
    for name, size in (("Empty", 0), ("Single", 1), ("Pair", 2)):
        tp = sum(len(a) == size and len(b) == size for a, b in zip(canonical_pred, canonical_true))
        fp = sum(len(a) == size and len(b) != size for a, b in zip(canonical_pred, canonical_true))
        fn = sum(len(a) != size and len(b) == size for a, b in zip(canonical_pred, canonical_true))
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        output[name + "_precision"] = precision
        output[name + "_recall"] = recall
        output[name + "_f1"] = 2 * precision * recall / max(1e-12, precision + recall)
    output["predicted_route_rates"] = {str(size): sum(len(row) == size for row in canonical_pred) / max(1, len(canonical_pred)) for size in range(3)}
    output["average_active_experts"] = sum(map(len, canonical_pred)) / max(1, len(canonical_pred))
    output["single_to_pair_error_rate"] = sum(len(a) == 2 and len(b) == 1 for a, b in zip(canonical_pred, canonical_true)) / max(1, sum(len(b) == 1 for b in canonical_true))
    output["empty_to_pair_error_rate"] = sum(len(a) == 2 and len(b) == 0 for a, b in zip(canonical_pred, canonical_true)) / max(1, sum(len(b) == 0 for b in canonical_true))
    return output
