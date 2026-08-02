"""Dependency-light binary sufficiency metrics and calibration."""


def _auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.0
    wins = sum(p > n for p in positives for n in negatives) + 0.5 * sum(p == n for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def sufficiency_metrics(labels, probabilities, threshold=0.5, bins=10):
    predictions = [value >= threshold for value in probabilities]
    tp = sum(a and b for a, b in zip(predictions, labels))
    fp = sum(a and not b for a, b in zip(predictions, labels))
    fn = sum(not a and b for a, b in zip(predictions, labels))
    tn = sum(not a and not b for a, b in zip(predictions, labels))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    ece = 0.0
    for index in range(bins):
        members = [i for i, value in enumerate(probabilities) if index / bins <= value < (index + 1) / bins or (index == bins - 1 and value == 1.0)]
        if members:
            ece += len(members) / len(labels) * abs(sum(probabilities[i] for i in members) / len(members) - sum(labels[i] for i in members) / len(members))
    ranked = sorted(zip(probabilities, labels), reverse=True)
    precisions, seen = [], 0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            seen += 1
            precisions.append(seen / rank)
    return {
        "accuracy": (tp + tn) / max(1, len(labels)), "auroc": _auc(labels, probabilities),
        "auprc": sum(precisions) / max(1, sum(labels)), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_sufficient_rate": fp / max(1, fp + tn), "false_insufficient_rate": fn / max(1, fn + tp), "ece": ece,
    }
