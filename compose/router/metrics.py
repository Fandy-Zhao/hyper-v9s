"""Stage 05 retrieval metrics with explicit Empty/Single/Pair accounting."""

from collections import Counter
from typing import Dict, Sequence


def retrieval_metrics(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], ks=(1, 2, 4)) -> Dict[str, object]:
    if len(predictions) != len(targets):
        raise ValueError("prediction/target length mismatch")
    output = {}
    nonempty = [index for index, target in enumerate(targets) if target]
    for k in ks:
        recalls = []
        set_hits = []
        for index in nonempty:
            predicted = set(predictions[index][:k])
            target = set(targets[index])
            recalls.append(len(predicted & target) / len(target))
            set_hits.append(target.issubset(predicted))
        output[f"OracleMemberRecall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        output[f"OracleSetRecall@{k}"] = sum(set_hits) / len(set_hits) if set_hits else 0.0
    single = [i for i, target in enumerate(targets) if len(target) == 1]
    pair = [i for i, target in enumerate(targets) if len(target) == 2]
    empty = [i for i, target in enumerate(targets) if not target]
    output["SingleRecall@1"] = sum(targets[i][0] in predictions[i][:1] for i in single) / len(single) if single else 0.0
    output["PairBothMembersRecall@4"] = sum(set(targets[i]).issubset(predictions[i][:4]) for i in pair) / len(pair) if pair else 0.0
    output["PairAtLeastOneRecall@4"] = sum(bool(set(targets[i]) & set(predictions[i][:4])) for i in pair) / len(pair) if pair else 0.0
    output["EmptyFalseRecallRate"] = sum(bool(predictions[i]) for i in empty) / len(empty) if empty else 0.0
    reciprocal = []
    for i in nonempty:
        ranks = [predictions[i].index(value) + 1 for value in targets[i] if value in predictions[i]]
        reciprocal.append(1.0 / min(ranks) if ranks else 0.0)
    output["MRR"] = sum(reciprocal) / len(reciprocal) if reciprocal else 0.0
    output["expert_selection_histogram"] = dict(Counter(value for row in predictions for value in row))
    return output
