"""Frozen validation-selected 0/1/2 set inference."""

from typing import List, Tuple

import torch


def predict_sets(output, pair_threshold: float = 0.65, single_only: bool = False) -> List[Tuple[int, ...]]:
    if pair_threshold not in (0.50, 0.65, 0.80):
        raise ValueError("pair threshold must be one of the preregistered candidates")
    cardinalities = output.cardinality_logits.argmax(dim=-1).tolist()
    predictions = []
    for row, cardinality in enumerate(cardinalities):
        if cardinality == 0 or not output.candidate_expert_ids[row]:
            predictions.append(())
        elif cardinality == 1 or single_only or not output.pair_ids[row]:
            best = int(output.single_scores[row].argmax())
            predictions.append((output.candidate_expert_ids[row][best],))
        else:
            probabilities = torch.sigmoid(output.pair_scores[row])
            best = int(probabilities.argmax())
            if float(probabilities[best]) >= pair_threshold:
                predictions.append(tuple(output.pair_ids[row][best]))
            else:
                single = int(output.single_scores[row].argmax())
                predictions.append((output.candidate_expert_ids[row][single],))
    return predictions
