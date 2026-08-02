"""Order-insensitive supervised losses for Empty/Single/Pair routing."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SetLossConfig:
    lambda_member: float = 1.0
    lambda_pair: float = 1.0
    lambda_retrieval: float = 0.5
    lambda_empty: float = 1.0


def set_router_loss(output, targets, expert_ids, config=SetLossConfig()):
    device = output.cardinality_logits.device
    cardinality_targets = torch.tensor([len(value) for value in targets], device=device)
    if torch.any(cardinality_targets > 2):
        raise ValueError("V6 Router supports at most two experts")
    cardinality = F.cross_entropy(output.cardinality_logits, cardinality_targets)
    member_terms, pair_terms, retrieval_terms = [], [], []
    for row, target_values in enumerate(targets):
        target = set(map(int, target_values))
        if not target:
            continue
        candidates = output.candidate_expert_ids[row]
        labels = torch.tensor([value in target for value in candidates], device=device, dtype=output.single_scores[row].dtype)
        if len(labels):
            member_terms.append(F.binary_cross_entropy_with_logits(output.single_scores[row], labels))
            retrieval_terms.append(1.0 - labels.mean())
        if len(target) == 2 and output.pair_ids[row]:
            canonical = tuple(sorted(target))
            pair_labels = torch.tensor([tuple(sorted(value)) == canonical for value in output.pair_ids[row]], device=device, dtype=output.pair_scores[row].dtype)
            pair_terms.append(F.binary_cross_entropy_with_logits(output.pair_scores[row], pair_labels))
    zero = output.cardinality_logits.sum() * 0.0
    member = torch.stack(member_terms).mean() if member_terms else zero
    pair = torch.stack(pair_terms).mean() if pair_terms else zero
    retrieval = torch.stack([value if isinstance(value, Tensor) else zero + value for value in retrieval_terms]).mean() if retrieval_terms else zero
    total = cardinality + config.lambda_member * member + config.lambda_pair * pair + config.lambda_retrieval * retrieval
    return total, {"cardinality": cardinality, "member": member, "pair": pair, "retrieval": retrieval}
