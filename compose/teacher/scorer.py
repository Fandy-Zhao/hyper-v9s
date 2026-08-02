"""Teacher-forced answer-token scoring with an explicit autoregressive shift."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def answer_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_token_id: Optional[int] = None,
    include_eos: bool = False,
    return_per_token: bool = False,
) -> Dict[str, torch.Tensor]:
    """Score labels[t+1] from logits[t], never prompt or masked tokens."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected aligned logits [B,T,V] and labels [B,T]")
    if not include_eos and eos_token_id is None:
        raise ValueError("eos_token_id is required when EOS is excluded")
    shifted_logits = logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(IGNORE_INDEX)
    if not include_eos:
        valid = valid & shifted_labels.ne(int(eos_token_id))
    counts = valid.sum(dim=1)
    invalid = torch.where(counts.eq(0))[0]
    if invalid.numel():
        raise ValueError("samples have zero answer tokens after shift/mask: {}".format(invalid.cpu().tolist()))
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        safe_labels.reshape(-1), reduction="none",
    ).reshape_as(shifted_labels)
    masked = losses * valid
    sums = masked.sum(dim=1)
    predictions = shifted_logits.argmax(dim=-1)
    exact = ((predictions.eq(safe_labels) | ~valid).all(dim=1))
    result = {
        "sum_nll": sums,
        "mean_nll": sums / counts.float(),
        "token_count": counts,
        "exact_teacher_forced": exact,
    }
    if return_per_token:
        result["per_token_nll"] = tuple(masked[index][valid[index]].detach().cpu() for index in range(labels.shape[0]))
    return result
