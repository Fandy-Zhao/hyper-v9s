from typing import Dict, Union

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def compute_per_sample_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    return_details: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions must match")
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(IGNORE_INDEX)
    valid_token_count = valid.sum(dim=1)
    zero = torch.where(valid_token_count.eq(0))[0]
    if zero.numel():
        raise ValueError(
            "samples have zero target tokens after shifting: {}".format(
                zero.detach().cpu().tolist()
            )
        )
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    loss_sum = (token_losses * valid).sum(dim=1)
    mean_nll = loss_sum / valid_token_count.float()
    if return_details:
        return {
            "loss_sum": loss_sum,
            "valid_token_count": valid_token_count,
            "mean_nll": mean_nll,
        }
    return mean_nll
