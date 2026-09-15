"""V9-S objective terms (spec §13, §14, §35).

    L_total = L_ans + lambda_key * L_key + lambda_sparse * L_sparse
              [ + lambda_budget * L_budget ]   # only if use_budget_loss

``L_key`` is the *only* term whose gradient reaches a routing key.  It is not
optional and it has no counterpart: the answer reaches a key through the
responsibility teacher and through nothing else (see ``compose.v9.router``).

``L_key`` supervises the **routing probabilities** with the answer-derived
responsibility.  V9-S pairs an independent sigmoid with a plain BCE, because a
sigmoid output is already a per-expert probability and BCE is its matching
proper scoring rule.  A normalised-divergence objective such as KL is defined on
a distribution, and an independent sigmoid is not one -- mixing the two silently
compares quantities that do not live on the same scale.

``L_sparse`` and ``L_budget`` shape *how many* experts the deployed Top-2 will
keep.  Early in training a row may sit at ``[0.4, 0.35, 0.3, 0.25]``; the
sparse term is what pulls it toward ``[0.91, 0.63, 0.02, 0.00]`` without ever
telling the method *which* expert should win -- that remains the answer's job.

Both are defined on the **routing row**, so neither is masked by
``contribution.valid``.  That row-level mask says "the answer could not rank
this sample", which is a statement about the *key* supervision and nothing
else.  Letting it reach these two made them vanish on exactly the early steps
whose routing is undecided -- the steps they exist for -- and vanish
*silently*, reporting ``0.0`` where a reader expects the pressure to be.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .config import V9LossConfig


#: Keeps ``log(p)`` finite at the sigmoid's saturated ends.  Far below any
#: probability a routing gate actually reaches, so it never shapes the fit.
BCE_FLOOR = 1e-6


@dataclass
class V9LossTerms:
    total: torch.Tensor
    answer: torch.Tensor
    key: torch.Tensor
    sparse: torch.Tensor
    budget: torch.Tensor
    active_mass: torch.Tensor

    def detached(self) -> dict:
        return {
            "loss_total": float(self.total.detach().item()),
            "loss_answer": float(self.answer.detach().item()),
            "loss_key": float(self.key.detach().item()),
            "loss_sparse": float(self.sparse.detach().item()),
            "loss_budget": float(self.budget.detach().item()),
            "mean_active_experts": float(self.active_mass.detach().mean().item()),
        }


def key_responsibility_loss(
    probabilities: torch.Tensor,
    responsibility: torch.Tensor,
    valid_rows: torch.Tensor,
    slot_mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """``BCE(route_probability, responsibility)`` on supervised slots only.

    ``responsibility`` is a detached teacher target.  Rows the answer could not
    rank (no positive contribution anywhere) are dropped rather than given an
    invented uniform or zero target; a zero target on such a row would push
    every key in it toward the same "never route here" answer, which is a
    statement the answer never made.
    """
    mask = valid_rows.unsqueeze(1) & slot_mask
    denominator = mask.sum().to(probabilities.dtype).clamp_min(1.0)
    clamped = probabilities.clamp(float(epsilon), 1.0 - float(epsilon))
    target = responsibility.detach()
    elementwise = -(
        target * clamped.log() + (1.0 - target) * (1.0 - clamped).log()
    )
    return (elementwise * mask.to(elementwise.dtype)).sum() / denominator


def _active_mass(
    probabilities: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """``sum_k a_ik`` per row.

    The sum is over the row as it stands, with padded slots contributing their
    zero gate.  Dividing by the number of live slots would make the sparse and
    budget penalties depend on the routing-row width, so the same ``lambda``
    would mean something different on task 0 than on task 5.
    """
    mask = slot_mask.to(probabilities.dtype)
    return (probabilities * mask).sum(dim=1)


def sparse_loss(
    probabilities: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """``mean(sum_k a_ik)`` -- the expected number of active experts."""
    return _active_mass(probabilities, slot_mask).mean()


def budget_loss(
    probabilities: torch.Tensor,
    slot_mask: torch.Tensor,
    budget: float,
) -> torch.Tensor:
    """``mean(relu(sum_k a_ik - B)^2)`` -- one-sided pressure toward the budget.

    One-sided on purpose: it never penalises a sample for routing to *fewer*
    experts than the budget, so a sample that genuinely needs one expert is not
    pushed to invent a second.
    """
    excess = (_active_mass(probabilities, slot_mask) - float(budget)).clamp_min(0.0)
    return excess.pow(2).mean()


def compose_total_loss(
    answer_loss: torch.Tensor,
    probabilities: torch.Tensor,
    responsibility: torch.Tensor,
    valid_rows: torch.Tensor,
    slot_mask: torch.Tensor,
    config: V9LossConfig,
) -> V9LossTerms:
    """Assemble the V9-S objective; ``answer_loss`` is already per-micro-batch.

    The budget term joins the sum only when ``use_budget_loss`` is set.  It is
    still *computed* either way so the logged value stays comparable across
    runs, but a term that is computed and not added contributes no gradient --
    which is the difference between reporting a quantity and optimising it.
    """
    key = key_responsibility_loss(
        probabilities, responsibility, valid_rows, slot_mask, epsilon=BCE_FLOOR
    )
    sparse = sparse_loss(probabilities, slot_mask)
    budget = budget_loss(probabilities, slot_mask, config.sparse_budget)
    total = (
        answer_loss
        + float(config.lambda_key) * key
        + float(config.lambda_sparse) * sparse
    )
    if config.use_budget_loss:
        total = total + float(config.lambda_budget) * budget
    with torch.no_grad():
        # Un-normalised, so ``mean_active_experts`` means the same thing here as
        # it does in ``V9RouteOutput.active_counts``: the expected number of
        # experts the deployed Top-2 will have to choose between.
        active_mass = _active_mass(probabilities.detach(), slot_mask)
    return V9LossTerms(
        total=total,
        answer=answer_loss,
        key=key,
        sparse=sparse,
        budget=budget,
        active_mass=active_mass,
    )


__all__ = [
    "V9LossTerms",
    "budget_loss",
    "compose_total_loss",
    "key_responsibility_loss",
    "sparse_loss",
]
