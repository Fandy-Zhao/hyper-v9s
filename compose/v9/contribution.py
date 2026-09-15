"""Answer-derived responsibility (spec §10, §11, §12).

The central claim of V9 is that the ground-truth answer can *judge* an expert's
participation without ever enumerating experts or pairs.  Writing the answer
loss as a function of the differentiable gates, the first-order effect of
scaling expert ``k``'s contribution on sample ``i`` is

    dL_ans/da_ik        (one ``torch.autograd.grad`` against the gate tensor)
    G_ik = -a_ik * dL_ans/da_ik   (stop-gradient)

so ``G_ik > 0`` means *increasing this expert's participation would lower the
answer loss on this sample*, ``G_ik ~ 0`` means weak or uncertain, and
``G_ik < 0`` means the current combination is being hurt by it.

**This is a local conditional contribution estimate, not a marginal
contribution.**  It is the directional derivative of the loss along the gate at
the composition point the forward pass actually visited, in the presence of the
other active experts.  It is not ``L(S \\ E_k) - L(S)``, it is not an oracle,
and nothing in this module or in the logs may call it one.  The exact
remove-and-reroute quantity is computed only in
:func:`exact_removal_contribution`, on a small held-out sample, purely to check
that the local estimate is a usable proxy (spec §30).

Two implementation rules are load-bearing:

* the contribution teacher is **stop-gradient** -- ``create_graph=False`` -- so
  no second-order graph is ever built (spec §11);
* samples whose positive contributions all vanish produce **no teacher at all**.
  They still contribute to ``L_ans``; they are simply excluded from the
  responsibility loss rather than being handed an invented target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class V9Contribution:
    """One micro-step's local contribution estimate and its responsibility."""

    #: ``G_ik = -a_ik * dL_ans/da_ik``, detached.  ``[B, S]``.
    raw: torch.Tensor
    #: ``max(G_ik, 0)``.  ``[B, S]``.
    positive: torch.Tensor
    #: ``G_pos / (sum_j G_pos_ij + eps)``.  ``[B, S]``.
    responsibility: torch.Tensor
    #: ``sum_j G_pos_ij > eps`` -- the rows that may supervise the keys.
    valid: torch.BoolTensor

    @property
    def valid_rate(self) -> float:
        if self.valid.numel() == 0:
            return 0.0
        return float(self.valid.to(torch.float32).mean().item())


def gate_gradient(
    loss: torch.Tensor,
    gates: torch.Tensor,
    retain_graph: bool = True,
) -> torch.Tensor:
    """``d loss / d gates`` with no second-order graph.

    ``create_graph`` is deliberately False: the result is a *teacher target*, so
    differentiating through it would buy nothing and cost a full second-order
    graph (spec §11 forbids it in V9 v1).

    A gate that does not require grad, or that the loss does not depend on, is
    **not** reported as a zero gradient.  A zero is a legitimate mathematical
    answer, but here it is never the right one: it means the composition never
    consumed this tensor, so the answer made no statement about any expert and
    the responsibility teacher is empty.  Returning ``0`` would let a run train
    for its full length with no key supervision at all while every logged
    quantity read a clean ``0.0`` -- which is what happened once, and cost a
    preflight to find.
    """
    if not gates.requires_grad:
        raise RuntimeError(
            "gate_gradient: the gate does not require grad, so the answer loss "
            "is a constant in it and no contribution can be measured. Build the "
            "gate from detached inputs and mark it a leaf with "
            "requires_grad_(True) (see compose.v9.router.V9Router.route)."
        )
    gradient = torch.autograd.grad(
        outputs=loss,
        inputs=gates,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        raise RuntimeError(
            "gate_gradient: the loss does not depend on the gate, so the graph "
            "is disconnected from the composition. No responsibility teacher "
            "exists for this step; this is a wiring bug, not a zero."
        )
    return gradient


def local_conditional_contribution(
    loss_ans: torch.Tensor,
    gates: torch.Tensor,
    retain_graph: bool = True,
    values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``G_ik = -a_ik * dL_ans/da_ik`` (spec §10), detached on both factors.

    ``gates`` is the differentiable tensor the derivative is taken against;
    ``values`` is the participation the *forward pass actually used*, which is
    the gate the ``a_ik`` factor must be.  The two coincide in the soft stage and
    differ in the other two: the bootstrap stage floors every gate, and the
    discretisation stage drives the forward with the Top-2 indicator while the
    backward keeps the straight-through surrogate.  Multiplying by the surrogate
    there would credit an expert with a contribution the deployed forward never
    gave it.
    """
    gradient = gate_gradient(loss_ans, gates, retain_graph=retain_graph)
    participation = gates.detach() if values is None else values.detach()
    return -(participation * gradient.detach())


def answer_derived_responsibility(
    contribution: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
) -> V9Contribution:
    """Turn local contributions into a per-sample responsibility distribution.

    Only positive contributions survive: a negative estimate says "this
    combination is worse with this expert", which is a statement about the
    *current mixture*, not a licence to push the expert's key away from a query
    it may genuinely serve.  Rows whose positives are all below ``epsilon`` get
    ``valid=False`` and are excluded from key supervision.
    """
    positive = contribution.clamp_min(0.0)
    if valid is not None:
        positive = positive * valid.to(positive.dtype)
    total = positive.sum(dim=1, keepdim=True)
    responsibility = positive / (total + float(epsilon))
    row_template = total.squeeze(1)
    rows = torch.ones_like(row_template, dtype=torch.bool)
    if valid is not None:
        # A per-slot mask is accepted and reduced, but the decision being made
        # here is per *row*: "did the answer rank anything for this sample".
        rows = valid.any(dim=1) if valid.ndim == 2 else valid
        if rows.shape != row_template.shape:
            raise ValueError(
                "valid mask has shape {} for {} rows".format(
                    tuple(rows.shape), tuple(row_template.shape)
                )
            )
    rows = rows & (row_template > float(epsilon))
    responsibility = torch.where(
        rows.unsqueeze(1), responsibility, torch.zeros_like(responsibility)
    )
    return V9Contribution(
        raw=contribution,
        positive=positive,
        responsibility=responsibility,
        valid=rows,
    )


def contribution_statistics(
    contribution: torch.Tensor,
    responsibility: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Aggregate scalars for the step log -- never per-sample routing dumps.

    ``contribution`` and ``responsibility`` are both required, and deliberately
    so.  They answer different questions -- the contribution says whether an
    expert helped, the responsibility says how much of the row's credit it was
    given -- and an earlier version of this function derived the second from the
    first, so ``responsibility_mean`` was a second name for
    ``mean_positive * positive_rate``.  A metric that cannot disagree with the
    one beside it is not a measurement.
    """
    values = contribution.detach()
    shares = responsibility.detach()
    if valid is not None:
        values = values[valid]
        shares = shares[valid]
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "mean_positive": 0.0,
            "mean_negative": 0.0,
            "positive_rate": 0.0,
            "std": 0.0,
            "responsibility_mean": 0.0,
            "responsibility_max": 0.0,
        }
    positives = values[values > 0]
    negatives = values[values < 0]
    return {
        "mean": float(values.mean().item()),
        "mean_positive": float(positives.mean().item()) if positives.numel() else 0.0,
        "mean_negative": float(negatives.mean().item()) if negatives.numel() else 0.0,
        "positive_rate": float((values > 0).to(torch.float32).mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        # ``r`` is the teacher the keys are fitted to, so its mean and its
        # maximum are the two numbers that say whether the answer is committing
        # to an expert or splitting the credit evenly.
        "responsibility_mean": float(shares.mean().item()),
        "responsibility_max": float(shares.max().item()),
    }


# ----------------------------------------------------------------------
# Exact remove-and-reroute, for calibration only (spec §30)
# ----------------------------------------------------------------------
def exact_removal_contribution(
    loss_with: torch.Tensor,
    loss_without: torch.Tensor,
) -> torch.Tensor:
    """``G_exact(k) = L(S \\ E_k) - L(S)`` for one sample and one expert.

    Both losses must be *per-sample* answer NLLs measured under the same
    scoring rule.  This is only ever used on a handful of held-out samples to
    check the gate-gradient proxy; it must never appear in the training loop.
    """
    return (loss_without - loss_with).detach()


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    a = left - left.mean()
    b = right - right.mean()
    denominator = float(a.norm() * b.norm())
    if denominator <= 0:
        return 0.0
    return float((a @ b).item() / denominator)


def calibration_report(
    grad_contribution: torch.Tensor,
    exact_contribution: torch.Tensor,
    top_k: int = 2,
) -> Dict[str, float]:
    """Correlation between the local estimate and the exact removal effect.

    Reported once per validation pass.  If the two are uncorrelated, the
    gate-gradient proxy is not measuring what the method assumes and the run
    should stop rather than train for days on a broken signal.
    """
    g = grad_contribution.detach().float().reshape(-1)
    e = exact_contribution.detach().float().reshape(-1)
    if g.numel() != e.numel() or g.numel() == 0:
        raise ValueError("calibration needs two equally sized non-empty vectors")
    k = min(int(top_k), g.numel())
    report = {
        "samples": int(g.numel()),
        "pearson": _pearson(g, e),
        "spearman": _pearson(_rank(g), _rank(e)),
        "sign_agreement": float(((g > 0) == (e > 0)).to(torch.float32).mean().item()),
        "top1_agreement": float(
            (g.argmax() == e.argmax()).to(torch.float32).item()
        ),
        "grad_mean": float(g.mean().item()),
        "exact_mean": float(e.mean().item()),
    }
    if k > 0:
        grad_top = torch.topk(g, k).indices
        exact_top = torch.topk(e, k).indices
        report["topk_recall"] = float(
            len(set(grad_top.tolist()) & set(exact_top.tolist())) / k
        )
    return report


def pair_rerank_report(
    deployed_loss: torch.Tensor,
    pair_losses: torch.Tensor,
    pair_is_deployed: torch.Tensor,
) -> Dict[str, float]:
    """Does reranking the *pairs* of experts buy anything? (spec §35)

    ``deployed_loss`` is the per-sample answer NLL of the pair the gate score
    actually selects; ``pair_losses[b, p]`` is the same sample's NLL when only
    challenger pair ``p`` is routed.  A negative ``mean_regret`` would mean the
    gate ranking is systematically missing a better pair -- which is the only
    reason V9 would ever need V8's pair enumeration in the loop.

    ``pair_is_deployed`` is the **indicator** the probe emits per sample: 1.0
    when the pair in column 0 is the pair that sample's deployed rule serves,
    0.0 when it is not.  It is not an index, and reading it as one is a mistake
    this function made: comparing the arg-min *position* against a 1.0 flag
    reports ``0/N`` for a run in which column 0 was served on every sample, and
    reports a spurious success for a sample whose deployed pair was never
    measured.  The rate below is therefore computed on the comparable subset --
    samples where the deployed pair is one of the measured ones -- and the
    denominator is reported with it, because a rate over a subset the reader
    cannot see is the same defect one level up.

    The main experiment declares ``pair_rerank: false``; this makes that a
    measured statement rather than an assumption, and it is the whole reason the
    flag exists.  It runs on the bounded calibration sample and never in
    training.
    """
    deployed = deployed_loss.detach().float().reshape(-1)
    alternatives = pair_losses.detach().float()
    if alternatives.ndim != 2 or alternatives.shape[0] != deployed.numel():
        raise ValueError(
            "pair_rerank_report expects [{}, P] pair losses".format(deployed.numel())
        )
    deployed = deployed.unsqueeze(1)
    best = alternatives.argmin(dim=1)
    # Per sample, not per element: "how much did the gate ranking lose on this
    # sample" is a per-sample question, and averaging over the pair axis would
    # dilute one bad sample by however many pairs were measured.
    regret = (deployed - alternatives.min(dim=1, keepdim=True).values).clamp_min(0)
    served = pair_is_deployed.detach().reshape(-1).to(torch.float32)
    if served.numel() != deployed.numel():
        raise ValueError(
            "pair_rerank_report expects {} deployed flags".format(deployed.numel())
        )
    comparable = served.gt(0)
    # Column 0 is the pair with the largest summed gate mass -- the pair the
    # gate ranking itself names -- so "the best measured pair is the deployed
    # one" is ``argmin == 0``, evaluated where the deployed pair was measured.
    best_is_column_zero = best.eq(0)
    count = int(comparable.sum().item())
    if count:
        rate = float(best_is_column_zero[comparable].to(torch.float32).mean().item())
    else:
        # Nothing was measurable.  Reporting 0.0 would read as "the gate never
        # deploys the best pair", which is a claim, not an absence.
        rate = float("nan")
    return {
        "samples": int(deployed.numel()),
        "comparable_samples": count,
        "pairs_measured": int(alternatives.shape[1]),
        "mean_deployed_loss": float(deployed.mean().item()),
        "mean_best_pair_loss": float(alternatives.min(dim=1).values.mean().item()),
        "mean_regret": float(regret.mean().item()),
        "best_pair_is_deployed_rate": rate,
        "deployed_beats_mean_pair_rate": float(
            (deployed < alternatives.mean(dim=1, keepdim=True))
            .to(torch.float32)
            .mean()
            .item()
        ),
    }


__all__ = [
    "V9Contribution",
    "answer_derived_responsibility",
    "calibration_report",
    "contribution_statistics",
    "exact_removal_contribution",
    "gate_gradient",
    "local_conditional_contribution",
    "pair_rerank_report",
]
