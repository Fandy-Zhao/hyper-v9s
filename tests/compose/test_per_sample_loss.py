"""The per-sample answer loss, and why its unit has to be a sum.

``V7ComposeTrainer`` divides this term by the micro-batch width so that the
objective does not depend on how the accumulation window was split.  That only
works if the helper returns a *sum* of per-sample means rather than a
token-weighted batch mean -- the two differ as soon as the batch's samples have
different numbers of supervised tokens, which is the normal case here.

The shared helper is exercised directly because it has two call sites now:
``LlavaLlamaForCausalLM.forward``, where the branch has always lived, and
``ComposeLlavaForCausalLM.forward``, where it was missing entirely.
"""

import os
import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from llava.model.language_model.llava_llama import (  # noqa: E402
    sum_of_per_sample_token_means,
)


def _case():
    """Logits and labels whose samples carry different supervised token counts."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    # sample 0: three supervised tokens, sample 1: one
    labels = torch.tensor([[1, 2, 3, 4], [1, -100, -100, 4]])
    return logits, labels


def _per_sample_means(logits, labels):
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    means = []
    for row in range(shift_labels.shape[0]):
        valid = shift_labels[row].ne(-100)
        losses = F.cross_entropy(
            shift_logits[row], shift_labels[row], ignore_index=-100, reduction="none"
        )
        means.append(float(losses[valid].mean()))
    return means


def test_it_sums_the_per_sample_means():
    logits, labels = _case()
    value = float(sum_of_per_sample_token_means(logits, labels))
    per_sample = _per_sample_means(logits, labels)
    assert value == pytest.approx(sum(per_sample), rel=1e-5)
    assert value == pytest.approx(len(per_sample) * statistics.mean(per_sample), rel=1e-5)


def test_it_is_not_the_token_weighted_batch_mean():
    """The distinction the micro-batch fix rests on.

    A token-weighted mean over the batch averages the two samples' tokens
    together; the sum of per-sample means does not.  With three supervised
    tokens against one, the two agree only by coincidence.
    """
    logits, labels = _case()
    value = float(sum_of_per_sample_token_means(logits, labels))
    token_weighted = float(
        F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
    )
    assert abs(value - token_weighted) > 1e-3


def test_one_sample_has_no_scaling():
    """At batch one the sum has a single term, so dividing by the width is a no-op."""
    logits, labels = _case()
    single = float(sum_of_per_sample_token_means(logits[:1], labels[:1]))
    assert single == pytest.approx(_per_sample_means(logits[:1], labels[:1])[0], rel=1e-6)


def test_a_sample_with_no_supervised_tokens_is_refused():
    """A sample with nothing to learn from would otherwise vanish silently."""
    logits, labels = _case()
    labels[1, 1:] = -100
    with pytest.raises(ValueError, match="at least one supervised answer token"):
        sum_of_per_sample_token_means(logits, labels)
