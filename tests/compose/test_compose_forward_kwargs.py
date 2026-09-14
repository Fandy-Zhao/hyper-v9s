"""The compose model must honour the loss contract the V7 trainer asks for.

``V7ComposeTrainer.compute_loss`` sets ``v7_sum_per_sample_loss`` on every step.
The compose model used to accept ``**kwargs`` and drop it, and -- because it
inherits ``LlamaForCausalLM`` rather than ``LlavaLlamaForCausalLM`` -- had no
per-sample branch to reach even had the flag survived.  The answer term stayed a
token-weighted batch mean while the key term was a batch sum, so the effective
key weight scaled with the micro-batch width.

These tests pin both halves: the flag is read when set, and it is not read when
absent -- the second matters because the flag must stay a no-op at batch size
one, which is the width every frozen baseline and production run used.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from compose.model.compose_llava import ComposeLlavaForCausalLM  # noqa: E402

CANNED_LOSS = 7.5


def _expected_sum_of_per_sample_means(logits, labels) -> float:
    """An independent re-implementation of the contract, one sample at a time."""
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    total = 0.0
    for row in range(shift_labels.shape[0]):
        valid = shift_labels[row].ne(-100)
        losses = F.cross_entropy(
            shift_logits[row], shift_labels[row], ignore_index=-100, reduction="none"
        )
        total += float(losses[valid].mean())
    return total


def _run(monkeypatch, labels, **extra):
    """Drive the real ``forward`` with a stubbed parent so no weights are needed."""
    torch.manual_seed(0)
    logits = torch.randn(labels.shape[0], labels.shape[1], 5)

    def fake_forward(self, **kwargs):
        return CausalLMOutputWithPast(loss=torch.tensor(CANNED_LOSS), logits=logits)

    monkeypatch.setattr(LlamaForCausalLM, "forward", fake_forward)
    model = object.__new__(ComposeLlavaForCausalLM)
    embeds = torch.zeros(labels.shape[0], labels.shape[1], 4)
    return model.forward(
        inputs_embeds=embeds, labels=labels, return_dict=True, **extra
    )


@pytest.fixture
def labels():
    # Three supervised tokens for sample 0, one for sample 1, so the
    # token-weighted mean and the mean of per-sample means genuinely differ.
    return torch.tensor([[1, 2, 3, 4], [1, -100, -100, 4]])


def test_flag_absent_leaves_the_model_loss_alone(monkeypatch, labels):
    assert float(_run(monkeypatch, labels).loss) == CANNED_LOSS


def test_flag_present_replaces_the_loss_with_the_per_sample_sum(monkeypatch, labels):
    out = _run(monkeypatch, labels, v7_sum_per_sample_loss=True)
    assert float(out.loss) == pytest.approx(
        _expected_sum_of_per_sample_means(out.logits, labels), rel=1e-5
    )
    assert float(out.loss) != CANNED_LOSS


def test_the_single_sample_case_is_left_to_the_model(monkeypatch):
    """At batch one a sum and a mean coincide, so the branch must not fire.

    This is what keeps the change invisible to every width-1 run, including the
    frozen baselines and the production V7/V8 runs.
    """
    one = torch.tensor([[1, 2, 3, 4]])
    out = _run(monkeypatch, one, v7_sum_per_sample_loss=True)
    assert float(out.loss) == CANNED_LOSS
