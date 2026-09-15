"""The compose model must honour the loss contract the V7 trainer asks for.

``V7ComposeTrainer.compute_loss`` sets ``v7_sum_per_sample_loss`` on every step.
The compose model used to accept ``**kwargs`` and drop it, and -- because it
inherits ``LlamaForCausalLM`` rather than ``LlavaLlamaForCausalLM`` -- had no
per-sample branch to reach even had the flag survived.  The answer term stayed a
token-weighted batch mean while the key term was a batch sum, so the effective
key weight scaled with the micro-batch width.

The flag makes the parent run without labels, so from that point on this branch
is the only thing that can put a loss on the output -- and it has to put it
where the HF trainer looks.  ``ModelOutput.__setattr__`` mirrors a value into the
mapping only for a field that is *already* present, so an attribute write reads
back correctly through ``out.loss`` while leaving ``"loss" not in out`` true and
the step rejected.  The stub parent therefore drops its loss when it is called
without labels: a stub that always returns one hides that integration failure
instead of reproducing it.
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
        # Mirror the real parent: ``LlamaForCausalLM`` returns no loss at all
        # when it is called without labels, which is precisely the case the V8
        # flag creates.  A stub that always manufactures a loss hides the
        # integration failure these tests exist to catch.
        loss = torch.tensor(CANNED_LOSS) if kwargs.get("labels") is not None else None
        return CausalLMOutputWithPast(loss=loss, logits=logits)

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
    out = _run(monkeypatch, labels)
    assert "loss" in out
    assert float(out.loss) == CANNED_LOSS


def test_flag_present_replaces_the_loss_with_the_per_sample_sum(monkeypatch, labels):
    out = _run(monkeypatch, labels, v7_sum_per_sample_loss=True)
    # The parent was called with ``labels=None``, so this loss exists only
    # because the branch wrote it.  Membership is asserted, not just the
    # attribute: ``ModelOutput`` keeps a value that was assigned as an
    # attribute out of the mapping, and the HF trainer reads the mapping.
    assert "loss" in out
    assert float(out.loss) == pytest.approx(
        _expected_sum_of_per_sample_means(out.logits, labels), rel=1e-5
    )
    assert float(out.loss) != CANNED_LOSS


def test_the_single_sample_case_takes_the_same_path_as_any_other_width(monkeypatch):
    """Width one must take the per-sample branch too, or the step carries no loss.

    The parent is called with ``labels=None`` at every width now -- that is what
    removes the second cross-entropy rather than merely reordering it -- so a
    width-one batch that skipped this branch would come back with neither a
    mapping entry nor a value.  The number is the one the parent's own CE would
    have produced, since a sum of one mean is that mean: that is what keeps the
    change invisible to every width-1 run.
    """
    one = torch.tensor([[1, 2, 3, 4]])
    out = _run(monkeypatch, one, v7_sum_per_sample_loss=True)
    assert "loss" in out
    assert float(out.loss) == pytest.approx(
        _expected_sum_of_per_sample_means(out.logits, one), rel=1e-5
    )
    assert float(out.loss) != CANNED_LOSS


def test_the_trainer_fetches_the_outputs_the_quality_weight_needs():
    """``compute_loss`` requests the outputs for itself, not for its caller.

    The reused-key quality weight is read off the model's per-sample answer NLL,
    which exists only on the output object, while both callers -- HF's
    ``training_step`` and ``V7ComposeTrainer._ddp_training_step`` -- ask for a
    scalar.  Forwarding the caller's ``return_outputs`` through to the pinned HF
    ``compute_loss`` leaves ``outputs`` as ``None`` and fails the first step of a
    formal V8 run with "formal V8 requires per-sample routed answer NLL".  The
    caller's own contract is re-applied on the way out instead.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("compose/v7/hf_trainer.py").read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    delegations = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compute_loss"
    ]
    assert len(delegations) == 1, "expected one delegation to the pinned HF compute_loss"
    requested = {keyword.arg: keyword.value for keyword in delegations[0].keywords}
    assert isinstance(requested.get("return_outputs"), ast.Constant) and (
        requested["return_outputs"].value is True
    ), "the trainer must ask for the outputs regardless of what its caller asked for"
