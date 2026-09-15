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

The per-sample NLL the trainer folds into the reuse-quality weight travels on
the *module*, not on the output object, for the same class of reason.  The
plumbing between the two rebuilds the output from its mapping
(``type(out)(**out)``): the dataclass fields come through, every other attribute
does not.  A per-sample NLL parked on the output is therefore already gone by
the time the trainer reads it, and the step dies with "formal V8 requires
per-sample routed answer NLL" while ``loss`` -- the very same per-sample sum --
arrives intact.
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


def _expected_per_sample_means(logits, labels):
    """An independent re-implementation of the contract, one sample at a time."""
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


def _expected_sum_of_per_sample_means(logits, labels) -> float:
    return sum(_expected_per_sample_means(logits, labels))


def _run(monkeypatch, labels, **extra):
    """Drive the real ``forward`` with a stubbed parent so no weights are needed.

    Returns the model as well as its output: the per-sample NLL is a module
    attribute, so a test that only kept the output could not see it.
    """
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
    out = model.forward(
        inputs_embeds=embeds, labels=labels, return_dict=True, **extra
    )
    return model, out


@pytest.fixture
def labels():
    # Three supervised tokens for sample 0, one for sample 1, so the
    # token-weighted mean and the mean of per-sample means genuinely differ.
    return torch.tensor([[1, 2, 3, 4], [1, -100, -100, 4]])


def test_flag_absent_leaves_the_model_loss_alone(monkeypatch, labels):
    _, out = _run(monkeypatch, labels)
    assert "loss" in out
    assert float(out.loss) == CANNED_LOSS


def test_flag_present_replaces_the_loss_with_the_per_sample_sum(monkeypatch, labels):
    _, out = _run(monkeypatch, labels, v7_sum_per_sample_loss=True)
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
    _, out = _run(monkeypatch, one, v7_sum_per_sample_loss=True)
    assert "loss" in out
    assert float(out.loss) == pytest.approx(
        _expected_sum_of_per_sample_means(out.logits, one), rel=1e-5
    )
    assert float(out.loss) != CANNED_LOSS


def test_the_per_sample_nll_survives_the_output_being_rebuilt(monkeypatch, labels):
    """The trainer's channel has to outlive the plumbing's rebuild of the output.

    ``type(out)(**out)`` is what the training plumbing does to a model output on
    its way from the model to the trainer, and it is lossy in exactly one way:
    the dataclass fields are carried over and every other attribute is dropped.
    A per-sample NLL parked on the output therefore never reaches the trainer --
    the first step of a formal V8 run fails with "formal V8 requires per-sample
    routed answer NLL" while the loss it wants is sitting right there in the
    mapping.  Carrying it on the module is what makes the pair survive.
    """
    model, out = _run(monkeypatch, labels, v7_sum_per_sample_loss=True)

    rebuilt = type(out)(**out)

    assert not hasattr(rebuilt, "v7_per_sample_answer_nll")
    assert float(model.v7_per_sample_answer_nll.sum()) == pytest.approx(
        _expected_sum_of_per_sample_means(out.logits, labels), rel=1e-5
    )
    assert [
        float(value) for value in model.v7_per_sample_answer_nll
    ] == pytest.approx(_expected_per_sample_means(out.logits, labels), rel=1e-5)


def test_the_module_holds_no_nll_when_the_flag_is_absent(monkeypatch, labels):
    """A stale value must not be able to satisfy the formal-V8 check.

    The trainer reads the attribute straight after its own model call, so
    clearing it on entry is what makes "absent" mean "this forward did not
    produce one" rather than "some earlier step did".
    """
    model, _ = _run(monkeypatch, labels)
    assert model.v7_per_sample_answer_nll is None


def test_the_trainer_reads_the_per_sample_nll_off_the_module():
    """The read must not go through the output object.

    An AST check rather than a behavioural one because the failure needs the
    plumbing in the loop: the value is present on the output at the moment the
    model returns it, and absent by the moment the trainer looks.  What is
    checked here is the channel the trainer committed to, and the channel it
    must not use again.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("compose/v7/hf_trainer.py").read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    reads = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "v7_per_sample_answer_nll"
    ]
    assert len(reads) == 1, "expected exactly one read of the per-sample NLL"
    source = ast.unparse(reads[0].args[0])
    assert "outputs" not in source, (
        "the per-sample NLL must be read off the module, not off the output "
        "object the plumbing rebuilds"
    )
