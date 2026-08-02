import torch

from compose.expansion.checkpoint import load_sufficiency_checkpoint, save_sufficiency_checkpoint
from compose.expansion.sufficiency_head import SufficiencyHead


def test_sufficiency_checkpoint_round_trip(tmp_path):
    model = SufficiencyHead()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "head.pt"
    save_sufficiency_checkpoint(path, model, extra={"threshold": 0.5})
    for value in model.parameters(): value.data.zero_()
    assert load_sufficiency_checkpoint(path, model) == {"threshold": 0.5}
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
