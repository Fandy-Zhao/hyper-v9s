import torch

from compose.router.set_router import ExpertSetRouter


def test_set_router_state_round_trip(tmp_path):
    source = ExpertSetRouter()
    path = tmp_path / "set_router.pt"
    torch.save(source.state_dict(), path)
    target = ExpertSetRouter()
    target.load_state_dict(torch.load(path))
    assert all(torch.equal(value, target.state_dict()[name]) for name, value in source.state_dict().items())
