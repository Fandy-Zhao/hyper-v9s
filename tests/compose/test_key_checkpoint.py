import torch

from compose.router.checkpoint import load_router_checkpoint, save_router_checkpoint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.query_encoder import MultimodalQueryEncoder


def test_query_key_checkpoint_round_trip(tmp_path):
    encoder = MultimodalQueryEncoder(3, 5)
    store = ExpertKeyStore([ExpertKeyMetadata(0, 0, "abc")])
    original = {name: value.clone() for name, value in encoder.state_dict().items()}
    path = tmp_path / "router.pt"
    save_router_checkpoint(path, encoder, store, extra={"mode": "continual_anchor"})
    for value in encoder.parameters():
        value.data.zero_()
    assert load_router_checkpoint(path, encoder, store) == {"mode": "continual_anchor"}
    assert all(torch.equal(encoder.state_dict()[name], value) for name, value in original.items())
