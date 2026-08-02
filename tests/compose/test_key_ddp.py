from compose.router.checkpoint import state_fingerprint
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore


def test_identical_seed_has_identical_ddp_state_fingerprint():
    metadata = [ExpertKeyMetadata(0, 0, "h0"), ExpertKeyMetadata(1, 1, "h1")]
    assert state_fingerprint(ExpertKeyStore(metadata, seed=42)) == state_fingerprint(ExpertKeyStore(metadata, seed=42))
