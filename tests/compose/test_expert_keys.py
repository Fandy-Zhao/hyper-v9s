import torch

from compose.experts import ExpertMetadata, ExpertRegistry
from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore


def metadata(expert_id):
    return ExpertKeyMetadata(expert_id, expert_id, "hash-{}".format(expert_id))


def test_keys_are_normalized_and_registry_bound():
    store = ExpertKeyStore([metadata(0), metadata(1)])
    assert torch.allclose(store.normalized().norm(dim=-1), torch.ones(2), atol=1e-6)
    registry = ExpertRegistry()
    for expert_id in range(2):
        registry.register(ExpertMetadata(expert_id, "adapter", creation_task=expert_id, checkpoint_sha256="hash-{}".format(expert_id)))
    store.validate_registry(registry)


def test_post_task_positive_mean_initialization():
    store = ExpertKeyStore([metadata(0)])
    queries = torch.zeros(2, 128)
    queries[:, 3] = 1
    assert store.initialize_from_queries(0, queries)
    assert store.normalized()[0].argmax().item() == 3
