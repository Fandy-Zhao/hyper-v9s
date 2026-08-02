import torch

from compose.router.query_encoder import MultimodalQueryEncoder, QueryInputs


def test_query_encoder_returns_normalized_128d_queries():
    torch.manual_seed(42)
    model = MultimodalQueryEncoder(4, 6)
    values = model(QueryInputs(torch.randn(3, 4), torch.randn(3, 6), torch.tensor([1, 1, 0]), torch.ones(3)))
    assert values.shape == (3, 128)
    assert torch.allclose(values.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_missing_modality_is_masked_not_read():
    model = MultimodalQueryEncoder(2, 2)
    common = dict(text_features=torch.ones(1, 2), image_available=torch.zeros(1), text_available=torch.ones(1))
    a = model(QueryInputs(image_features=torch.zeros(1, 2), **common))
    b = model(QueryInputs(image_features=torch.full((1, 2), 9999.0), **common))
    assert torch.equal(a, b)
