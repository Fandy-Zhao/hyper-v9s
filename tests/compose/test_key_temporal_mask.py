import pytest

from compose.router.expert_keys import ExpertKeyMetadata, ExpertKeyStore
from compose.router.validation import validate_temporal_targets


def test_historical_visibility_excludes_current_and_future_keys():
    store = ExpertKeyStore([ExpertKeyMetadata(i, i, "h") for i in range(4)])
    assert store.visible_expert_ids(2, historical_only=True) == (0, 1)
    assert store.visible_expert_ids(2, historical_only=False) == (0, 1, 2)


def test_future_or_current_historical_target_is_rejected():
    with pytest.raises(ValueError, match="future"):
        validate_temporal_targets([[0, 2]], {0: 0, 2: 2}, [2])
